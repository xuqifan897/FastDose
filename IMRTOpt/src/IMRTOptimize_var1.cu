#include "IMRTOptimize.cuh"

template<class T>
IMRT::array_1d<T>::~array_1d() {
    if (this->data != nullptr) {
        checkCudaErrors(cudaFree(this->data));
        this->data = nullptr;
    }
    if (this->vec != nullptr) {
        checkCusparse(cusparseDestroyDnVec(this->vec));
        this->vec = nullptr;
    }
}

template<class T>
bool IMRT::array_1d<T>::resize(size_t new_size) {
    if (new_size > this->size) {
        std::cerr << "Only dimension reduction supported, "
            "expansion not supported." << std::endl;
        return 1;
    }
    if (this->vec != nullptr)
        checkCusparse(cusparseDestroyDnVec(this->vec));
    checkCusparse(cusparseCreateDnVec(&this->vec, new_size, this->data, CUDA_R_32F));
    this->size = new_size;
    return 0;
}

template <class T>
IMRT::array_1d<T>& IMRT::array_1d<T>::operator=(const IMRT::array_1d<T>& other) {
    if (this != &other) {  // Avoid self-assignment
        if (this->size == other.size) {
            // of the same size, no need to allocate memory
            checkCudaErrors(cudaMemcpy(this->data, other.data,
                other.size*sizeof(float), cudaMemcpyDeviceToDevice));
            if (other.vec != nullptr && this->vec == nullptr)
                checkCusparse(cusparseCreateDnVec(&this->vec, this->size, this->data, CUDA_R_32F));
        } else {
            if (this->vec != nullptr)
                checkCusparse(cusparseDestroyDnVec(this->vec));
            if (this->data != nullptr)
                checkCudaErrors(cudaFree(this->data));
            
            this->size = other.size;
            checkCudaErrors(cudaMalloc((void**)&this->data, this->size*sizeof(float)));
            checkCudaErrors(cudaMemcpy(this->data, other.data,
                this->size*sizeof(float), cudaMemcpyDeviceToDevice));
            if (other.vec != nullptr)
                checkCusparse(cusparseCreateDnVec(&this->vec, this->size, this->data, CUDA_R_32F));
        }
    }
    return *this;
}

template class IMRT::array_1d<float>;
template class IMRT::array_1d<uint8_t>;

bool IMRT::beamWeightsInit_func(
    const MatReservior& VOIRes, array_1d<float>& beamWeightsInit,
    size_t ptv_voxels, size_t oar_voxels, const cusparseHandle_t& handle,
    const cublasHandle_t& cublas_handle) {
    size_t bufferSize = 0;
    void* buffer = nullptr;
    array_1d<float> output;
    arrayInit(output, ptv_voxels + oar_voxels);
    int nBeams = VOIRes.reservior.size();
    std::vector<float> beamWeightsInit_h(nBeams, 0.0f);
    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i=0; i<nBeams; i++) {
        const MatCSR64& mat = VOIRes.reservior[i];
        size_t nBeamlets = mat.numCols;

        std::vector<float> input_h(nBeamlets, 1.0f);
        array_1d<float> input;
        arrayInit(input, nBeamlets);
        checkCudaErrors(cudaMemcpy(input.data, input_h.data(),
            nBeamlets*sizeof(float), cudaMemcpyHostToDevice));
        
        size_t local_bufferSize = 0;
        checkCusparse(cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat.matA, input.vec, &beta, output.vec,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &local_bufferSize));
        if (local_bufferSize > bufferSize) {
            bufferSize = local_bufferSize;
            if (buffer != nullptr) {
                checkCudaErrors(cudaFree(buffer));
            }
            checkCudaErrors(cudaMalloc(&buffer, bufferSize));
        }

        checkCusparse(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat.matA, input.vec, &beta, output.vec,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        float sum = 0.0f;
        checkCublas(cublasSasum(cublas_handle, ptv_voxels, output.data, 1, &sum));
        beamWeightsInit_h[i] = sqrtf(sum / (ptv_voxels * nBeamlets));
    }

    // post-processing
    float maximum_value = 0.0f;
    for (int i=0; i<nBeams; i++) {
        maximum_value = max(maximum_value, beamWeightsInit_h[i]);
    }
    for (int i=0; i<nBeams; i++) {
        beamWeightsInit_h[i] /= maximum_value;
        if (beamWeightsInit_h[i] < 0.1f)
            beamWeightsInit_h[i] = 0.1f;
    }

    #if false
        // for debug purposes
        std::cout << "beamWeightsInit_h:\n";
        for (int i=0; i<nBeams; i++)
            std::cout << beamWeightsInit_h[i] << "  ";
        std::cout << std::endl;
        return 0;
    #endif

    // here we assume the output, beamWeightsInit, is already initialized
    if (beamWeightsInit.size != nBeams) {
        std::cerr << "beamWeightsInit is not initialized." << std::endl;
        return 1;
    }
    checkCudaErrors(cudaMemcpy(beamWeightsInit.data, beamWeightsInit_h.data(),
        nBeams*sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

bool IMRT::arrayInit_group2(const std::vector<MatCSR64*>& array_group2,
    const std::vector<uint8_t>& fluenceArray, int nBeams, int fluenceDim) {
    // sanity check
    if (nBeams * fluenceDim * fluenceDim != fluenceArray.size()) {
        std::cerr << "Fluence array size mismatch: (nBeams, fluenceDim, fluenceArray.size()) == ("
            << nBeams << ", " << fluenceDim << ", " << fluenceArray.size() << ")" << std::endl;
        return 1;
    }
    size_t max_active_beamlets = 0;
    for (size_t i=0; i<fluenceArray.size(); i++) {
        max_active_beamlets += (fluenceArray[i] > 0);
    }
    for (MatCSR64* ptr : array_group2) {
        ptr->numRows = nBeams;
        ptr->numCols = fluenceDim * fluenceDim;
        ptr->nnz = max_active_beamlets;
        checkCudaErrors(cudaMalloc((void**)&(ptr->d_csr_offsets), (ptr->numRows+1)*sizeof(size_t)));
        checkCudaErrors(cudaMalloc((void**)&(ptr->d_csr_columns), ptr->nnz*sizeof(size_t)));
        checkCudaErrors(cudaMalloc((void**)&(ptr->d_csr_values), ptr->nnz*sizeof(float)));
        ptr->matA = nullptr;
    }
    return 0;
}

bool IMRT::resize_group2::evaluate(const std::vector<MatCSR64*>& array_group2,
    const array_1d<float>& fluenceArray, int nBeams, int fluenceDim) {
    if (this->handle == nullptr)
        checkCusparse(cusparseCreate(&this->handle));

    cusparseDnMatDescr_t matDense;
    int nBeamletsPerBeam = fluenceDim * fluenceDim;
    checkCusparse(cusparseCreateDnMat(
        &matDense, nBeams, nBeamletsPerBeam, nBeamletsPerBeam,
        fluenceArray.data, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    
    // act on the first matrix
    MatCSR64& example = *(array_group2[0]);
    if (example.matA)
        checkCusparse(cusparseDestroySpMat(example.matA));
    
    checkCusparse(cusparseCreateCsr(
        &example.matA, nBeams, nBeamletsPerBeam, 0,
        example.d_csr_offsets, nullptr, nullptr,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    size_t bufferSize_local = 0;
    checkCusparse(cusparseDenseToSparse_bufferSize(
        this->handle, matDense, example.matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize_local));
    if (bufferSize_local > this->bufferSize) {
        this->bufferSize = bufferSize_local;
        if (this->buffer != nullptr)
            checkCudaErrors(cudaFree(this->buffer));
        checkCudaErrors(cudaMalloc(&(this->buffer), this->bufferSize));
    }

    checkCusparse(cusparseDenseToSparse_analysis(
        this->handle, matDense, example.matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, this->buffer));
    
    int64_t num_rows_tmp, num_cols_tmp, nnz_tmp;
    checkCusparse(cusparseSpMatGetSize(
        example.matA, &num_rows_tmp, &num_cols_tmp, &nnz_tmp));
    
    example.numRows = nBeams;
    example.numCols = nBeamletsPerBeam;
    example.nnz = nnz_tmp;

    // as memory is already allocated, no need to allocate again
    checkCusparse(cusparseCsrSetPointers(example.matA, example.d_csr_offsets,
        example.d_csr_columns, example.d_csr_values));
    checkCusparse(cusparseDenseToSparse_convert(handle, matDense, example.matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, this->buffer));

    // clean up
    checkCusparse(cusparseDestroyDnMat(matDense));
    
    for (int i=1; i<array_group2.size(); i++) {
        MatCSR64* ptr = array_group2[i];
        ptr->numRows = example.numRows;
        ptr->numCols = example.numCols;
        ptr->nnz = example.nnz;
        checkCudaErrors(cudaMemcpy(ptr->d_csr_offsets, example.d_csr_offsets,
            (example.numRows+1)*sizeof(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(ptr->d_csr_columns, example.d_csr_columns,
            example.nnz*sizeof(size_t), cudaMemcpyDeviceToDevice));
        // no need to set values
        if (ptr->matA != nullptr)
            checkCusparse(cusparseDestroySpMat(ptr->matA));
        checkCusparse(cusparseCreateCsr(
            &(ptr->matA), ptr->numRows, ptr->numCols, ptr->nnz,
            ptr->d_csr_offsets, ptr->d_csr_columns, ptr->d_csr_values,
            CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }
    return 0;
}

bool IMRT::arrayToMatScatter(const array_1d<float>& source, MatCSR64& target) {
    if (source.size != target.nnz) {
        std::cerr << "The size of the source input vector should match the "
            "number of non-zero elements of the target sparse matrix. But "
            "(source.size, target.nnz) == (" << source.size << ", "
            << target.nnz << ")" << std::endl;
        return 1;
    }
    checkCudaErrors(cudaMemcpy(target.d_csr_values, source.data,
        target.nnz*sizeof(float), cudaMemcpyDeviceToDevice));
    return 0;
}

bool IMRT::elementWiseMax(MatCSR64& target, float value) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (target.nnz + blockSize.x - 1) / blockSize.x;
    d_elementWiseMax<<<gridSize, blockSize>>>(target.d_csr_values, value, target.nnz);
    return 0;
}

__global__ void
IMRT::d_elementWiseMax(float* target, float value, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    
    target[idx] = max(target[idx], value);
}

bool IMRT::elementWiseSquare(float* source, float* target, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_elementWiseSquare<<<gridSize, blockSize>>>(source, target, size);
    return 0;
}

__global__ void IMRT::d_elementWiseSquare(float* source, float* target, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    float source_value = source[idx];
    target[idx] = source_value * source_value;
}

bool IMRT::elementWisePower(float* source, float* target, float power, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_elementWisePower<<<gridSize, blockSize>>>(source, target, power, size);
    return 0;
}

__global__ void IMRT::d_elementWisePower(float* source,
    float* target, float power, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    float source_value = source[idx];
    target[idx] = powf(source_value, power);
}

bool IMRT::elementWiseMul(
    float* a, float* b, float* c, float t, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_elementWiseMul<<<gridSize, blockSize>>>(a, b, c, t, size);
    return 0;
}

__global__ void IMRT::d_elementWiseMul(
    float* a, float* b, float* c, float t, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    c[idx] = a[idx] * b[idx] * t;
}

bool IMRT::calc_sHat(float* sHat, float* alpha, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_calc_sHat<<<gridSize, blockSize>>>(sHat, alpha, size);
    return 0;
}

__global__ void IMRT::d_calc_sHat(float* sHat, float* alpha, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    
#define TWO_OVER_ROOT3 1.1547005383792517f
#define THREE_QUARTERS_ROOT3 1.299038105676658f
#define PI_OVER_TWO 1.5707963267948966f
    
    float input_value = alpha[idx];
    float output_value = TWO_OVER_ROOT3 * sinf((acosf(THREE_QUARTERS_ROOT3
        * input_value) + PI_OVER_TWO) / 3);
    sHat[idx] = output_value;

#undef PI_OVER_TWO
#undef THREE_QUARTERS_ROOT3
#undef TWO_OVER_ROOT3
}

bool IMRT::tHat_step2(float* tHat, float* alpha, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_tHat_step2<<<gridSize, blockSize>>>(tHat, alpha, size);
    return 0;
}

__global__ void IMRT::d_tHat_step2(float* tHat, float* alpha, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
#define TWO_TIMES_ROOT6_OVER_NINE 0.5443310539518174f
    float alpha_value = alpha[idx];
    if (alpha_value > TWO_TIMES_ROOT6_OVER_NINE)
        tHat[idx] = 0.0f;
#undef TWO_TIMES_ROOT6_OVER_NINE
}

bool IMRT::calc_prox_tHat_g0(
    size_t numRows, size_t numCols, size_t nnz,
    float* tHat, size_t* g0_offsets, size_t* g0_columns,
    float* g0_values, float* prox_values) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (numRows + blockSize.x - 1) / blockSize.x;
    d_calc_prox_tHat_g0<<<gridSize, blockSize>>>(
        numRows, numCols, nnz,
        tHat, g0_offsets, g0_columns,
        g0_values, prox_values);
    return 0;
}

__global__ void IMRT::d_calc_prox_tHat_g0(
    size_t numRows, size_t numCols, size_t nnz,
    float* tHat, size_t* g0_offsets, size_t* g0_columns,
    float* g0_values, float* prox_values) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numRows)
        return;
    float tHat_value = tHat[idx];
    size_t row_idx_start = g0_offsets[idx];
    size_t row_idx_end = g0_offsets[idx+1];
    for (size_t i=row_idx_start; i<row_idx_end; i++) {
        prox_values[i] = tHat_value * g0_values[i];
    }
}

bool IMRT::elementWiseSqrt(float* source, float* target, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_elementWiseSqrt<<<gridSize, blockSize>>>(source, target, size);
    return 0;
}

__global__ void IMRT::d_elementWiseSqrt(
    float* source, float* target, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    target[idx] = sqrtf(source[idx]);
}

bool IMRT::elementWiseScale(float* source, float a, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_elementWiseScale<<<gridSize, blockSize>>>(source, a, size);
    return 0;
}

__global__ void IMRT::d_elementWiseScale(
    float* source, float a, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    source[idx] = source[idx] * a;
}

bool IMRT::elementWiseGreater(float* source,
    float* target, float a, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_elementWiseGreater<<<gridSize, blockSize>>>(source, target, a, size);
    return 0;
}

__global__ void IMRT::d_elementWiseGreater(float* source,
    float* target, float a, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    target[idx] = (source[idx] > a);
}

bool IMRT::beamSort(const std::vector<float>& beamNorms_h,
    std::vector<std::pair<int, float>>& beamNorms_pair) {
    int nBeams = beamNorms_h.size();
    beamNorms_pair.resize(nBeams);
    for (int i=0; i<nBeams; i++) {
        beamNorms_pair[i] = std::make_pair(i, beamNorms_h[i]);
    }
    auto customComparator = [](const std::pair<int, float>& a,
        const std::pair<int, float>& b) {
        return a.second > b.second; };
    std::sort(beamNorms_pair.begin(), beamNorms_pair.end(), customComparator);
    return 0;
}

bool IMRT::proxL2Onehalf_QL_gpu::customInit(const MatCSR64& g0) {
    // In here, we assume the input g0 has the maximum nnz
    arrayInit(this->g0_square_values, g0.nnz);
    arrayInit(this->sum_input, g0.numCols);
    arrayInit(this->nrm2, g0.numRows);
    arrayInit(this->nrm234, g0.numRows);
    arrayInit(this->alpha, g0.numRows);
    arrayInit(this->sHat, g0.numRows);
    arrayInit(this->tHat, g0.numRows);
    arrayInit(this->prox_square, g0.nnz);
    arrayInit(this->nrm2newbuff, g0.numRows);
    checkCusparse(cusparseCreate(&this->handle));
    this->initFlag = true;
    return 0;
}


IMRT::proxL2Onehalf_QL_gpu::~proxL2Onehalf_QL_gpu() {
    if (this->handle != nullptr) {
        checkCusparse(cusparseDestroy(this->handle));
    }
    if (this->sum_buffer != nullptr) {
        checkCudaErrors(cudaFree(this->sum_buffer));
    }
}


bool IMRT::proxL2Onehalf_QL_gpu::evaluate(
    const MatCSR64& g0, const array_1d<float>& beamWeights, float t,
    MatCSR64& prox, array_1d<float>& nrmnew) {
    elementWiseSquare(g0.d_csr_values, this->g0_square_values.data, g0.nnz);
    // for safety concerns, synchronize between the cuda kernels and cusparse APIs
    checkCudaErrors(cudaDeviceSynchronize());

    // construct a temporary sparse matrix that stores g0_square,
    // with the same sparsity pattern as g0
    cusparseSpMatDescr_t g0_2_mat = nullptr;
    checkCusparse(cusparseCreateCsr(
        &g0_2_mat, g0.numRows, g0.numCols, g0.nnz,
        g0.d_csr_offsets, g0.d_csr_columns, this->g0_square_values.data,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    float alpha = 1.0f;
    float beta = 0.0f;

    size_t bufferSize = 0;
    checkCusparse(cusparseSpMV_bufferSize(
        this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, g0_2_mat, this->sum_input.vec, &beta, this->nrm2.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (this->sum_buffer_size < bufferSize) {
        this->sum_buffer_size =  bufferSize;
        if (this->sum_buffer != nullptr)
            checkCudaErrors(cudaFree(this->sum_buffer));
        checkCudaErrors(cudaMalloc(&this->sum_buffer, bufferSize));
    }

    checkCusparse(cusparseSpMV(
        this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, g0_2_mat, this->sum_input.vec, &beta, this->nrm2.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, this->sum_buffer));

    elementWisePower(this->nrm2.data, this->nrm234.data, -0.75f, this->nrm2.size);
    
    elementWiseMul(beamWeights.data, this->nrm234.data,
        this->alpha.data, t, this->alpha.size);
    
    calc_sHat(this->sHat.data, this->alpha.data, this->alpha.size);

    elementWiseSquare(this->sHat.data, this->tHat.data, this->sHat.size);

    tHat_step2(this->tHat.data, this->alpha.data, this->tHat.size);

    if (g0.numRows != prox.numRows || g0.numCols != prox.numCols
        || g0.nnz != prox.nnz) {
        std::cerr << "The input matrices g0 and prox are supposed "
            "to be of the same sparsity pattern" << std::endl;
        return 1;
    }
    calc_prox_tHat_g0(g0.numRows, g0.numCols, g0.nnz,
        this->tHat.data, g0.d_csr_offsets, g0.d_csr_columns, g0.d_csr_values,
        prox.d_csr_values);
    
    elementWiseSquare(prox.d_csr_values, this->prox_square.data, prox.nnz);
    // for safety concerns, synchronize between the cuda kernels and cusparse APIs
    checkCudaErrors(cudaDeviceSynchronize());

    // construct a temporary cusparseSpMatDescr_t class.
    cusparseSpMatDescr_t prox_square_matA;
    checkCusparse(cusparseCreateCsr(
        &prox_square_matA, g0.numRows, g0.numCols, g0.nnz,
        g0.d_csr_offsets, g0.d_csr_columns, this->prox_square.data,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    checkCusparse(cusparseSpMV(
        this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, prox_square_matA, this->sum_input.vec, &beta, this->nrm2newbuff.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, this->sum_buffer));

    if (nrmnew.size != g0.numRows) {
        std::cerr << "The size of nrmnew should match g0.numRows." << std::endl;
        return 1;
    }
    elementWiseSqrt(this->nrm2newbuff.data, nrmnew.data, nrmnew.size);

    checkCusparse(cusparseDestroySpMat(g0_2_mat));
    checkCusparse(cusparseDestroySpMat(prox_square_matA));
    return 0;
}

IMRT::calc_rhs::calc_rhs() {
    this->firstTerm.vec = nullptr;
    this->firstTerm.data = nullptr;
    this->firstTerm.size = 0;
    this->x_minus_y.vec = nullptr;
    this->x_minus_y.data = nullptr;
    this->x_minus_y.size = 0;
}

bool IMRT::calc_rhs::customInit(size_t bufferSize) {
    if (this->firstTerm.data != nullptr || this->x_minus_y.data != nullptr) {
        std::cerr << "The buffer is supposed to be un-initialized." << std::endl;
        return 1;
    }
    arrayInit(this->firstTerm, bufferSize);
    arrayInit(this->x_minus_y, bufferSize);
    return 0;
}

bool IMRT::calc_rhs::evaluate(
    float& rhs, float gy, float t, const array_1d<float>& gradAty,
    const array_1d<float>& x, const array_1d<float>& y,
    const cublasHandle_t& cublas_handle) {
    if (x.size != gradAty.size || x.size != y.size) {
        std::cerr << "Input size inconsistency." << std::endl;
        return 1;
    }
    if (x.size > this->firstTerm.size) {
        std::cerr << "Insufficient buffer size." << std::endl;
        return 1;
    }

    linearComb_array_1d(1.0f, x, -1.0f, y, this->x_minus_y);
    elementWiseMul(gradAty.data, this->x_minus_y.data,
        this->firstTerm.data, 1.0f, gradAty.size);
    float value1;
    checkCublas(cublasSasum(cublas_handle, gradAty.size,
        this->firstTerm.data, 1, &value1));

    checkCudaErrors(cudaDeviceSynchronize());
    elementWiseSquare(this->x_minus_y.data,
        this->firstTerm.data, gradAty.size);
    float value2;
    checkCublas(cublasSasum(cublas_handle, gradAty.size,
        this->firstTerm.data, 1, &value2));
    
    rhs = gy + value1 + value2 * 0.5f / t;
    return 0;
}

IMRT::calc_loss::calc_loss() {
    this->buffer_sqrt.size = 0;
    this->buffer_sqrt.vec = nullptr;
    this->buffer_sqrt.data = nullptr;
    this->buffer_mul.size = 0;
    this->buffer_mul.vec = nullptr;
    this->buffer_mul.data = nullptr;
}

bool IMRT::calc_loss::customInit(size_t nBeams) {
    arrayInit(this->buffer_sqrt, nBeams);
    arrayInit(this->buffer_mul, nBeams);
    return 0;
}

bool IMRT::calc_loss::evaluate(
    float& cost, float gx, const array_1d<float>& beamWeights,
    const array_1d<float>& beamNorms, const cublasHandle_t& cublas_handle) {
    // sanity check
    if (beamWeights.size != beamNorms.size ||
        beamWeights.size != this->buffer_sqrt.size ||
        beamWeights.size != this->buffer_mul.size) {
        std::cerr << "Vector sizes don't match with each other." << std::endl;
        return 1;
    }
    elementWiseSqrt(beamNorms.data, this->buffer_sqrt.data, beamNorms.size);
    elementWiseMul(this->buffer_sqrt.data, beamWeights.data,
        this->buffer_mul.data, 1.0f, beamNorms.size);
    
    checkCudaErrors(cudaDeviceSynchronize());
    float value1;
    checkCublas(cublasSasum(cublas_handle, beamNorms.size,
        this->buffer_mul.data, 1, &value1));
    cost = gx + value1;
    return 0;
}