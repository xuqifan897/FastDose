#include "IMRTOptimize.cuh"

#define TWO_OVER_ROOT3 1.1547005383792517f
#define THREE_QUARTERS_ROOT3 1.299038105676658f
#define PI_OVER_TWO 1.5707963267948966f
#define TWO_TIMES_ROOT6_OVER_NINE 0.5443310539518174f

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
bool IMRT::array_1d<T>::copy(const IMRT::array_1d<T>& old) {
    if (this->size != old.size) {
        std::cerr << "Size unmatch in function array_1d<T>::copy" << std::endl;
        return 1;
    }
    checkCudaErrors(cudaMemcpy(this->data, old.data,
        this->size*sizeof(T), cudaMemcpyDeviceToDevice));
    return 0;
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


bool IMRT::beamWeightsInit_func(
    const std::vector<const MatCSR_Eigen*>& VOIMatrices,
    std::vector<float>& beamWeightsInit,
    size_t ptv_voxels, size_t oar_voxels
) {
    size_t numBeams = VOIMatrices.size();
    beamWeightsInit.resize(numBeams);

    Eigen::VectorXf result_vec(ptv_voxels + oar_voxels);
    for (size_t i=0; i<numBeams; i++) {
        const MatCSR_Eigen& VOIMat = *VOIMatrices[i];
        size_t numBeamlets = VOIMat.getCols();

        Eigen::VectorXf sum_vec(numBeamlets);
        for (size_t j=0; j<numBeamlets; j++)
            sum_vec(j) = 1.0f;
        result_vec = VOIMat * sum_vec;

        float local_sum = 0.0f;
        for (size_t j=0; j<ptv_voxels; j++)
            local_sum += result_vec(j);
        
        beamWeightsInit[i] = sqrtf(local_sum / (ptv_voxels * numBeamlets));
    }

    // post-processing
    float maximum_value = 0.0f;
    for (size_t i=0; i<numBeams; i++)
        maximum_value = max(maximum_value, beamWeightsInit[i]);
    for (size_t i=0; i<numBeams; i++) {
        beamWeightsInit[i] /= maximum_value;
        if (beamWeightsInit[i] < 0.1f)
            beamWeightsInit[i] = 0.1f;
    }
    
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

bool IMRT::elementWiseMax(array_1d<float>& target, float value) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (target.size + blockSize.x - 1) / blockSize.x;
    d_elementWiseMax<<<gridSize, blockSize>>>(target.data, value, target.size);
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

bool IMRT::elementWiseScale(float* source, float* target, float a, size_t size) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (size + blockSize.x - 1) / blockSize.x;
    d_elementWiseScale<<<gridSize, blockSize>>>(source, target, a, size);
    return 0;
}

__global__ void IMRT::d_elementWiseScale(
    float* source, float* target, float a, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    target[idx] = source[idx] * a;
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

bool IMRT::proxL2Onehalf_QL_gpu::customInit(const MatCSR64& g0,
    const cusparseHandle_t& handle_cusparse) {
    // this->g02 should be of the same sparsity pattern as g0
    this->numRows = g0.numRows;
    this->numCols = g0.numCols;
    this->nnz = g0.nnz;
    this->blockSize = dim3(64, 1, 1);
    this->gridSize = dim3((this->numRows + blockSize.x - 1) / blockSize.x, 1, 1);
    checkCudaErrors(cudaMalloc((void**)&(this->g02.d_csr_offsets),
        (this->numRows+1)*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&(this->g02.d_csr_columns), this->nnz*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&(this->g02.d_csr_values), this->nnz*sizeof(float)));
    
    checkCudaErrors(cudaMemcpy(this->g02.d_csr_offsets, g0.d_csr_offsets,
        (this->numRows+1)*sizeof(size_t), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(this->g02.d_csr_columns, g0.d_csr_columns,
        this->nnz * sizeof(size_t), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(this->g02.d_csr_values, g0.d_csr_values,
        this->nnz * sizeof(float), cudaMemcpyDeviceToDevice));

    this->g02.numRows = this->numRows;
    this->g02.numCols = this->numCols;
    this->g02.nnz = this->nnz;
    checkCusparse(cusparseCreateCsr(
        &(this->g02.matA), this->numRows, this->numCols, this->nnz,
        this->g02.d_csr_offsets, this->g02.d_csr_columns, this->g02.d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    arrayInit(this->sum_arr, this->numCols);
    std::vector<float> sum_arr_host(this->numCols, 1.0f);
    checkCudaErrors(cudaMemcpy(this->sum_arr.data, sum_arr_host.data(),
        this->numCols*sizeof(float), cudaMemcpyHostToDevice));
    arrayInit(this->buffer, this->numRows);

    // initialize the multiplication buffer of g02
    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    checkCusparse(cusparseSpMV_bufferSize(
        handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, this->g02.matA, this->sum_arr.vec, &beta, this->buffer.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc(&this->g02.d_buffer_spmv, bufferSize));
    return 0;
}

bool IMRT::proxL2Onehalf_QL_gpu::evaluate(
    const MatCSR64& g0, const array_1d<float>& tau,
    MatCSR64& prox, array_1d<float>& nrmnew,
    const cusparseHandle_t& handle_cusparse) {
    // calculate this->g02
    elementWiseSquare(g0.d_csr_values, this->g02.d_csr_values, g0.nnz);
    checkCudaErrors(cudaDeviceSynchronize());

    // calculate nrm2 = sum(g0.^2, 1) = sum(this->g02, 1);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    checkCusparse(cusparseSpMV(
        handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, this->g02.matA, this->sum_arr.vec, &beta, this->buffer.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, this->g02.d_buffer_spmv));

    checkCudaErrors(cudaDeviceSynchronize());
    d_proxL2Onehalf_calc_tHat<<<this->gridSize, this->blockSize>>>(
        this->buffer.data, tau.data, this->numRows);
    
    // calculate prox
    calc_prox_tHat_g0(this->numRows, this->numCols, this->nnz,
        this->buffer.data, g0.d_csr_offsets, g0.d_csr_columns,
        g0.d_csr_values, prox.d_csr_values);
    
    // calculate prox square. As this->g02 is no longer needed,
    // we use it to store the results.
    elementWiseSquare(prox.d_csr_values, this->g02.d_csr_values, this->nnz);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCusparse(cusparseSpMV(
        handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, this->g02.matA, this->sum_arr.vec, &beta, this->buffer.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, this->g02.d_buffer_spmv));
    checkCudaErrors(cudaDeviceSynchronize());

    // calculate nrmnew
    elementWiseSqrt(this->buffer.data, nrmnew.data, this->numRows);
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}


__global__ void IMRT::d_proxL2Onehalf_calc_tHat(float* buffer, float* tau, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    
    float tau_value = tau[idx];
    float nrm2_value = buffer[idx];
    // for arithmatic stability, clamp on nrm2_value
    float nrm234_value = 1008611.0f;  // assign a dummy value.
    if (nrm2_value > eps_fastdose)
        nrm234_value = powf(nrm2_value, -0.75f);

    float alpha_value = tau_value * nrm234_value;
    if (alpha_value > TWO_TIMES_ROOT6_OVER_NINE) {
        buffer[idx] = 0.0f;
        return;
    }

    float source_value  = alpha_value;
    source_value *= THREE_QUARTERS_ROOT3;
    source_value = acosf(source_value);
    source_value += PI_OVER_TWO;
    source_value /= 3.0f;
    source_value = std::sin(source_value);
    source_value *= TWO_OVER_ROOT3;  // sHat
    buffer[idx] = source_value * source_value;  // tHat
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