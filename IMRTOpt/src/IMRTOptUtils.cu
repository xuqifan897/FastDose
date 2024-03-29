#include <iostream>
#include <iomanip>
#include "IMRTOptimize.cuh"
#include "IMRTOptimize_var.cuh"


IMRT::eval_g::eval_g(size_t ptv_voxels, size_t oar_voxels, size_t d_rows) {
    this->PTV_voxels = ptv_voxels;
    this->OAR_voxels = oar_voxels;
    this->D_rows = d_rows;
    this->alpha = 1.0f;
    this->beta = 0.0f;

    checkCudaErrors(cudaStreamCreate(&this->stream1));
    checkCudaErrors(cudaStreamCreate(&this->stream2));
    checkCudaErrors(cudaStreamCreate(&this->stream3));
    checkCudaErrors(cudaStreamCreate(&this->stream4));
    checkCudaErrors(cudaStreamCreate(&this->stream5));

    arrayInit(this->Ax, this->PTV_voxels + this->OAR_voxels);
    arrayInit(this->prox1, this->PTV_voxels);
    arrayInit(this->prox2, this->PTV_voxels + this->OAR_voxels);
    arrayInit(this->term3, this->OAR_voxels);
    arrayInit(this->term4, this->D_rows);
    arrayInit(this->prox4, this->D_rows);

    arrayInit(this->sumProx1, this->prox1.size);
    arrayInit(this->sumProx2, this->prox2.size);
    arrayInit(this->sumTerm3, this->term3.size);
    arrayInit(this->sumProx4, this->prox4.size);
    arrayInit(this->sumProx4Term4, this->prox4.size);
}

bool IMRT::arrayInit(array_1d<float>& arr, size_t size) {
    if (arr.init_flag) {
        std::cerr << "The array is already initialized. Double "
            "initialization not allowed." << std::endl;
        return 1;
    }
    arr.init_flag = true;
    arr.size = size;
    checkCudaErrors(cudaMalloc((void**)&arr.data, arr.size * sizeof(float)));
    checkCusparse(cusparseCreateDnVec(&arr.vec, arr.size, arr.data, CUDA_R_32F));
    return 0;
}

bool IMRT::arrayInit(array_1d<float>& arr, const Eigen::VectorXf& source) {
    if (arr.init_flag) {
        std::cerr << "The array is already initialized. Double "
            "initialization not allowed." << std::endl;
        return 1;
    }
    arr.init_flag = true;
    arr.size = source.size();
    checkCudaErrors(cudaMalloc((void**)&arr.data, arr.size*sizeof(float)));
    checkCudaErrors(cudaMemcpy(arr.data, source.data(), arr.size*sizeof(float), cudaMemcpyHostToDevice));
    checkCusparse(cusparseCreateDnVec(&arr.vec, arr.size, arr.data, CUDA_R_32F));
    return 0;
}

IMRT::eval_g::~eval_g() {
    if (this->stream1)
        checkCudaErrors(cudaStreamDestroy(this->stream1));
    if (this->stream2)
        checkCudaErrors(cudaStreamDestroy(this->stream2));
    if (this->stream3)
        checkCudaErrors(cudaStreamDestroy(this->stream3));
    if (this->stream4)
        checkCudaErrors(cudaStreamDestroy(this->stream4));
    if (this->stream5)
        checkCudaErrors(cudaStreamDestroy(this->stream5));
}

float IMRT::eval_g::evaluate(const MatCSR64& A, const MatCSR64& D,
    const array_1d<float>& x, float gamma,
    const cusparseHandle_t& handle_cusparse,
    const cublasHandle_t& handle_cublas,
    
    const float* maxDose,
    const float* minDoseTarget,
    const float* minDoseTargetWeights,
    const float* maxWeightsLong,
    const float* OARWeightsLong,
    float eta) {
    #if sanityCheck
        if (A.numCols != x.size || A.numRows != Ax.size ||
            D.numCols != x.size || D.numRows != term4.size) {
            std::cerr << "The size of the matrices A, D and the "
                "size of the input vector x are incompatible." << std::endl;
            exit(EXIT_FAILURE);
        }
    #endif

    checkCusparse(cusparseSpMV(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &this->alpha, A.matA, x.vec, &this->beta, this->Ax.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, A.d_buffer_spmv));

    checkCusparse(cusparseSpMV(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &this->alpha, D.matA, x.vec, &this->beta, this->term4.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, D.d_buffer_spmv));

    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blockSize(256, 1, 1);
    dim3 gridSize((this->prox1.size + blockSize.x - 1) / blockSize.x, 1, 1);
    d_calc_prox1<<<gridSize, blockSize, 0, this->stream1>>>(
        this->prox1.data, this->Ax.data, minDoseTarget, this->prox1.size);
    
    gridSize.x = (this->prox2.size + blockSize.x - 1) / blockSize.x;
    d_calc_prox2<<<gridSize, blockSize, 0, this->stream2>>>(
        this->prox2.data, this->Ax.data, maxDose, this->prox2.size);

    checkCudaErrors(cudaMemcpyAsync(this->term3.data, this->Ax.data + this->PTV_voxels,
        this->OAR_voxels * sizeof(float), cudaMemcpyDeviceToDevice, this->stream3));

    gridSize.x = (this->D_rows + blockSize.x - 1) / blockSize.x;
    d_prox1Norm<<<gridSize, blockSize, 0, this->stream4>>>(
        this->prox4.data, this->term4.data, gamma, this->D_rows);

    gridSize.x = (this->prox1.size + blockSize.x - 1) / blockSize.x;
    d_ATimesBSquare<<<gridSize, blockSize, 0, this->stream1>>>(
        this->sumProx1.data, minDoseTargetWeights, this->prox1.data, this->prox1.size);
    
    gridSize.x = (this->prox2.size + blockSize.x - 1) / blockSize.x;
    d_ATimesBSquare<<<gridSize, blockSize, 0, this->stream2>>>(
        this->sumProx2.data, maxWeightsLong, this->prox2.data, this->prox2.size);

    gridSize.x = (this->term3.size + blockSize.x - 1) / blockSize.x;
    d_ATimesBSquare<<<gridSize, blockSize, 0, this->stream3>>>(
        this->sumTerm3.data, OARWeightsLong, this->term3.data, this->term3.size);

    gridSize.x = (this->prox4.size + blockSize.x - 1) / blockSize.x;
    d_calcSumProx4<<<gridSize, blockSize, 0, this->stream4>>>(
        this->sumProx4.data, this->prox4.data, this->prox4.size);
    
    d_calcProx4Term4<<<gridSize, blockSize, 0, this->stream5>>>(
        this->sumProx4Term4.data, this->prox4.data, this->term4.data, this->prox4.size);

    checkCudaErrors(cudaDeviceSynchronize());
    float sum1, sum2, sum3, sum4, sum5;
    checkCublas(cublasSasum(handle_cublas, this->sumProx1.size,
        this->sumProx1.data, 1, &sum1));
    checkCublas(cublasSasum(handle_cublas, this->sumProx2.size,
        this->sumProx2.data, 1, &sum2));
    checkCublas(cublasSasum(handle_cublas, this->sumTerm3.size,
        this->sumTerm3.data, 1, &sum3));
    checkCublas(cublasSasum(handle_cublas, this->sumProx4.size,
        this->sumProx4.data, 1, &sum4));
    checkCublas(cublasSasum(handle_cublas, this->sumProx4Term4.size,
        this->sumProx4Term4.data, 1, &sum5));
    
    float result = 0.5f * (sum1 + sum2 + sum3)
        +  eta * (sum4 + 0.5f / gamma * sum5);

    return result;
}

__global__ void
IMRT::d_calc_prox1(float* prox1_data, float* Ax_data,
    const float* minDoseTargetData, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    prox1_data[idx] = min(Ax_data[idx] - minDoseTargetData[idx], 0.0f);
}

__global__ void
IMRT::d_calc_prox2(float* prox2_data, float* Ax_data, const float* maxDose, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    prox2_data[idx] = max(Ax_data[idx] - maxDose[idx], 0.0f);
}

__global__ void
IMRT::d_prox1Norm(float* y, float* x, float t, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    
    float xValue = x[idx];
    float mx = max(abs(xValue) - t, 0.0f);
    int sign_x = d_sign(xValue);
    y[idx] = mx * sign_x;
}

__device__ int
IMRT::d_sign(float x) {
    return (x > 0) - (x < 0);
}

__global__ void
IMRT::d_ATimesBSquare(float* C, const float* A, float* B, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    
    float AValue = A[idx];
    float BValue = B[idx];
    float CValue = AValue * BValue * BValue;
    C[idx] = CValue;
}

__global__ void
IMRT::d_calcSumProx4(float* sumProx4Data, float* prox4Data, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;

    sumProx4Data[idx] = abs(prox4Data[idx]);
}

__global__ void
IMRT::d_calcProx4Term4(float* sumProx4Term4Data, float* prox4Data,
    float* term4Data, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;

    float tmp = prox4Data[idx] - term4Data[idx];
    sumProx4Term4Data[idx] = tmp * tmp;
}

bool IMRT::assignmentTest(){
    array_1d<float> source;
    source.size = 4;
    std::vector<float> source_data_h {1.0f, 2.0f, 3.0f, 4.0f};
    checkCudaErrors(cudaMalloc((void**)&source.data, source.size*sizeof(float)));
    checkCudaErrors(cudaMemcpy(source.data, source_data_h.data(),
        source.size*sizeof(float), cudaMemcpyHostToDevice));

    array_1d<float> dest;
    dest = source;
    std::vector<float> dest_data_h(dest.size);
    checkCudaErrors(cudaMemcpy(dest_data_h.data(), dest.data,
        dest.size * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Array dest size: " << dest.size << std::endl;
    for (int i=0; i<dest.size; i++)
        std::cout << dest_data_h[i] << " ";
    std::cout << std::endl;
    return 0;
}


void IMRT::arrayRand01(array_1d<float>& arr) {
    std::vector<float> arr_h(arr.size, 0.0f);
    for (size_t i=0; i<arr_h.size(); i++)
        arr_h[i] = (float)std::rand() / RAND_MAX;

    checkCudaErrors(cudaMemcpy(arr.data, arr_h.data(),
        arr_h.size()*sizeof(float), cudaMemcpyHostToDevice));
}


bool IMRT::StructsTrim(std::vector<StructInfo>& structs_trimmed,
    const std::vector<StructInfo>& structs) {
    std::cout << "Remove the overlapping part of the OARs with the PTV, "
        "exclude the parts that are irrelevant in the optimization. " 
        "Assuming that the zeroth structure is PTV, the first structure is BODY"
        << std::endl;
    
    const StructInfo& PTV = structs[0];
    const StructInfo& BODY = structs[1];
    const std::vector<uint8_t> PTV_mask = PTV.mask;
    uint3 size = PTV.size;

    std::vector<uint8_t> PTV_dilate;
    uint3 kernelSize{3, 3, 3};
    imdilate(PTV_dilate, PTV_mask, size, kernelSize);

    std::vector<uint8_t> IsPTV(structs.size(), 0);
    IsPTV[0] = 1;
    std::vector<uint8_t> IsOAR(structs.size(), 0);
    for (int i=0; i<structs.size(); i++) {
        const StructInfo& local_struct = structs[i];
        if (local_struct.maxWeights < eps_fastdose
            && local_struct.minDoseTargetWeights < eps_fastdose
            && local_struct.OARWeights < eps_fastdose) {
            IsOAR[i] = 0;
            continue;
        }
        IsOAR[i] = 1;
    }
    return 0;
}


bool IMRT::imdilate(std::vector<uint8_t>& dest,
    const std::vector<uint8_t>& source, uint3 shape, uint3 kernelSize) {
    // firstly, all the elements of kernelSize should be odd numbers
    if (kernelSize.x % 2 == 0 || kernelSize.y % 2 == 0 || kernelSize.z % 2 == 0) {
        std::cerr << "At least one kernelSize element is even. kernelSize: "
            << kernelSize << std::endl;
        return 1;
    }
    dest.resize(source.size());
    std::fill(dest.begin(), dest.end(), 0);

    uint3 halfKernelSize {(kernelSize.x - 1) / 2, (kernelSize.y - 1) / 2,
        (kernelSize.z - 1) / 2};

    for (int k=0; k<shape.z; k++) {
        int k_begin = std::max(0, k - (int)halfKernelSize.z);
        int k_end = std::min((int)shape.z, k + (int)halfKernelSize.z + 1);
        for (int j=0; j<shape.y; j++) {
            int j_begin = std::max(0, j - (int)halfKernelSize.y);
            int j_end = std::min((int)shape.y, j + (int)halfKernelSize.y + 1);
            for (int i=0; i<shape.x; i++) {
                size_t idx_source = i + shape.x * (j + shape.y * k);
                if (source[idx_source]) {
                    // dilate around the voxel
                    int i_begin = std::max(0, i - (int)halfKernelSize.x);
                    int i_end = std::min((int)shape.x, i + (int)halfKernelSize.x + 1);


                    for (int kk=k_begin; kk<k_end; kk++) {
                        for (int jj=j_begin; jj<j_end; jj++) {
                            for (int ii=i_begin; ii<i_end; ii++) {
                                size_t idx_dest = ii + shape.x * (jj + shape.y * kk);
                                dest[idx_dest] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}


bool IMRT::test_imdilate(){
    uint3 shape {10, 10, 3};
    std::vector<uint8_t> middle_slice {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    std::vector<uint8_t> source(shape.x * shape.y * shape.z, 0);
    std::copy(middle_slice.begin(), middle_slice.end(), source.begin() + shape.x * shape.y);

    std::vector<uint8_t> result;
    uint3 kernelSize{3, 3, 3};
    if (imdilate(result, source, shape, kernelSize)) {
        std::cerr << "The function imdilate() error." << std::endl;
        return 1;
    }
    for (int k=0; k<shape.z; k++) {
        for (int j=0; j<shape.y; j++) {
            for (int i=0; i<shape.x; i++) {
                size_t idx = i + shape.x * (j + shape.y * k);
                std::cout << (int)result[idx] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
    return 0;
}

bool IMRT::caseInsensitiveStringCompare(char c1, char c2) {
    return std::tolower(c1) == std::tolower(c2);
}

bool IMRT::containsCaseInsensitive(const std::string& data, const std::string& pattern) {
    auto it = std::search(data.begin(), data.end(),
        pattern.begin(), pattern.end(),
        caseInsensitiveStringCompare);
    
    return it != data.end();
}

bool IMRT::testContains() {
    std::vector<std::string> dataArray{"happy", "HaPpY", "Hobby", "hapPY"};
    std::string pattern {"APP"};
    for (const std::string& data : dataArray) {
        std::cout << "find(\"" << data << "\", \"" << pattern << "\") = "
            << containsCaseInsensitive(data, pattern) << std::endl;;
    }
    return 0;
}


bool IMRT::structComp(const std::tuple<StructInfo, bool, size_t>& a,
    const std::tuple<StructInfo, bool, size_t>& b) {
    return (std::get<1>(a) && !std::get<1>(b));
}


bool IMRT::testWeights(const Weights_h& weights) {
    std::cout << std::fixed << std::setprecision(0);
    for (size_t i=0; i<weights.maxDose.size(); i++)
        std::cout << weights.maxDose[i] << " ";
    std::cout << "\n maxDose size: " << weights.maxDose.size() << "\n" << std::endl;

    for (size_t i=0; i< weights.maxWeightsLong.size(); i++)
        std::cout << weights.maxWeightsLong[i] << " ";
    std::cout << "\nmaxWeightsLong size: " << weights.maxWeightsLong.size() << "\n" << std::endl;

    for (size_t i=0; i<weights.minDoseTarget.size(); i++)
        std::cout << weights.minDoseTarget[i] << " ";
    std::cout << "\nminDoseTarget size: " << weights.minDoseTarget.size() << "\n" << std::endl;

    for (size_t i=0; i<weights.minDoseTargetWeights.size(); i++)
        std::cout << weights.minDoseTargetWeights[i] << " ";
    std::cout << "\nminDoseTargetWeights: " << weights.minDoseTargetWeights.size()
        << "\n" << std::endl;
    
    for (size_t i=0; i<weights.OARWeightsLong.size(); i++)
        std::cout << weights.OARWeightsLong[i] << " ";
    std::cout << "\nOARWeightsLong: " << weights.OARWeightsLong.size() << "\n" << std::endl;
    return 0;
}


bool IMRT::Weights_d::fromHost(const Weights_h& source) {
    // sanity check
    if (source.maxDose.size() != source.voxels_PTV + source.voxels_OAR
        || source.maxWeightsLong.size() != source.voxels_PTV + source.voxels_OAR
        || source.minDoseTarget.size() != source.voxels_PTV
        || source.minDoseTargetWeights.size() != source.voxels_PTV
        || source.OARWeightsLong.size() != source.voxels_OAR) {
        std::cerr << "There is inconsistency between the weights array sizes." << std::endl;
        return 1;
    }

    if (this->maxDose.data || this->maxDose.vec
        || this->maxWeightsLong.data || this->maxWeightsLong.vec
        || this->minDoseTarget.data || this->minDoseTarget.vec
        || this->minDoseTargetWeights.data || this->minDoseTargetWeights.vec
        || this->OARWeightsLong.data || this->OARWeightsLong.vec) {
        std::cerr << "All weight arrays are supposed to be un-initualized." << std::endl;
        return 1;
    }

    this->voxels_PTV = source.voxels_PTV;
    this->voxels_OAR = source.voxels_OAR;

    arrayInit(this->maxDose, this->voxels_PTV + this->voxels_OAR);
    arrayInit(this->maxWeightsLong, this->voxels_PTV + this->voxels_OAR);
    arrayInit(this->minDoseTarget, this->voxels_PTV);
    arrayInit(this->minDoseTargetWeights, this->voxels_PTV);
    arrayInit(this->OARWeightsLong, this->voxels_OAR);

    checkCudaErrors(cudaMemcpy(this->maxDose.data, source.maxDose.data(),
        source.maxDose.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->maxWeightsLong.data, source.maxWeightsLong.data(),
        source.maxWeightsLong.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->minDoseTarget.data, source.minDoseTarget.data(),
        source.minDoseTarget.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->minDoseTargetWeights.data, source.minDoseTargetWeights.data(),
        source.minDoseTargetWeights.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->OARWeightsLong.data, source.OARWeightsLong.data(),
        source.OARWeightsLong.size()*sizeof(float), cudaMemcpyHostToDevice));

    return 0;
}


IMRT::eval_grad::eval_grad(size_t ptv_voxels, size_t oar_voxels,
    size_t d_rows, size_t num_beamlets): PTV_voxels(ptv_voxels),
    OAR_voxels(oar_voxels), D_rows(d_rows), numBeamlets(num_beamlets),
    alpha(1.0f), beta(0.0f)
{
    checkCudaErrors(cudaStreamCreate(&this->stream1));
    checkCudaErrors(cudaStreamCreate(&this->stream2));
    checkCudaErrors(cudaStreamCreate(&this->stream3));
    checkCudaErrors(cudaStreamCreate(&this->stream4));
    checkCudaErrors(cudaStreamCreate(&this->stream5));

    arrayInit(this->Ax, this->PTV_voxels + this->OAR_voxels);
    arrayInit(this->prox1, this->PTV_voxels);
    arrayInit(this->prox2, this->PTV_voxels + this->OAR_voxels);
    arrayInit(this->term3, this->OAR_voxels);
    arrayInit(this->term4, this->D_rows);
    arrayInit(this->prox4, this->D_rows);

    arrayInit(this->sumProx1, this->prox1.size);
    arrayInit(this->sumProx2, this->prox2.size);
    arrayInit(this->sumTerm3, this->term3.size);
    arrayInit(this->sumProx4, this->prox4.size);
    arrayInit(this->sumProx4Term4, this->prox4.size);

    arrayInit(this->grad_term1_input, this->PTV_voxels + this->OAR_voxels);
    arrayInit(this->grad_term1_output, this->numBeamlets);
    arrayInit(this->grad_term2_input, this->D_rows);
    arrayInit(this->grad_term2_output, this->numBeamlets);
}

IMRT::eval_grad::~eval_grad() {
    if (this->stream1)
        checkCudaErrors(cudaStreamDestroy(this->stream1));
    if (this->stream2)
        checkCudaErrors(cudaStreamDestroy(this->stream2));
    if (this->stream3)
        checkCudaErrors(cudaStreamDestroy(this->stream3));
    if (this->stream4)
        checkCudaErrors(cudaStreamDestroy(this->stream4));
    if (this->stream5)
        checkCudaErrors(cudaStreamDestroy(this->stream5));
}


float IMRT::eval_grad::evaluate(
    const MatCSR64& A, const MatCSR64& ATrans,
    const MatCSR64& D, const MatCSR64& DTrans,
    const array_1d<float>& x, array_1d<float>& grad, float gamma,
    const cusparseHandle_t& handle_cusparse,
    const cublasHandle_t& handle_cublas,
    
    const float* maxDose,
    const float* minDoseTarget,
    const float* minDoseTargetWeights,
    const float* maxWeightsLong,
    const float* OARWeightsLong,
    float eta) {
    #if sanityCheck
        if (A.numCols != x.size || A.numRows != this->Ax.size ||
            D.numCols != x.size || D.numRows != this->term4.size ||
            x.size != grad.size) {
            std::cerr << "The size of the matrices A, D and the "
                "size of the input vector x are incompatible." << std::endl;
            exit(EXIT_FAILURE);
        }
    #endif

    checkCusparse(cusparseSpMV(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &this->alpha, A.matA, x.vec, &this->beta, this->Ax.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, A.d_buffer_spmv));

    checkCusparse(cusparseSpMV(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &this->alpha, D.matA, x.vec, &this->beta, this->term4.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, D.d_buffer_spmv));

    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blockSize(256, 1, 1);
    dim3 gridSize((this->prox1.size + blockSize.x - 1) / blockSize.x, 1, 1);
    d_calc_prox1<<<gridSize, blockSize, 0, this->stream1>>>(
        this->prox1.data, this->Ax.data, minDoseTarget, this->prox1.size);
    
    gridSize.x = (this->prox2.size + blockSize.x - 1) / blockSize.x;
    d_calc_prox2<<<gridSize, blockSize, 0, this->stream2>>>(
        this->prox2.data, this->Ax.data, maxDose, this->prox2.size);

    checkCudaErrors(cudaMemcpyAsync(this->term3.data, this->Ax.data + this->PTV_voxels,
        this->OAR_voxels * sizeof(float), cudaMemcpyDeviceToDevice, this->stream3));

    gridSize.x = (this->D_rows + blockSize.x - 1) / blockSize.x;
    d_prox1Norm<<<gridSize, blockSize, 0, this->stream4>>>(
        this->prox4.data, this->term4.data, gamma, this->D_rows);

    gridSize.x = (this->prox1.size + blockSize.x - 1) / blockSize.x;
    d_ATimesBSquare<<<gridSize, blockSize, 0, this->stream1>>>(
        this->sumProx1.data, minDoseTargetWeights, this->prox1.data, this->prox1.size);
    
    gridSize.x = (this->prox2.size + blockSize.x - 1) / blockSize.x;
    d_ATimesBSquare<<<gridSize, blockSize, 0, this->stream2>>>(
        this->sumProx2.data, maxWeightsLong, this->prox2.data, this->prox2.size);

    gridSize.x = (this->term3.size + blockSize.x - 1) / blockSize.x;
    d_ATimesBSquare<<<gridSize, blockSize, 0, this->stream3>>>(
        this->sumTerm3.data, OARWeightsLong, this->term3.data, this->term3.size);

    gridSize.x = (this->prox4.size + blockSize.x - 1) / blockSize.x;
    d_calcSumProx4<<<gridSize, blockSize, 0, this->stream4>>>(
        this->sumProx4.data, this->prox4.data, this->prox4.size);
    
    d_calcProx4Term4<<<gridSize, blockSize, 0, this->stream5>>>(
        this->sumProx4Term4.data, this->prox4.data, this->term4.data, this->prox4.size);

    checkCudaErrors(cudaDeviceSynchronize());
    float sum1, sum2, sum3, sum4, sum5;
    checkCublas(cublasSasum(handle_cublas, this->sumProx1.size,
        this->sumProx1.data, 1, &sum1));
    checkCublas(cublasSasum(handle_cublas, this->sumProx2.size,
        this->sumProx2.data, 1, &sum2));
    checkCublas(cublasSasum(handle_cublas, this->sumTerm3.size,
        this->sumTerm3.data, 1, &sum3));
    checkCublas(cublasSasum(handle_cublas, this->sumProx4.size,
        this->sumProx4.data, 1, &sum4));
    checkCublas(cublasSasum(handle_cublas, this->sumProx4Term4.size,
        this->sumProx4Term4.data, 1, &sum5));

    
    // calculate gradient
    gridSize.x = (this->PTV_voxels + this->OAR_voxels + blockSize.x - 1) / blockSize.x;
    d_calc_grad_term1_input<<<gridSize, blockSize, 0, this->stream1>>> (
        this->grad_term1_input.data,
        this->PTV_voxels, minDoseTargetWeights, this->prox1.data,
        this->OAR_voxels, OARWeightsLong, this->term3.data,
        maxWeightsLong, this->prox2.data);
    
    gridSize.x = (this->D_rows + blockSize.x - 1) / blockSize.x;
    d_calc_grad_term2_input<<<gridSize, blockSize, 0, this->stream2>>> (
        this->grad_term2_input.data, this->term4.data,
        this->prox4.data, eta / gamma, this->D_rows);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCusparse(cusparseSpMV(
        handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &this->alpha, ATrans.matA,
        this->grad_term1_input.vec, &this->beta, grad_term1_output.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, ATrans.d_buffer_spmv));
    
    checkCusparse(cusparseSpMV(
        handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &this->alpha, DTrans.matA,
        this->grad_term2_input.vec, &this->beta, this->grad_term2_output.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, DTrans.d_buffer_spmv));
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    gridSize.x = (this->numBeamlets + blockSize.x - 1) / blockSize.x;
    d_elementWiseAdd<<<gridSize, blockSize>>>
        (grad.data, this->grad_term1_output.data,
        this->grad_term2_output.data, this->numBeamlets);

    checkCudaErrors(cudaDeviceSynchronize());
    
    float result = 0.5f * (sum1 + sum2 + sum3)
        +  eta * (sum4 + 0.5f / gamma * sum5);
    return result;
}


void IMRT::copy_array_1d(array_1d<float>& a, const array_1d<float>& b) {
    // sanity check
    if (a.size != b.size) {
        std::cerr << "the sizes of the input arrays, a and b, are expect to be the same." << std::endl;
    }
    checkCudaErrors(cudaMemcpy(a.data, b.data, a.size*sizeof(float), cudaMemcpyDeviceToDevice));
}


void IMRT::linearComb_array_1d(float alpha, const array_1d<float>& a,
    float beta, const array_1d<float>& b, array_1d<float>& c) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (a.size + blockSize.x - 1) / blockSize.x;
    d_linearComb<<<gridSize, blockSize>>>(c.data, alpha, a.data, beta, b.data, a.size);
}


__global__ void
IMRT::d_linearComb(float* c, float alpha, float* a, float beta, float* b, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    c[idx] = alpha * a[idx] + beta * b[idx];
}

__global__ void
IMRT::d_calc_grad_term1_input(float* output,
    size_t PTV_voxels, const float* minDoseTargetWeights, float* prox1Data,
    size_t OAR_voxels, const float* OARWeightsLong, float* term3Data,
    const float* maxWeightsLong, float* prox2Data) {

    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    float value1;
    if (idx < PTV_voxels)
        value1 = minDoseTargetWeights[idx] * prox1Data[idx];
    else if (idx < PTV_voxels + OAR_voxels){
        size_t idx_2 = idx - PTV_voxels;
        value1 = OARWeightsLong[idx_2] * term3Data[idx_2];
    } else
        return;

    float value2 = maxWeightsLong[idx] * prox2Data[idx];
    output[idx] = value1 + value2;
}

__global__ void
IMRT::d_calc_grad_term2_input(float* output, float* term4Data,
    float* prox4Data, float eta_over_gamma, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    output[idx] = (term4Data[idx] - prox4Data[idx]) * eta_over_gamma;
}

__global__ void
IMRT::d_elementWiseAdd(float* c, float* a, float* b, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    c[idx] = a[idx] + b[idx];
}


bool IMRT::arrayInit_group1(const std::vector<array_1d<float>*>& array_group1,
    size_t numBeamlets_max) {
    for (auto* ptr : array_group1) {
        arrayInit(*ptr, numBeamlets_max);
    }
    // randomize xkm1
    arrayRand01(*array_group1[0]);
    // set zero values for vkm1
    checkCudaErrors(cudaMemset(array_group1[1]->data, 0,
        array_group1[1]->size * sizeof(float)));
    return 0;
}

bool IMRT::bufferAllocate(MatCSR64& target, const array_1d<float>& input,
    const array_1d<float>& output, const cusparseHandle_t& handle) {
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, target.matA, input.vec, &beta, output.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
    ));
    checkCudaErrors(cudaMalloc(&(target.d_buffer_spmv), bufferSize));
    return 0;
}


bool IMRT::DimReduction(
    std::vector<int>& activeBeams, Eigen::VectorXf& beamWeights_cpu,
    Eigen::VectorXf& xkm1_cpu, Eigen::VectorXf& vkm1_cpu,
    const array_1d<float>& beamWeights, const array_1d<float>& xkm1,
    const array_1d<float>& vkm1, const std::vector<float>& nrm_cpu,
    int numActiveBeamsStrict, const std::vector<MatCSR_Eigen>& VOIMatrices
) {
    Eigen::VectorXf beamWeights_copy(beamWeights.size),
        xkm1_copy(xkm1.size), vkm1_copy(vkm1.size);
    checkCudaErrors(cudaMemcpy(beamWeights_copy.data(), beamWeights.data,
        beamWeights.size*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(xkm1_copy.data(), xkm1.data,
        xkm1.size*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(vkm1_copy.data(), vkm1.data,
        vkm1.size*sizeof(float), cudaMemcpyDeviceToHost));

    size_t numBeams_old = activeBeams.size();
    if (beamWeights.size != numBeams_old
        || nrm_cpu.size() != numBeams_old) {
        std::cerr << "The sizes of activeBeams, beamWeights, "
            "and nrm_cpu should be equal" << std::endl;
        return 1;
    }

    // localIdx, globalIdx, dimension, xkm1_segment, vkm1_segment, beamWeight, norm
    std::vector<std::tuple<int, int, int, Eigen::VectorXf, Eigen::VectorXf,
        float, float>> rankings(numBeams_old);
    size_t vector_offset = 0;
    for (int i=0; i<numBeams_old; i++) {
        int globalIdx = activeBeams[i];
        int dimension = VOIMatrices[globalIdx].getCols();
        std::get<0>(rankings[i]) = i;
        std::get<1>(rankings[i]) = globalIdx;
        std::get<2>(rankings[i]) = dimension;
        std::get<3>(rankings[i]) = xkm1_copy.segment(vector_offset, dimension);
        std::get<4>(rankings[i]) = vkm1_copy.segment(vector_offset, dimension);
        vector_offset += dimension;
        std::get<5>(rankings[i]) = beamWeights_copy(i);
        std::get<6>(rankings[i]) = nrm_cpu[i];
    }

    std::sort(rankings.begin(), rankings.end(),
        [](const std::tuple<int, int, int, Eigen::VectorXf, Eigen::VectorXf, float, float>& a,
            std::tuple<int, int, int, Eigen::VectorXf, Eigen::VectorXf, float, float>& b) {
                return std::get<6>(a) > std::get<6>(b);});
    
    rankings.resize(numActiveBeamsStrict);
    activeBeams.resize(numActiveBeamsStrict);
    beamWeights_cpu.resize(numActiveBeamsStrict);
    int dimension_new = 0;
    for (int i=0; i<numActiveBeamsStrict; i++) {
        activeBeams[i] = std::get<1>(rankings[i]);
        beamWeights_cpu(i) = std::get<5>(rankings[i]);
        dimension_new += std::get<2>(rankings[i]);
    }
    xkm1_cpu.resize(dimension_new);
    vkm1_cpu.resize(dimension_new);
    vector_offset = 0;
    for (int i=0; i<numActiveBeamsStrict; i++) {
        int dimension = std::get<2>(rankings[i]);
        const Eigen::VectorXf xkm1_segment = std::get<3>(rankings[i]);
        const Eigen::VectorXf vkm1_segment = std::get<4>(rankings[i]);
        xkm1_cpu.segment(vector_offset, dimension) = xkm1_segment;
        vkm1_cpu.segment(vector_offset, dimension) = vkm1_segment;
        vector_offset += dimension;
    }
    return 0;
}