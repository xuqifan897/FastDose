#include <iostream>
#include "IMRTOptimize.cuh"

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

    checkCublas(cublasCreate(&this->cublasHandle));
}

void IMRT::arrayInit(array_1d<float>& arr, size_t size) {
    arr.size = size;
    checkCudaErrors(cudaMalloc((void**)&arr.data, arr.size * sizeof(float)));
    checkCusparse(cusparseCreateDnVec(&arr.vec, arr.size, arr.data, CUDA_R_32F));
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
    if (this->cublasHandle)
        checkCublas(cublasDestroy(this->cublasHandle));
}


float IMRT::eval_g::evaluate(const MatCSR64& A, const MatCSR64& D,
    const array_1d<float>& x, float* maxDose, float gamma,
    const cusparseHandle_t& handle,

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

    checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &this->alpha, A.matA, x.vec, &this->beta, this->Ax.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, A.d_buffer_spmv));

    checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
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
    checkCublas(cublasSasum(this->cublasHandle, this->sumProx1.size,
        this->sumProx1.data, 1, &sum1));
    checkCublas(cublasSasum(this->cublasHandle, this->sumProx2.size,
        this->sumProx2.data, 1, &sum2));
    checkCublas(cublasSasum(this->cublasHandle, this->sumTerm3.size,
        this->sumTerm3.data, 1, &sum3));
    checkCublas(cublasSasum(this->cublasHandle, this->sumProx4.size,
        this->sumProx4.data, 1, &sum4));
    checkCublas(cublasSasum(this->cublasHandle, this->sumProx4Term4.size,
        this->sumProx4Term4.data, 1, &sum5));
    
    float result = 0.5f * sum1  +  0.5f * sum2  +  0.5f * sum3
        +  eta * sum4  +  0.5f / gamma * sum5;
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