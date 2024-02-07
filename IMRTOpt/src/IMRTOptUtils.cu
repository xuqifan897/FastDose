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


IMRT::eval_g::eval_g(size_t ptv_voxels, size_t oar_voxels) {
    this->PTV_voxels = ptv_voxels;
    this->OAR_voxels = oar_voxels;

    checkCudaErrors(cudaMalloc((void**)this->Ax.data, (ptv_voxels+oar_voxels)*sizeof(float)));
    this->Ax.size = ptv_voxels + oar_voxels;
    checkCusparse(cusparseCreateDnVec(&this->Ax.vec, this->Ax.size, this->Ax.data, CUDA_R_32F));

    // checkCudaErrors()
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