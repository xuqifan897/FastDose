#include "utils.cuh"
#include <iostream>
#include <random>

namespace fd = fastdose;

std::ostream& operator<<(std::ostream& os, const uint2& obj) {
    os << "(" << obj.x << ", " << obj.y << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const float2& obj) {
    os << "(" << obj.x << ", " << obj.y << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const int2& obj) {
    os << "(" << obj.x << ", " << obj.y << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const uint3& obj) {
    os << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const float3& obj) {
    os << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const int3& obj) {
    os << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const dim3& obj) {
    os << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")";
    return os;
}

void fd::h_readTexture3D(float* output, cudaTextureObject_t input,
        int width, int height, int depth
) {
    dim3 blockSize{4, 4, 4};
    dim3 gridSize{
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z
    };
    readTexture3D<<<gridSize, blockSize>>>(output, input, width, height, depth);
}

__global__ void
fd::readTexture3D(float* output, cudaTextureObject_t input, int width, int height, int depth) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx_x>=width || idx_y>=height || idx_z>=depth)
        return;
    size_t idx = idx_x + width * (idx_y + height * idx_z);
    float fidx_x = idx_x + .5;
    float fidx_y = idx_y + .5;
    float fidx_z = idx_z + .5;
    output[idx] = tex3D<float>(input, fidx_x, fidx_y, fidx_z);
}

float fd::rand01() {
    return (float)std::rand() / RAND_MAX;
}

void fd::pitched2contiguous(std::vector<float>& output, 
    std::vector<float>& pitched_input, int width, 
    int height, int depth, int pitch
) {
    for (int i=0; i<depth; i++) {
        for (int j=0; j<height; j++) {
            for (int k=0; k<width; k++) {
                size_t idx_pitch = k + pitch * (j + i * height);
                size_t idx_output = k + width * (j + i * height);
                output[idx_output] = pitched_input[idx_pitch];
            }
        }
    }
}


bool fd::showDeviceProperties(int deviceIdx) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    std::cout << "Device " << deviceIdx << ": " << deviceProp.name << std::endl;
    std::cout << "    Global Memory Size: " << 
        deviceProp.totalGlobalMem << " bytes" << std::endl;
    std::cout << "    Shared Memory Size per Block: " << 
        deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "    Number of Registers per Block: " <<
        deviceProp.regsPerBlock << " bytes" << std::endl;
    std::cout << "    Max Blocks per Multiprocessor: " <<
        deviceProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "    Maximum resident threads per Multiprocessor: " <<
        deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "    Shared memory available per Multiprocessor: " << 
        deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "    Number of Threads per Warp: " <<
        deviceProp.warpSize << std::endl;
    std::cout << "    Number of SMs in total: " << 
        deviceProp.multiProcessorCount << std::endl;
    int* maxTexture3D = deviceProp.maxTexture3D;
    std::cout << "    Maximum 3D texture dimensions: (" << maxTexture3D[0] << ", " 
        << maxTexture3D[1] << ", " << maxTexture3D[2] << ")" << std::endl;
    std::cout << std::endl;
    return 0;
}