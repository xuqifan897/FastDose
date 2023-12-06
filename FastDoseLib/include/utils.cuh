#ifndef __UTILS_H__
#define __UTILS_H__
#include <iostream>
#include <vector>
#include "cuda_runtime.h"

std::ostream& operator<<(std::ostream& os, const uint2& obj);
std::ostream& operator<<(std::ostream& os, const float2& obj);
std::ostream& operator<<(std::ostream& os, const int2& obj);

std::ostream& operator<<(std::ostream& os, const uint3& obj);
std::ostream& operator<<(std::ostream& os, const float3& obj);
std::ostream& operator<<(std::ostream& os, const int3& obj);
std::ostream& operator<<(std::ostream& os, const dim3& obj);

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& arr)
{
    os << "[";
    int N = arr.size();
    for (int i=0; i<N; i++)
    {
        os << arr[i];
        if (i < N-1)
            os << ", ";
    }
    os << "]";
    return os;
}

namespace fastdose {
    float rand01();
    void h_readTexture3D(float* output, cudaTextureObject_t input,
        int width, int height, int depth);
    __global__ void
    readTexture3D(float* output, cudaTextureObject_t input,
        int width, int height, int depth);
}

#endif