#ifndef __DENSITY_CUH__
#define __DENSITY_CUH__
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include <vector>
#include <iostream>

namespace fastdose {
    class DENSITY_h {
    public:
        float& operator()(int x, int y, int z) {
            size_t idx = x + VolumeDim.x * (y + VolumeDim.y * z);
            return density[idx];
        }

        const float& operator()(int x, int y, int z) const {
            size_t idx = x + VolumeDim.x * (y + VolumeDim.y * z);
            return density[idx];
        }

        friend std::ostream& operator<<(std::ostream& os, const DENSITY_h& obj);

        float3 VoxelSize;  // in cm
        uint3 VolumeDim;
        uint3 BBoxStart;
        uint3 BBoxDim;
        std::vector<float> density;
    };

    class DENSITY_d {
    public:
        ~DENSITY_d();

        float3 VoxelSize;  // in cm
        uint3 VolumeDim;
        uint3 BBoxStart;
        uint3 BBoxDim;
        
        cudaArray* densityArray = nullptr;
        cudaTextureObject_t densityTex;
    };

    std::ostream& operator<<(std::ostream& os, const DENSITY_h& obj);
    void density_h2d(DENSITY_h& density_h, DENSITY_d& density_d);
    void test_density();
}

#endif