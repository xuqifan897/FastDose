#ifndef __BEAM_CUH__
#define __BEAM_CUH__
#include <vector>
#include <iostream>
#include <helper_cuda.h>

#include "density.cuh"

namespace fastdose {
    class BEAM_h {
    public:
        float& operator()(int x, int y) {
            return fluence[y * this->fmap_size.x + x];
        }
        const float& operator()(int x, int y) const {
            return fluence[y * this->fmap_size.x + x];
        }
        friend std::ostream& operator<<(std::ostream& os, const BEAM_h& obj);

        float3 isocenter;     // cm
        float2 beamlet_size;  // cm
        uint2 fmap_size;
        float sad;            // cm
        float3 angles;        // azimuth, zenith, collimator
        float long_spacing;   // cm
        std::vector<float> fluence;

        // This function calculates the range of the beam, i.e., the lower bound and upper bound.
        bool calc_range(const DENSITY_h& density_h);
        float lim_min;
        float lim_max;
        uint long_dim;
    };

    class BEAM_d {
    public:
        __inline__ __device__ float& operator()(int x, int y) {
            return fluence[y * this->fmap_size.x + x];
        }

        __inline__ __host__ BEAM_d() {
            this->TermaBEVPitch.ptr = nullptr;
            this->fluence = nullptr;
        }

        __inline__ __host__ ~BEAM_d() {
            if (this->fluence != nullptr)
                checkCudaErrors(cudaFree(this->fluence));
            if (this->TermaBEVPitch.ptr != nullptr)
                checkCudaErrors(cudaFree(this->TermaBEVPitch.ptr));
        }

        float3 isocenter;
        float2 beamlet_size;
        uint2 fmap_size;
        float sad;
        float3 angles;
        float long_spacing;
        float* fluence=nullptr;

        float lim_min;
        float lim_max;
        uint long_dim;

        cudaPitchedPtr TermaBEVPitch;
    };
    std::ostream& operator<<(std::ostream& os, const BEAM_h& obj);
    void beam_h2d(BEAM_h& beam_h, BEAM_d& beam_d);
    void beam_d2h(BEAM_d& beam_d, BEAM_h& beam_h);
    void test_beam_io();
    void test_TermaBEVPitch(BEAM_d& beam_d);
}

#endif