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
        float3 source;
    };

    class BEAM_d {
    public:
        __inline__ __device__ float& operator()(int x, int y) {
            return fluence[y * this->fmap_size.x + x];
        }

        __inline__ __host__ BEAM_d(): fluence(nullptr), TermaBEV(nullptr), DensityBEV(nullptr) {}

        __inline__ __host__ ~BEAM_d() {
            if (this->fluence != nullptr)
                checkCudaErrors(cudaFree(this->fluence));
            if (this->TermaBEV != nullptr)
                checkCudaErrors(cudaFree(this->TermaBEV));
            if (this->DensityBEV != nullptr)
                checkCudaErrors(cudaFree(this->DensityBEV));
        }

        float3 isocenter;
        float2 beamlet_size;
        uint2 fmap_size;
        float sad;
        float3 angles;
        float long_spacing;
        float* fluence;

        float lim_min;
        float lim_max;
        uint long_dim;
        float3 source;

        float* TermaBEV;
        size_t TermaBEV_pitch;
        float* DensityBEV;
        size_t DensityBEV_pitch;
        float* DoseBEV;
        size_t DoseBEV_pitch;
    };

    class d_BEAM_d {
        // This variable is to avoid the copy of TermaBEVPitch and fluence. Otherwise the same as above
        public:
        d_BEAM_d(const BEAM_d& old):
            isocenter(old.isocenter),
            beamlet_size(old.beamlet_size),
            fmap_size(old.fmap_size),
            sad(old.sad),
            angles(old.angles),
            long_spacing(old.long_spacing),

            lim_min(old.lim_min),
            lim_max(old.lim_max),
            long_dim(old.long_dim),
            source(old.source),
            TermaBEV_pitch(old.TermaBEV_pitch),
            DensityBEV_pitch(old.DensityBEV_pitch)
        {}

        float3 isocenter;
        float2 beamlet_size;
        uint2 fmap_size;
        float sad;
        float3 angles;
        float long_spacing;

        float lim_min;
        float lim_max;
        uint long_dim;
        float3 source;
        size_t TermaBEV_pitch;
        size_t DensityBEV_pitch;
    };

    std::ostream& operator<<(std::ostream& os, const BEAM_h& obj);
    bool beam_h2d(BEAM_h& beam_h, BEAM_d& beam_d);
    void beam_d2h(BEAM_d& beam_d, BEAM_h& beam_h);
    bool test_beam_io();
    void test_TermaBEVPitch(BEAM_d& beam_d);
}

#endif