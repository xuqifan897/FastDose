#ifndef __TermaComp_cuh__
#define __TermaComp_cuh__
#include "fastdose.cuh"
#include "cuda_runtime.h"

namespace fastdose {
    bool TermaCompute(BEAM_d& beam_d, DENSITY_d& density_d, SPECTRUM_h& spectrum_h, cudaStream_t stream=0);

    __global__ void
    d_TermaCompute(
        d_BEAM_d beam_d,
        float* fluence_map,
        cudaPitchedPtr TermaBEVPitch,
        cudaPitchedPtr DenseBEVPitch,
        cudaTextureObject_t densityTex,
        float3 voxel_size,
        int nkern
    );

    bool TermaComputeCollective(
        std::vector<BEAM_d>& beams,
        DENSITY_d& density_d,
        SPECTRUM_h& spectrum_h,
        cudaStream_t stream=0
    );

    __global__ void
    d_TermaComputeCollective(
        d_BEAM_d* beams,
        float** fluence_maps,
        cudaPitchedPtr* TermaBEVPitch_array,
        cudaPitchedPtr* DenseBEVPitch_array,
        cudaTextureObject_t densityTex,
        float3 voxel_size,
        int nkern
    );

    void BEV2PVCS(
        BEAM_d& beam_d,
        DENSITY_d& density_d,
        cudaPitchedPtr& PitchedOutput,
        cudaTextureObject_t BEVTex,
        cudaStream_t stream=0
    );

    __global__ void
    d_BEV2PVCS(
        d_BEAM_d beam_d,
        cudaPitchedPtr PitchedArray,
        cudaTextureObject_t BEVTex,
        uint3 DenseDim,
        float3 voxel_size
    );

    bool test_TermaCompute(BEAM_d& beam_d, DENSITY_d& density_d, SPECTRUM_h& spectrum_h,
        const std::string& outputFolder);

    bool profile_TermaCompute(std::vector<BEAM_d>& beams_d,
        DENSITY_d& density_d, SPECTRUM_h& spectrum, const std::string& outputFolder);
}

#endif