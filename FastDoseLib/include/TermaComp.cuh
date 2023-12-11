#ifndef __TermaComp_cuh__
#define __TermaComp_cuh__
#include "fastdose.cuh"
#include "cuda_runtime.h"

namespace fastdose {
    bool TermaComputeCollective(
        size_t fmap_npixels,
        size_t n_beams,
        d_BEAM_d* beams,
        float** fluence_array,
        float** TermaBEV_array,
        float** DensityBEV_array,
        DENSITY_d& density_d,
        SPECTRUM_h& spectrum_h,
        cudaStream_t stream=0
    );

    __global__ void
    d_TermaComputeCollective(
        d_BEAM_d* beams,
        float** fluence_maps,
        float** TermaBEV_array,
        float** DenseBEV_array,
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

    bool test_TermaComputeCollective(std::vector<BEAM_d>& beams,
        DENSITY_d& density_d, SPECTRUM_h& spectrum,
        const std::string& outputFolder, cudaStream_t stream=0);
}

#endif