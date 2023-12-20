#ifndef __DOSECOMP_CUH__
#define __DOSECOMP_CUH__
#include "beam.cuh"

namespace fastdose {
    bool DoseComputeCollective(
        size_t fmap_npixels,
        size_t n_beams,
        d_BEAM_d* d_beams,
        float** TermaBEV_array,
        float** DensityBEV_array,
        float** DoseBEV_array,
        int nTheta,
        int nPhi,
        cudaStream_t stream
    );

    __global__ void
    d_DoseComputeCollective(
        d_BEAM_d* beams,
        float** TermaBEV_array,
        float** DensityBEV_array,
        float** DoseBEV_array,
        int nTheta,
        int nPhi
    );

    bool test_DoseComputeCollective(std::vector<BEAM_d>& beams,
        const std::string& outputFolder, const KERNEL_h& kernel_h, cudaStream_t stream=0);
}

#endif