#ifndef __DOSECOMP_CUH__
#define __DOSECOMP_CUH__
#include "beam.cuh"
#include "kernel.cuh"

#define XDebug false
#define DoseDebug true
#define ProbePixelIdx false
#define probeSize 16*16*1024

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
#if XDebug || DoseDebug
        , const std::string outputFolder
#endif
    );

    __global__ void
    d_DoseComputeCollective(
        d_BEAM_d* beams,
        float** TermaBEV_array,
        float** DensityBEV_array,
        float** DoseBEV_array,
        int nTheta,
        int nPhi
#if XDebug || DoseDebug
        , float* debugProbe
#endif
    );

    bool test_DoseComputeCollective(std::vector<BEAM_d>& beams,
        const std::string& outputFolder, const KERNEL_h& kernel_h, cudaStream_t stream=0);
}

#endif