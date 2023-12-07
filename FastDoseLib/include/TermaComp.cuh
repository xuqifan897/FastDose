#ifndef __TermaComp_cuh__
#define __TermaComp_cuh__
#include "fastdose.cuh"
#include "cuda_runtime.h"

namespace fastdose {
    bool TermaCompute(BEAM_d& beam_d, DENSITY_d& density_d, cudaStream_t stream=0);
    __global__ void
    d_TermaCompute(BEAM_d beam_d, DENSITY_d density_d);
}

#endif