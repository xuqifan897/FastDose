#ifndef __PLANOPTMINIT_CUH__
#define __PLANOPTMARGS_CUH__
#include "fastdose.cuh"

namespace PlanOptm {
    bool densityInit(fastdose::DENSITY_h& density_h, fastdose::DENSITY_d& density_d);
    bool specInit(fastdose::SPECTRUM_h& spectrum_h);
    bool kernelInit(fastdose::KERNEL_h& kernel_h);
}

#endif