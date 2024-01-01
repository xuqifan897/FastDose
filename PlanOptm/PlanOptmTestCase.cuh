#ifndef __PLANOPTMTESTCASE_CUH__
#define __PLANOPTMTESTCASE_CUH__
#include "fastdose.cuh"
#include "PlanOptmBeamBundle.cuh"
#include "cuda_runtime.h"

#include <vector>

namespace PlanOptm {
    bool testCase(
        fastdose::DENSITY_h& density_h,
        fastdose::DENSITY_d& density_d,
        fastdose::SPECTRUM_h& spectrum_h,
        fastdose::KERNEL_h& kernel_h,
        cudaStream_t stream=0
    );

    bool beamBundleTestCase(
        std::vector<BeamBundle> beam_bundles,
        fastdose::DENSITY_h& density_h,
        fastdose::DENSITY_d& density_d,
        fastdose::SPECTRUM_h& spectrum_h,
        fastdose::KERNEL_h& kernel_h,
        cudaStream_t stream=0
    );

    bool textureMemTest();
}

#endif