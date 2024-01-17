#ifndef __IMRTDEBUG_CUH__
#define __IMRTDEBUG_CUH__
#include <vector>
#include "IMRTBeamBundle.cuh"
#include "fastdose.cuh"

namespace IMRT {
    bool doseCalcDebug (
        std::vector<BeamBundle>& beam_bundles,
        fastdose::DENSITY_d& density_d,
        fastdose::SPECTRUM_h& spectrum_h,
        fastdose::KERNEL_h& kernel_h,
        cudaStream_t stream = 0
    );
}

#endif