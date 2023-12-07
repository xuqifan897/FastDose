#ifndef __INIT_H__
#define __INIT_H__
#include "fastdose.cuh"

namespace example {
    bool densityInit(fastdose::DENSITY_h& _density_h_, fastdose::DENSITY_d& _density_d_);
    void densityTest(fastdose::DENSITY_h& _density_h_, fastdose::DENSITY_d& _density_d_);

    bool beamsInit(std::vector<fastdose::BEAM_h>& beams_h,
        std::vector<fastdose::BEAM_d>& beams_d,
        fastdose::DENSITY_h& density_h);
    bool specInit(fastdose::SPECTRUM_h& spectrum_h);
}

#endif