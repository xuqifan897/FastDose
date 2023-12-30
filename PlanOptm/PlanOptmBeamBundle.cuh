#ifndef __PLANOPTMBEAMBUNDLE_CUH__
#define __PLANOPTMBEAMBUNDLE_CUH__
#include <vector>

#include "fastdose.cuh"

namespace PlanOptm {
    class BeamBundle {
    public:
        std::vector<fastdose::BEAM_h> beams_h;
        std::vector<fastdose::BEAM_d> beams_d;

        float3 isocenter;
        float2 beamletSize;
        int2 fluenceDim;
        float SAD;
        float3 angles;
        float longSpacing;

        int2 subFluenceDim;
        int2 subFluenceOn;

        bool beamletsInit();
        bool beamletInit(fastdose::BEAM_h& beam_h, int idx_x, int idx_y);
    };

    bool BeamBundleInit(std::vector<BeamBundle>& beam_bundle);
}

#endif