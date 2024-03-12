#ifndef __IMRTBEAMBUNDLE_CUH__
#define __IMRTBEAMBUNDLE_CUH__
#include <vector>

#include "fastdose.cuh"
#include "IMRTInit.cuh"

namespace IMRT {
    class BeamBundle {
    public:
        // basic properties
        float3 isocenter;
        float2 beamletSize;
        int2 fluenceDim;
        float SAD;
        float3 angles;
        float longSpacing;

        // the properties for beamlet dose calculation
        int2 subFluenceDim;
        int2 subFluenceOn;

        // indicate whether a beamlet is on or off
        std::vector<bool> beamletFlag;
        std::vector<fastdose::BEAM_h> beams_h;
        std::vector<fastdose::BEAM_d> beams_d;

        bool beamletsInit(const fastdose::DENSITY_h& density_h);
        bool beamletInit(fastdose::BEAM_h& beam_h, int idx_x,
            int idx_y, const fastdose::DENSITY_h& density_h);
    };

    bool BeamBundleInit(std::vector<BeamBundle>& beam_bundles,
        const fastdose::DENSITY_h& density_h,
        const std::vector<StructInfo>& structs);

    bool beamletFlagInit(std::vector<BeamBundle>& beam_bundles,
        const std::vector<uint8_t>& PTV_mask,
        const fastdose::DENSITY_h& density_h,
        cudaStream_t stream=0);
    
    bool beamletFlagSave(const std::vector<BeamBundle>& beam_bundles,
        const std::string& resultFile);

    bool doseDataSave(const std::vector<BeamBundle>& beam_bundles,
        const std::string& resultFile);

    __global__ void
    d_beamletFlagInit(
        fastdose::d_BEAM_d* beams_d, uint8_t* fmapOn,
        cudaTextureObject_t PTVTex, float3 voxelSize,
        int superSampling
    );
}

#endif