#include "macros.h"
#include "beam.cuh"
// To comply with the definition of rotation in the baseline code, 
// inverseRotateBeamAtOriginRHS is to rotate the vector from the BEV to PVCS
// rotateBeamAtOriginRHS is vice versa

namespace fastdose{
    float3 rotateAroundAxisAtOriginRHS(
        const float3& p, const float3& r, const float& t
    );

    float3 rotateBeamAtOriginRHS(
        const float3& vec, const float& theta, const float& phi, const float& coll
    );

    float3 inverseRotateBeamAtOriginRHS(
        const float3& vec, const float& theta, const float& phi, const float& coll
    );

    void test_rotateAroundAxisAtOrigin();

    float3 angle2Vector(float theta, float phi);

    void test_angle2Vector();

    float3 nextPoint(const float3& currentLocation, const float3& direction, bool* flag);

    void test_nextPoint();

    float calcLineSeg(const float3& start, const float3& end, const d_BEAM_d& beam);

    bool test_calcLineSeg(const std::vector<d_BEAM_d>& h_beams);

    int3 calcCoords(const float3& mid_point, const d_BEAM_d& beam);

    bool test_calcCoords(const std::vector<d_BEAM_d>& h_beams);

    // BEV to PVCS interpolation, but with supersampling
    void BEV2PVCS_SuperSampling(
        BEAM_d& beam_d,
        DENSITY_d& density_d,
        cudaPitchedPtr& PitchedOutput,
        cudaTextureObject_t BEVTex,
        int ssfactor = 5,
        float extent = 2.0,  // to determine the range to interpolate dose
        cudaStream_t stream = 0
    );

    __global__ void
    d_BEV2PVCS_SuperSampling(
        d_BEAM_d beam_d,
        cudaPitchedPtr PitchedArray,
        cudaTextureObject_t BEVTex,
        uint3 ArrayDim,
        float3 voxel_size,
        int ssfactor,
        float extent);
}