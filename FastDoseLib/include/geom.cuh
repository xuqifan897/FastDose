#include "macros.h"
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
}