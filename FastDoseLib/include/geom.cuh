#include "macros.h"
// To comply with the definition of rotation in the baseline code, 
// inverseRotateBeamAtOriginRHS is to rotate the vector from the BEV to PVCS
// rotateBeamAtOriginRHS is vice versa

__host__ __device__ float3 rotateAroundAxisAtOriginRHS(
    const float3& p, const float3& r, const float& t
);

__host__ __device__ float3 rotateBeamAtOriginRHS(
    const float3& vec, const float& theta, const float& phi, const float& coll
);

__host__ __device__ float3 inverseRotateBeamAtOriginRHS(
    const float3& vec, const float& theta, const float& phi, const float& coll
);

namespace fastdose {
    void test_rotateAroundAxisAtOrigin();
}