#ifndef __GEOMKERNELS_CUH__
#define __GEOMKERNELS_CUH__

#include "macros.h"

static __device__ float3 d_rotateAroundAxisAtOriginRHS(
    const float3& p, const float3& r, const float& t
) {
    /* derivation
    here the rotation axis r is a unit vector.
    So the vector to be rotate can be decomposed into 2 components.
    1. the component along r: z = (p \cdot r) r, which is of module |p|\cos\theta
    2. the component perpendicular to r: x = p - (p \cdot r) r, which is of module |p|\sin\theta
    To rotate it, we introduce a third component, which is y = r \times p, which is of module |p|\sin\theta, the same as above.
    So the result should be z + x\cos t + y \sin t, or
        (1-\cos t)(p \cdot r)r  +  p\cos t  +  (r \times p)\sin t
    */
    float sptr, cptr;
    fast_sincosf(t, &sptr, &cptr);
    float one_minus_cost = 1 - cptr;
    float p_dot_r = p.x * r.x + p.y * r.y + p.z * r.z;
    float first_term_coeff = one_minus_cost * p_dot_r;
    float3 result {
        first_term_coeff * r.x  +  cptr * p.x  +  sptr * (r.y * p.z - r.z * p.y),
        first_term_coeff * r.y  +  cptr * p.y  +  sptr * (r.z * p.x - r.x * p.z),
        first_term_coeff * r.z  +  cptr * p.z  +  sptr * (r.x * p.y - r.y * p.x) 
    };
    return result;
}

static __device__ float3 d_rotateBeamAtOriginRHS(
    const float3& vec, const float& theta, const float& phi, const float& coll
) {
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                          // couch rotation
    float3 tmp = d_rotateAroundAxisAtOriginRHS(vec, rotation_axis, -theta);          // gantry rotation
    return d_rotateAroundAxisAtOriginRHS(tmp, make_float3(0.f, 1.f, 0.f), phi+coll); // coll rotation + correction
}

// convert BEV coords to PVCS coords
static __device__ float3 d_inverseRotateBeamAtOriginRHS(
    const float3& vec, const float& theta, const float& phi, const float& coll
) {
    // invert what was done in forward rotation
    float3 tmp = d_rotateAroundAxisAtOriginRHS(vec, make_float3(0.f, 1.f, 0.f), -(phi+coll)); // coll rotation + correction
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                                   // couch rotation
    return d_rotateAroundAxisAtOriginRHS(tmp, rotation_axis, theta);                          // gantry rotation
}

#endif