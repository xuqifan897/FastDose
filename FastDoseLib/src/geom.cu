#include "geom.cuh"
#include "utils.cuh"
#include "math_constants.h"
#include <iostream>
namespace fd = fastdose;

__host__ __device__ float3 rotateAroundAxisAtOriginRHS(
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

__host__ __device__ float3 rotateBeamAtOriginRHS(
    const float3& vec, const float& theta, const float& phi, const float& coll
) {
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                          // couch rotation
    float3 tmp = rotateAroundAxisAtOriginRHS(vec, rotation_axis, -theta);          // gantry rotation
    return rotateAroundAxisAtOriginRHS(tmp, make_float3(0.f, 1.f, 0.f), phi+coll); // coll rotation + correction
}

// convert BEV coords to PVCS coords
__host__ __device__ float3 inverseRotateBeamAtOriginRHS(
    const float3& vec, const float& theta, const float& phi, const float& coll
) {
    // invert what was done in forward rotation
    float3 tmp = rotateAroundAxisAtOriginRHS(vec, make_float3(0.f, 1.f, 0.f), -(phi+coll)); // coll rotation + correction
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                                   // couch rotation
    return rotateAroundAxisAtOriginRHS(tmp, rotation_axis, theta);                          // gantry rotation
}

void fd::test_rotateAroundAxisAtOrigin() {
    float3 p;
    float3 r;
    float t;
    float3 result;

    // case 1
    p = float3{3, 0, 0};
    r = float3{0, 0, 1};
    t = CUDART_PI_F / 3;
    result = rotateAroundAxisAtOriginRHS(p, r, t);
    std::cout << "Case 1" << std::endl << "Expected result: (1.5, 2,5981, 0)" << std::endl;
    std::cout << "Result: " << result << std::endl << std::endl;

    // case 2
    p = float3{3, 0, 0};
    float r_component = sqrtf32(1. / 3);
    r = float3{r_component, r_component, r_component};
    t = CUDART_PI_F * 2 / 3;
    result = rotateAroundAxisAtOriginRHS(p, r, t);
    std::cout << "Case 2" << std::endl << "Expected result: (0, 3, 0)" << std::endl;
    std::cout << "Result: " << result << std::endl;

    // test to see if rotateBeamAtOriginRHS and inverseRotateBeamAtOriginRHS are inverse of each other
    int ncases = 50;

#if false
    for (int i=0; i<ncases; i++) {
        // forward cases
        float3 vector{rand01(), rand01(), rand01()};
        float theta = (rand01() - 0.5) * 2 * CUDART_PI_F;
        float phi = (rand01() - 0.5) * 2 * CUDART_PI_F;
        float coll = (rand01() - 0.5) * 2 * CUDART_PI_F;
        
        float3 vector_PVCS = rotateBeamAtOriginRHS(vector, theta, phi, coll);
        float3 vector_BEV = inverseRotateBeamAtOriginRHS(vector_PVCS, theta, phi, coll);
        std::cout << "The original vector: " << vector << std::endl
            << "result: " << vector_BEV << std::endl << std::endl;
    }
#endif

    for (int i=0; i<ncases; i++) {
        // backward cases
        float3 vector{rand01(), rand01(), rand01()};
        float theta = (rand01() - 0.5) * 2 * CUDART_PI_F;
        float phi = (rand01() - 0.5) * 2 * CUDART_PI_F;
        float coll = (rand01() - 0.5) * 2 * CUDART_PI_F;

        float3 vector_BEV = inverseRotateBeamAtOriginRHS(vector, theta, phi, coll);
        float3 vector_PVCS = rotateBeamAtOriginRHS(vector_BEV, theta, phi, coll);
        std::cout << "The original vector: " << vector << std::endl
            << "result: " << vector_PVCS << std::endl << std::endl;
    }
}