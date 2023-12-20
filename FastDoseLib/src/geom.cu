#include "geom.cuh"
#include "beam.cuh"
#include "utils.cuh"
#include "math_constants.h"
#include "helper_math.cuh"
#include "macros.h"
#include <iostream>
#include <iomanip>
namespace fd = fastdose;

float3 fd::rotateAroundAxisAtOriginRHS(
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

float3 fd::rotateBeamAtOriginRHS(
    const float3& vec, const float& theta, const float& phi, const float& coll
) {
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                          // couch rotation
    float3 tmp = rotateAroundAxisAtOriginRHS(vec, rotation_axis, -theta);          // gantry rotation
    return rotateAroundAxisAtOriginRHS(tmp, make_float3(0.f, 1.f, 0.f), phi+coll); // coll rotation + correction
}

// convert BEV coords to PVCS coords
float3 fd::inverseRotateBeamAtOriginRHS(
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


float3 fd::angle2Vector(float theta, float phi) {
    float sin_theta, cos_theta, sin_phi, cos_phi;
    fast_sincosf(theta, &sin_theta, &cos_theta);
    fast_sincosf(phi, &sin_phi, &cos_phi);
    float3 result{
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta};
    return result;
}


void fd::test_angle2Vector() {
    int nSamples = 100;
    std::vector<float2> samples(nSamples);
    // Specify two special cases
    samples[0] = make_float2(0.f, 0.f);
    samples[1] = make_float2(CUDART_PIO4, 0.f);
    samples[2] = make_float2(CUDART_PI/6, CUDART_PI/6);
    for (int i=3; i<nSamples; i++) {
        samples[i] = make_float2 (
            rand01() * CUDART_PI,
            rand01() * CUDART_PI * 2
        );
    }

    std::vector<float3> results(nSamples);
    for (int i=0; i<nSamples; i++) {
        results[i] = angle2Vector(samples[i].x, samples[i].y);
        std::cout << "theta: " << samples[i].x << ", phi: " << 
            samples[i].y << ", vector: " << results[i] << std::endl;
    }
    std::cout << std::endl;
}


float3 fd::nextPoint(
    const float3& currentLocation,
    const float3& direction,
    bool* flag
) {
    float3 stepSize;

    if (direction.x > 0) {
        stepSize.x = (ceilf(currentLocation.x + eps_fastdose) - currentLocation.x) / direction.x;
    } else {
        stepSize.x = (floorf(currentLocation.x - eps_fastdose) - currentLocation.x) / direction.x;
    }

    if (direction.y > 0) {
        stepSize.y = (ceilf(currentLocation.y + eps_fastdose) - currentLocation.y) / direction.y;
    } else {
        stepSize.y = (floorf(currentLocation.y - eps_fastdose) - currentLocation.y) / direction.y;
    }

    if (direction.z > 0) {
        stepSize.z = (ceilf(currentLocation.z + eps_fastdose) - currentLocation.z) / direction.z;
    } else {
        stepSize.z = (floorf(currentLocation.z - eps_fastdose) - currentLocation.z) / direction.z;
    }

    float stepTake = fmin(fmin(stepSize.x, stepSize.y), stepSize.z);
    *flag = abs(stepTake - stepSize.z) < eps_fastdose;
    return currentLocation + stepTake * direction;
}


void fd::test_nextPoint() {
    int nSamples = 100;
    for (int i=0; i<nSamples; i++) {
        float3 origin{rand01(), rand01(), rand01()};
        float3 direction{
            2 * rand01() - 1,
            2 * rand01() - 1,
            2 * rand01() - 1
        };
        direction = normalize(direction);
        bool flag;
        float3 np = nextPoint(origin, direction, &flag);
        std::cout << std::setprecision(4) << "origin: " << origin << ", next point: " << np
            << ", direction: " << direction << ", flag:" << flag << std::endl;
    }
}


float fd::calcLineSeg(const float3& origin,
    const float3& dest, const d_BEAM_d& beam
) {
    // the two inputs origin and dest are all normalized w.r.t voxel size,
    // this function returns the physical distance that the line intersects
    // with the voxel
    float3 origin_physical {
        origin.x * beam.beamlet_size.x,

    };
}