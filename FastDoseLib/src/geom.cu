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


float fd::calcLineSeg(
    const float3& start,
    const float3& end,
    const d_BEAM_d& beam
) {
    // the two inputs origin and dest are all normalized w.r.t voxel size,
    // this function returns the physical distance that the line intersects
    // with the voxel
    float3 start_physical {
        (start.x - beam.fmap_size.x * 0.5f) * beam.beamlet_size.x,
        (start.y - beam.fmap_size.y * 0.5f) * beam.beamlet_size.y,
        start.z * beam.long_spacing
    };
    float start_physical_z = start_physical.z + beam.lim_min;
    float factor = start_physical_z / beam.sad;
    start_physical.x *= factor;
    start_physical.y *= factor;

    float3 end_physical {
        (end.x - beam.fmap_size.x * 0.5f) * beam.beamlet_size.x,
        (end.y - beam.fmap_size.y * 0.5f) * beam.beamlet_size.y,
        end.z * beam.long_spacing
    };
    float end_physical_z = end_physical.z + beam.lim_min;
    factor = end_physical_z / beam.sad;
    end_physical.x *= factor;
    end_physical.y *= factor;
    
    float3 diff = end_physical - start_physical;
    return sqrtf32(dot(diff, diff));
}


bool fd::test_calcLineSeg(const std::vector<d_BEAM_d>& h_beams) {
    int nSamples = 100;
    if (nSamples > h_beams.size()) {
        std::cerr << "Too much samples." << std::endl;
        return 1;
    }
    std::vector<std::pair<float3, float3>> cases(nSamples);
    // specify two specific cases. In the first case, both the begin
    // and the end points are on the isocenter plane. So the factor should be 1.
    float z_norm = (h_beams[0].sad - h_beams[0].lim_min) / h_beams[0].long_spacing;
    float3 begin = make_float3(
        rand01() * h_beams[0].fmap_size.x,
        rand01() * h_beams[0].fmap_size.y,
        z_norm);
    float3 end = make_float3(
        rand01() * h_beams[0].fmap_size.x,
        rand01() * h_beams[0].fmap_size.y,
        z_norm);
    cases[0] = std::make_pair(begin, end);

    // In the second case, both the points are on the line
    // connecting the source and the isocenter
    begin = make_float3(
        h_beams[1].fmap_size.x * 0.5f,
        h_beams[1].fmap_size.y * 0.5f,
        rand01() * (h_beams[1].lim_max - h_beams[1].lim_min) / h_beams[1].long_spacing
    );
    end = make_float3(
        begin.x,
        begin.y,
        rand01() * (h_beams[1].lim_max - h_beams[1].lim_min) / h_beams[1].long_spacing
    );
    cases[1] = std::make_pair(begin, end);

    // Then generate the remaining cases randomly
    for (int i=2; i<nSamples; i++) {
        begin = make_float3(
            rand01() * h_beams[i].fmap_size.x,
            rand01() * h_beams[i].fmap_size.y,
            rand01() * h_beams[i].long_dim);
        end = make_float3(
            rand01() * h_beams[i].fmap_size.x,
            rand01() * h_beams[i].fmap_size.y,
            rand01() * h_beams[i].long_dim);
        cases[i] = std::make_pair(begin, end);
    }

    // generate results
    for (int i=0; i<nSamples; i++) {
        const float3& begin = cases[i].first;
        const float3& end = cases[i].second;
        float distance = calcLineSeg(begin, end, h_beams[i]);

        if (i == 0 || i == 1) {
            float3 diff = end - begin;
            diff.x *= h_beams[0].beamlet_size.x;
            diff.y *= h_beams[0].beamlet_size.y;
            diff.z *= h_beams[0].long_spacing;
            float expected = sqrt(dot(diff, diff));
            std::cout << "Start: " << begin << ", end: " << end <<
                ", expected: " << expected << ", result: " << distance <<
                std::endl;
        }
    }
    std::cout << std::endl;
    return 0;
}


int3 fd::calcCoords(const float3& mid_point, const d_BEAM_d& beam) {
    float3 result;
    result.x = fmodf(mid_point.x, static_cast<float>(beam.fmap_size.x));
    result.y = fmodf(mid_point.y, static_cast<float>(beam.fmap_size.y));
    result.z = mid_point.z;  // it doesn't get beyond the range

    result.x = (result.x < 0) ? result.x + beam.fmap_size.x : result.x;
    result.y = (result.y < 0) ? result.y + beam.fmap_size.y : result.y;

    int3 return_value {
        static_cast<int>(floorf(result.x)),
        static_cast<int>(floorf(result.y)),
        static_cast<int>(floorf(result.z))
    };
    return return_value;
}


bool fd::test_calcCoords(const std::vector<d_BEAM_d>& h_beams) {
    int nSamples = 100;
    if (nSamples > h_beams.size()) {
        std::cerr << "Too much samples." << std::endl;
        return 1;
    }

    float base = -100.f;
    float range = 200.f;
    for (int i=0; i<nSamples; i++) {
        const d_BEAM_d & beam = h_beams[i];
        float3 mid_point {
            base + rand01() * range,
            base + rand01() * range,
            rand01() * beam.long_dim
        };
        int3 result = calcCoords(mid_point, beam);
        std::cout << "Point: " << mid_point << ", coords: " << result << std::endl;
    }
    std::cout << std::endl;
    return 0;
}


void fd::BEV2PVCS_SuperSampling(
    BEAM_d& beam_d,
    DENSITY_d& density_d,
    cudaPitchedPtr& PitchedOutput,
    cudaTextureObject_t BEVTex,
    int ssfactor,
    cudaStream_t stream
) {
    d_BEAM_d beam_input(beam_d);
    uint width = density_d.VolumeDim.x;
    uint height = density_d.VolumeDim.y;
    uint depth = density_d.VolumeDim.z;
    dim3 blockSize(16, 8, 4);
    dim3 gridSize{
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z
    };

    size_t sharedMemSize = blockSize.x * blockSize.y * blockSize.z * sizeof(float);
    d_BEV2PVCS_SuperSampling<<<gridSize, blockSize, sharedMemSize, stream>>>(
        beam_input,
        PitchedOutput,
        BEVTex,
        density_d.VolumeDim,
        density_d.VoxelSize,
        ssfactor
    );
}


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


__global__ void
fd::d_BEV2PVCS_SuperSampling(
    d_BEAM_d beam_d,
    cudaPitchedPtr PitchedArray,
    cudaTextureObject_t BEVTex,
    uint3 ArrayDim,
    float3 voxel_size,
    int ssfactor
) {
    uint3 idx = threadIdx + blockDim * blockIdx;
    if (idx.x >= ArrayDim.x || idx.y >= ArrayDim.y || idx.z >= ArrayDim.z)
        return;

    extern __shared__ float buffer[];
    int bufferIdx = threadIdx.x + blockDim.x * ( threadIdx.y + blockDim.y * threadIdx.z);
    float* local_value = buffer + bufferIdx;
    *local_value = 0.0f;
    size_t pitch = PitchedArray.pitch / sizeof(float);
    float* ptr = (float*)PitchedArray.ptr;

    float inverse_ssfactor = 1. / ssfactor;
    for (int k=0; k<ssfactor; k++) {
        float offset_k = (k + 0.5f) * inverse_ssfactor;
        for (int j=0; j<ssfactor; j++) {
            float offset_j = (j + 0.5f) * inverse_ssfactor;
            for (int i=0; i<ssfactor; i++) {
                float offset_i = (i + 0.5f) * inverse_ssfactor;

                float3 coords{idx.x + offset_i, idx.y + offset_j, idx.z + offset_k};
                coords *= voxel_size;
                float3 coords_minus_source_PVCS = coords - beam_d.source;
                float3 coords_minus_source_BEV = d_rotateBeamAtOriginRHS(
                    coords_minus_source_PVCS, beam_d.angles.x, beam_d.angles.y, beam_d.angles.z);

                float2 voxel_size_at_this_point = beam_d.beamlet_size * (coords_minus_source_BEV.y / beam_d.sad);
                float3 coords_normalized {
                    coords_minus_source_BEV.x / voxel_size_at_this_point.x + 0.5f * beam_d.fmap_size.x,
                    (coords_minus_source_BEV.y - beam_d.lim_min) / beam_d.long_spacing,
                    coords_minus_source_BEV.z / voxel_size_at_this_point.y + 0.5f * beam_d.fmap_size.y
                };
                *local_value += tex3D<float>(BEVTex,
                    coords_normalized.x, coords_normalized.z, coords_normalized.y);
            }
        }
    }

    size_t global_coords = idx.x + pitch * (idx.y + ArrayDim.y * idx.z);
    ptr[global_coords] = (*local_value) / (ssfactor * ssfactor);
}