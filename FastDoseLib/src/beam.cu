#include "fastdose.cuh"
#include "utils.cuh"

#include <limits>
#include <iomanip>
#include <random>
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "math_constants.h"
#include "helper_math.cuh"

#define N_BOUNDARIES 5
#define FCOMP(x, j) *(((float*)&x)+j) // return component of float vector by index
#define eps 1e-4

namespace fd = fastdose;

std::ostream& fd::operator<<(std::ostream& os, const fd::BEAM_h& obj) {
    os << "isocenter: " << obj.isocenter << std::endl;
    os << "beamlet_size: " << obj.beamlet_size << std::endl;
    os << "fmap_size: " << obj.fmap_size << std::endl;
    os << "sad: " << obj.sad << std::endl;
    os << "angles: " << obj.angles << std::endl;
    os << "fluence: " << std::endl;
    for (int i=0; i<obj.fmap_size.y; i++) {
        for (int j=0; j<obj.fmap_size.x; j++) {
            os << std::left << std::setw(10) << std::scientific << std::setprecision(2) << obj(j, i);
        }
        os << std::endl;
    }
    return os;
}

void fd::beam_h2d(BEAM_h& beam_h, BEAM_d& beam_d) {
    beam_d.isocenter = beam_h.isocenter;
    beam_d.beamlet_size = beam_h.beamlet_size;
    beam_d.fmap_size = beam_h.fmap_size;
    beam_d.sad = beam_h.sad;
    beam_d.angles = beam_h.angles;
    if (beam_d.fluence) {
        checkCudaErrors(cudaFree(beam_d.fluence));
    }
    size_t volume = beam_d.fmap_size.x * beam_d.fmap_size.y;
    checkCudaErrors(cudaMalloc((void**)(&beam_d.fluence),
        volume * sizeof(float)));
    checkCudaErrors(cudaMemcpy(beam_d.fluence, beam_h.fluence.data(), 
        volume*sizeof(float), cudaMemcpyHostToDevice));
}

void fd::beam_d2h(BEAM_d& beam_d, BEAM_h& beam_h) {
    beam_h.isocenter = beam_d.isocenter;
    beam_h.beamlet_size = beam_d.beamlet_size;
    beam_h.fmap_size = beam_d.fmap_size;
    beam_h.sad = beam_d.sad;
    beam_h.angles = beam_d.angles;
    size_t volume = beam_h.fmap_size.x * beam_h.fmap_size.y;
    beam_h.fluence.resize(volume);
    checkCudaErrors(cudaMemcpy(beam_h.fluence.data(), beam_d.fluence,
        volume*sizeof(float), cudaMemcpyDeviceToHost));
}

void fd::test_beam_io() {
    std::cout << "beam io test:" << std::endl;
    uint2 fmap_size{10, 10};
    size_t volume = fmap_size.x * fmap_size.y;

    BEAM_h beam_h;
    beam_h.isocenter = float3{rand01(), rand01(), rand01()};
    beam_h.beamlet_size = float2{rand01(), rand01()};
    beam_h.fmap_size = fmap_size;
    beam_h.sad = rand01();
    beam_h.angles = float3{rand01(), rand01(), rand01()};
    beam_h.fluence.resize(volume);
    for (int i=0; i<volume; i++) {
        beam_h.fluence[i] = rand01();
    }
    std::cout << "original beam_h:" << std::endl;
    std::cout << beam_h << std::endl;

    BEAM_d beam_d;
    beam_h2d(beam_h, beam_d);

    BEAM_h beam_h_new;
    beam_d2h(beam_d, beam_h_new);
    std::cout << "We fisrtly transfer beam_h to beam_d, and secondly from beam_d to beam_h_new:" << std::endl;
    std::cout << beam_h_new << std::endl;
}

bool fd::BEAM_h::calc_range(const DENSITY_h& density_h) {
    float2 half_fluence_size{
        this->fmap_size.x * this->beamlet_size.x / 2,
        this->fmap_size.y * this->beamlet_size.y / 2};
    
    // Set the boundaries in the BEV coordinate system
    std::vector<float3> boundaries(N_BOUNDARIES);
    boundaries[0] = float3{half_fluence_size.x, this->sad, half_fluence_size.y};
    boundaries[1] = float3{half_fluence_size.x, this->sad, -half_fluence_size.y};
    boundaries[2] = float3{-half_fluence_size.x, this->sad, half_fluence_size.x};
    boundaries[3] = float3{-half_fluence_size.x, this->sad, -half_fluence_size.y};
    boundaries[4] = float3{0, this->sad, 0};

    // Calculate the boundaries from the BEV coordinates to PVCS coordinates
    for (int i=0; i<N_BOUNDARIES; i++) {
        boundaries[i] = inverseRotateBeamAtOriginRHS(
            boundaries[i], this->angles.x, this->angles.y, this->angles.z);
    }

    float3 source{0, -this->sad, 0};
    float3 source = inverseRotateBeamAtOriginRHS(
        source, this->angles.x, this->angles.y, this->angles.z);
    source = source + this->isocenter;

    float3 bbox_begin = make_float3(density_h.BBoxStart) * density_h.VoxelSize;
    float3 bbox_end = make_float3(density_h.BBoxStart + density_h.BBoxDim) * density_h.VoxelSize;

    // calculate the ranges
    float _min_dist_ = std::numeric_limits<float>::max();
    float _max_dist_ = std::numeric_limits<float>::min();
    bool intersect = false;
    for (int i=0; i<N_BOUNDARIES; i++) {
        const auto& boundary = boundaries[i];

        // evaluate intersection with each of the 6 calc_bbox faces
        float3 intersections[6] = {};
        for (int j=0; j<3; j++) {
            double denom = FCOMP(boundary, j);
            denom = std::signbit(denom) ? denom - eps : denom + eps;  // for numeric stability
            double alpha1 = ( FCOMP(bbox_begin, j) - FCOMP(source, j) ) / denom;
            double alpha2 = ( FCOMP(bbox_end, j) - FCOMP(source, j) ) / denom;
            intersections[2*j] = source + alpha1 * boundary;
            intersections[2*j+1] = source + alpha2 * boundary;
        }

        // check for valid intersectin with clac_bbox faces
        for (int j=0; j<3; j++) {
            for (int k=0; k<2; k++) {
                int idx = 2 * j + k;
                // given intersection with one dimension, do other two dims occur within bounds of box?
                if (FCOMP(intersections[idx],(j+1)%3) >= FCOMP(bbox_begin,(j+1)%3) 
                    && FCOMP(intersections[idx],(j+1)%3) <= FCOMP(bbox_end,(j+1)%3)
                    && FCOMP(intersections[idx],(j+2)%3) >= FCOMP(bbox_begin,(j+2)%3)
                    && FCOMP(intersections[idx],(j+2)%3) <= FCOMP(bbox_end,(j+2)%3)
                ) {
                    float3 _inter = intersections[idx];
                    // projection length
                    float _length = length(_inter - source) * this->sad / length(boundary);
                    _min_dist_ = min(_length, _min_dist_);
                    _max_dist_ = max(_length, _max_dist_);
                    intersect = true;
                }
            }
        }
    }

    if (! intersect) {
        std::cerr << "The beam doesn't intersect with the bounding box" << std::endl;
        return 1;
    }
    return 0;
}