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
    os << "long_spacing: " << obj.long_spacing << std::endl;
    os << "minimum range: " << obj.lim_min << std::endl;
    os << "maximum range: " << obj.lim_max << std::endl;
    os << "longitudinal dimension: " << obj.long_dim << std::endl;
    os << "source: " << obj.source << std::endl;
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
    beam_d.long_spacing = beam_h.long_spacing;
    beam_d.lim_min = beam_h.lim_min;
    beam_d.lim_max = beam_h.lim_max;
    beam_d.long_dim = beam_h.long_dim;
    beam_d.source = beam_h.source;

    if (beam_d.fluence) {
        checkCudaErrors(cudaFree(beam_d.fluence));
    }
    size_t volume = beam_d.fmap_size.x * beam_d.fmap_size.y;
    checkCudaErrors(cudaMalloc((void**)(&beam_d.fluence),
        volume * sizeof(float)));
    checkCudaErrors(cudaMemcpy(beam_d.fluence, beam_h.fluence.data(), 
        volume*sizeof(float), cudaMemcpyHostToDevice));

    int width = beam_d.fmap_size.x;
    int height = beam_d.fmap_size.y;
    int depth = beam_d.long_dim;
    cudaExtent extent = make_cudaExtent(width*sizeof(float), height, depth);
    checkCudaErrors(cudaMalloc3D(&(beam_d.TermaBEVPitch), extent));
    checkCudaErrors(cudaMemset3D(beam_d.TermaBEVPitch, 0., extent));

    checkCudaErrors(cudaMalloc3D(&(beam_d.DensityBEVPitch), extent));
    checkCudaErrors(cudaMemset3D(beam_d.DensityBEVPitch, 0., extent));
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

#define debug_calc_range false

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

    float3 source_{0, -this->sad, 0};
    source_ = inverseRotateBeamAtOriginRHS(
        source_, this->angles.x, this->angles.y, this->angles.z);
    source_ = source_ + this->isocenter;
    this->source = source_;

    float3 bbox_begin = make_float3(density_h.BBoxStart) * density_h.VoxelSize;
    float3 bbox_end = make_float3(density_h.BBoxStart + density_h.BBoxDim) * density_h.VoxelSize;

#if debug_calc_range
    std::cout << "bbox_begin: " << bbox_begin << std::endl;
    std::cout << "bbox_end: " << bbox_end << std::endl;
#endif
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
            double alpha1 = ( FCOMP(bbox_begin, j) - FCOMP(source_, j) ) / denom;
            double alpha2 = ( FCOMP(bbox_end, j) - FCOMP(source_, j) ) / denom;
            intersections[2*j] = source_ + alpha1 * boundary;
            intersections[2*j+1] = source_ + alpha2 * boundary;
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

#if debug_calc_range
                    std::cout << "Boudary: " << boundary << ", intersection: " << intersections[idx] << std::endl;
#endif
                    float3 _inter = intersections[idx];
                    // projection length
                    float _length = length(_inter - source_) * this->sad / length(boundary);
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
    this->lim_min = _min_dist_;
    this->lim_max = _max_dist_;
    this->long_dim = uint(std::ceil((this->lim_max - this->lim_min) / this->long_spacing));

#if debug_calc_range
    // for debug purposes
    std::cout << *this << std::endl;
#endif

    return 0;
}

void fastdose::test_TermaBEVPitch(BEAM_d& beam_d) {
    // assume beam_d.TermaBEVPitch is allocated
    int width = beam_d.fmap_size.x;
    int height = beam_d.fmap_size.y;
    int depth = beam_d.long_dim;
    size_t volume = height * width * depth;
    std::vector<float> h_data(volume);
    for (int i=0; i<volume; i++) {
        h_data[i] = rand01();
    }

    // copy data from host to device
    cudaExtent extent = make_cudaExtent(width*sizeof(float), height, depth);
    cudaMemcpy3DParms copyParms = {0};
    copyParms.srcPtr = make_cudaPitchedPtr(h_data.data(), width*sizeof(float), width, height);
    copyParms.dstPtr = beam_d.TermaBEVPitch;
    copyParms.extent = extent;
    copyParms.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParms));

    std::cout << "(width, height, depth) = (" << width << ", " << height <<
        ", " << depth << ")" << std::endl;
    std::cout << "pitch = " << beam_d.TermaBEVPitch.pitch << "[bytes]" << std::endl;

    // copy data from device to host
    size_t pitched_volume = depth * height *  beam_d.TermaBEVPitch.pitch / sizeof(float);
    std::vector<float> sample(pitched_volume);
    checkCudaErrors(cudaMemcpy(sample.data(), beam_d.TermaBEVPitch.ptr, 
        pitched_volume*sizeof(float), cudaMemcpyDeviceToHost));

    double absolute_diff = 0.;
    size_t pitch = beam_d.TermaBEVPitch.pitch / sizeof(float);
    size_t slicePitch = height * pitch;
    size_t slice = height * width;
    for (int i=0; i<depth; i++) {
        size_t idx_i = i * slice;
        size_t idx_i_pitched = i * slicePitch;
        for (int j=0; j<height; j++) {
            size_t idx_j = idx_i + j * width;
            size_t idx_j_pitched = idx_i_pitched + j * pitch;
            for (int k=0; k<width; k++) {
                size_t idx_k = idx_j + k;
                size_t idx_k_pitched = idx_j_pitched + k;
                absolute_diff += abs(sample[idx_k_pitched] - h_data[idx_k]);
            }
        }
    }

    std::cout << "Absolute difference: " << absolute_diff << std::endl;
}