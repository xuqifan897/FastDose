#ifndef __IMRTGEOM_CUH__
#define __IMRTGEOM_CUH__
#include "IMRTBeamBundle.cuh"
#include "fastdose.cuh"

namespace IMRT {
    bool BEV2PVCSInterp(
        float** d_dense, std::vector<fastdose::BEAM_d>& beamlets,
        const fastdose::d_BEAM_d* d_beamlets,
        const fastdose::DENSITY_d& density_d, int supersampling=5,
        float extent = 2.0f, cudaStream_t stream=0
    );

    bool BEV2PVCSInterp(
        float* d_dense,
        size_t d_dense_size,  // in float, not byte
        const fastdose::d_BEAM_d* d_beamlets,
        int nBeamlets,
        const fastdose::DENSITY_d& density_d,
        float** d_DoseArray,
        size_t pitch,  // in float, not byte
        bool* preSamplingArray,
        size_t preSamplingArraySize,
        float* packArray,
        int2 packDim,
        int2 fmap_size,
        int3 packArrayDim,
        cudaArray** DoseBEV_Arr,
        int* d_beamletLongArray,
        float extent,
        cudaStream_t stream = 0,
        cudaStream_t memsetStream = 0
    );

    __global__ void
    d_InterpArrayPrep (
        float* d_BEVLinear,
        int nBeamlets,
        int2 packDim,
        int2 fmap_size,
        int maximum_dim_long,
        float** d_BEVSourceArray,
        size_t BEVPitch,  // in float, not byte
        int* d_beamletLongArray
    );

    __global__ void
    d_superVoxelInterp(
        bool* d_preSamplingArray,
        dim3 samplingGridSize,
        dim3 superVoxelDim,
        const fastdose::d_BEAM_d* beamlets,
        int nBeamlets,
        float3 densityVoxelSize,
        float extent
    );

    __global__ void
    d_voxelInterp(
        float* d_samplingArray,
        bool* d_preSamplingArray,
        dim3 densityDim,
        float3 densityVoxelSize,
        dim3 samplingGridSize,
        dim3 superVoxelDim,

        const fastdose::d_BEAM_d* beamlets,
        int nBeamlets,
        cudaTextureObject_t DoseBEV_Tex,
        int2 packDim,
        int2 fmap_size,
        int ssfactor,
        float extent
    );
}

#endif