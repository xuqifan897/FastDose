#include <string>
#include <fstream>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "IMRTgeom.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTArgs.h"
#include "helper_math.cuh"

namespace fd = fastdose;


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


bool IMRT::BEV2PVCSInterp(
    float** d_dense, std::vector<fd::BEAM_d>& beamlets,
    const fd::d_BEAM_d* d_beamlets,
    const fd::DENSITY_d& density_d, int supersampling,
    float extent, cudaStream_t stream
) {
    // firstly, construct the texture memory of the BEV pillars
    // to get the maximum longitudinal dimension
    int maximum_dim_long = 0;
    for (int i=0; i<beamlets.size(); i++) {
        maximum_dim_long = max(maximum_dim_long, beamlets[i].long_dim);
    }
    int2 fmap_size = make_int2(beamlets[0].fmap_size);
    size_t BEVPitch = beamlets[0].DoseBEV_pitch / sizeof(float);

    int nBeamlets = beamlets.size();
    int2 packDim;
    packDim.x = (int)std::ceil(sqrt((float)nBeamlets));
    packDim.y = (int)std::ceil((float)nBeamlets / packDim.x);

    int3 packArrayDim;
    packArrayDim.x = packDim.x * fmap_size.x;
    packArrayDim.y = packDim.y * fmap_size.y;
    packArrayDim.z = maximum_dim_long;

    #if false
        std::cout << "Maximum beamlet longitudinal dimension: " << maximum_dim_long
            << std::endl << "Number of beamlets: " << beamlets.size() << std::endl
            << "Packed beamlets dimension: " << packDim << std::endl
            << "Packed array dimension: " << packArrayDim
            << std::endl << std::endl;
    #endif

    // prepare the texture
    float* d_BEVLinear = nullptr;
    size_t BEVLinearSize = packArrayDim.x * packArrayDim.y * packArrayDim.z;
    checkCudaErrors(cudaMalloc(
        (void**)&d_BEVLinear,
        BEVLinearSize*sizeof(float)));
    checkCudaErrors(cudaMemset(d_BEVLinear, 0, BEVLinearSize*sizeof(float)));

    std::vector<float*> h_BEVSourceArray(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++) {
        h_BEVSourceArray[i] = beamlets[i].DoseBEV;
    }
    float** d_BEVSourceArray = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_BEVSourceArray, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_BEVSourceArray, h_BEVSourceArray.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    // allocate an array containing the dim_long of individual beamlets
    std::vector<int> beamLongArray(nBeamlets, 0);
    for (int i=0; i<nBeamlets; i++) {
        beamLongArray[i] = beamlets[i].long_dim;
    }
    int* d_beamLongArray = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beamLongArray, nBeamlets*sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_beamLongArray, beamLongArray.data(),
        nBeamlets*sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize{8, 8, 8};
    dim3 gridSize{
        (packArrayDim.x + blockSize.x - 1) / blockSize.x,
        (packArrayDim.y + blockSize.y - 1) / blockSize.y,
        (packArrayDim.z + blockSize.z - 1) / blockSize.z
    };

    d_InterpArrayPrep<<<gridSize, blockSize, 0, stream>>>(
        d_BEVLinear,
        nBeamlets,
        packDim,
        fmap_size,
        maximum_dim_long,
        d_BEVSourceArray,
        BEVPitch,
        d_beamLongArray
    );
    checkCudaErrors(cudaFree(d_beamLongArray));
    checkCudaErrors(cudaFree(d_BEVSourceArray));

    cudaArray* DoseBEV_Arr;
    cudaTextureObject_t DoseBEV_Tex;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize = make_cudaExtent(packArrayDim.x, packArrayDim.y, packArrayDim.z);
    cudaMalloc3DArray(&DoseBEV_Arr, &channelDesc, volumeSize);
    // copy to cudaArray
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(
        (void*)d_BEVLinear,
        volumeSize.width*sizeof(float),
        volumeSize.width,
        volumeSize.height);
    copyParams.dstArray = DoseBEV_Arr;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = DoseBEV_Arr;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    checkCudaErrors(cudaCreateTextureObject(&DoseBEV_Tex, &texRes, &texDescr, nullptr));
    checkCudaErrors(cudaFree(d_BEVLinear));


    // prepare for interpolation. Here we utilize a two-stage algorithm.
    // In the first stage, we calculate the "super voxels" that contain
    // non-trivial voxels. Then we calculate the interpolated valus for
    // these voxels
    const uint3& densityDim = density_d.VolumeDim;
    dim3 samplingBlockSize{8, 8, 8};
    dim3 samplingGridSize {
        (densityDim.x - 1 + samplingBlockSize.x) / samplingBlockSize.x,
        (densityDim.y - 1 + samplingBlockSize.y) / samplingBlockSize.y,
        (densityDim.z - 1 + samplingBlockSize.z) / samplingBlockSize.z };

    dim3 preSamplingBlockSize{4, 4, 4};
    dim3 preSamplingGridSize {
        (samplingGridSize.x - 1 + preSamplingBlockSize.x) / preSamplingBlockSize.x,
        (samplingGridSize.y - 1 + preSamplingBlockSize.y) / preSamplingBlockSize.y,
        (samplingGridSize.z - 1 + preSamplingBlockSize.z) / preSamplingBlockSize.z };
    size_t preSamplingArraySize = samplingGridSize.x * samplingGridSize.y
        * samplingGridSize.z * nBeamlets;
    bool* d_preSamplingArray = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_preSamplingArray, preSamplingArraySize*sizeof(bool)));
    checkCudaErrors(cudaMemset(d_preSamplingArray, 0, preSamplingArraySize*sizeof(bool)));

    d_superVoxelInterp<<<preSamplingGridSize, preSamplingBlockSize, 0, stream>>> (
        d_preSamplingArray,
        samplingGridSize,
        samplingBlockSize,
        d_beamlets,
        nBeamlets,
        density_d.VoxelSize,
        extent
    );

    // Assume d_dense is already allocated
    d_voxelInterp<<<samplingGridSize, samplingBlockSize, 0, stream>>> (
        *d_dense,
        d_preSamplingArray,
        densityDim,
        density_d.VoxelSize,
        samplingGridSize,
        samplingBlockSize,

        d_beamlets,
        nBeamlets,
        DoseBEV_Tex,
        packDim,
        fmap_size,
        5,
        extent
    );

    checkCudaErrors(cudaFree(d_preSamplingArray));
    checkCudaErrors(cudaDestroyTextureObject(DoseBEV_Tex));
    checkCudaErrors(cudaFreeArray(DoseBEV_Arr));

    return 0;
}


__global__ void
IMRT::d_voxelInterp(
    float* d_samplingArray,
    bool* d_preSamplingArray,
    dim3 densityDim,
    float3 densityVoxelSize,
    dim3 samplingGridSize,
    dim3 superVoxelDim,

    const fd::d_BEAM_d* beamlets,
    int nBeamlets,
    cudaTextureObject_t DoseBEV_Tex,
    int2 packDim,
    int2 fmap_size,
    int ssfactor,
    float extent
) {
    dim3 voxelIdx = threadIdx + blockDim * blockIdx;
    if (voxelIdx.x >= densityDim.x || voxelIdx.y >= densityDim.y || voxelIdx.z >= densityDim.z)
        return;
    
    size_t voxel_idx = voxelIdx.x + densityDim.x * (voxelIdx.y + densityDim.y * voxelIdx.z);
    size_t samplingPitch = densityDim.x * densityDim.y * densityDim.z;

    dim3 superVoxelIdx = blockIdx;
    size_t super_voxel_idx = superVoxelIdx.x + samplingGridSize.x *
        (superVoxelIdx.y + samplingGridSize.y * superVoxelIdx.z);
    size_t preSamplingPitch = samplingGridSize.x * samplingGridSize.y * samplingGridSize.z;

    float inverse_ssfactor = 1.0f / ssfactor;
    float inverse_ssfactor_cube = inverse_ssfactor * inverse_ssfactor * inverse_ssfactor;
    float halfVoxelDiag = length(densityVoxelSize) * 0.5f;

    for (int i=0; i<nBeamlets; i++) {
        const fd::d_BEAM_d beamlet = beamlets[i];
        bool preSamplingFlag = d_preSamplingArray[super_voxel_idx];
        super_voxel_idx += preSamplingPitch;
        if (preSamplingFlag) {
            // to calculate its distance from the central axis. To exclude in advance
            float3 voxel_coords {voxelIdx.x + 0.5f, voxelIdx.y + 0.5f, voxelIdx.z + 0.5f};
            voxel_coords *= densityVoxelSize;
            float3 voxel_coords_minus_source_PVCS = voxel_coords - beamlet.source;
            float3 voxel_coords_minus_source_BEV = d_rotateBeamAtOriginRHS(
                voxel_coords_minus_source_PVCS, beamlet.angles.x,
                beamlet.angles.y, beamlet.angles.z );
            float distance = sqrt(voxel_coords_minus_source_BEV.x
                * voxel_coords_minus_source_BEV.x
                + voxel_coords_minus_source_BEV.z
                * voxel_coords_minus_source_BEV.z);
            if (distance > halfVoxelDiag + extent)
                continue;

            int2 packIdx {i % packDim.x, i / packDim.x};
            int2 packOffset = packIdx * fmap_size;
            
            float local_value = 0.0f;
            for (int kk=0; kk<ssfactor; kk++) {
                float offset_k = (kk + 0.5f) * inverse_ssfactor;
                for (int jj=0; jj<ssfactor; jj++) {
                    float offset_j = (jj + 0.5f) * inverse_ssfactor;
                    for (int ii=0; ii<ssfactor; ii++) {
                        float offset_i = (ii + 0.5f) * inverse_ssfactor;

                        float3 coords {voxelIdx.x + offset_i,
                            voxelIdx.y + offset_j,
                            voxelIdx.z + offset_k};
                        coords *= densityVoxelSize;
                        voxel_coords_minus_source_PVCS = coords - beamlet.source;
                        voxel_coords_minus_source_BEV = d_rotateBeamAtOriginRHS (
                            voxel_coords_minus_source_PVCS,
                            beamlet.angles.x, beamlet.angles.y, beamlet.angles.z);
                        
                        float2 voxel_size_at_this_point = beamlet.beamlet_size
                            * (voxel_coords_minus_source_BEV.y / beamlet.sad);
                        float3 coords_normalized {
                            voxel_coords_minus_source_BEV.x / voxel_size_at_this_point.x + 0.5f * beamlet.fmap_size.x,
                            (voxel_coords_minus_source_BEV.y - beamlet.lim_min) / beamlet.long_spacing,
                            voxel_coords_minus_source_BEV.z / voxel_size_at_this_point.y + 0.5f * beamlet.fmap_size.y
                        };

                        if (coords_normalized.x < 0
                            || coords_normalized.x > beamlet.fmap_size.x ||
                            coords_normalized.z < 0 ||
                            coords_normalized.z > beamlet.fmap_size.y )
                                continue;
                        
                        float3 texIdx {
                            coords_normalized.x + packOffset.x,
                            coords_normalized.z + packOffset.y,
                            coords_normalized.y };
                        local_value += tex3D<float>(DoseBEV_Tex, texIdx.x, texIdx.y, texIdx.z);
                    }
                }
            }
            size_t voxel_idx_local = voxel_idx + i * samplingPitch;
            d_samplingArray[voxel_idx_local] = local_value * inverse_ssfactor_cube;
        }
    }
}


__global__ void
IMRT::d_superVoxelInterp(
    bool* d_preSamplingArray,
    dim3 samplingGridSize,
    dim3 superVoxelDim,
    const fd::d_BEAM_d* beamlets,
    int nBeamlets,
    float3 densityVoxelSize,
    float extent
) {
    // firstly, calculate the coordinates of the center of the super voxel
    int3 superVoxelIdx = make_int3(threadIdx + blockDim * blockIdx);
    float3 superVoxelSize {
        superVoxelDim.x * densityVoxelSize.x,
        superVoxelDim.y * densityVoxelSize.y,
        superVoxelDim.z * densityVoxelSize.z };
    float halfSuperVoxelDiag = length(superVoxelSize) * 0.5f;

    size_t samplingGridElements = samplingGridSize.x
        * samplingGridSize.y * samplingGridSize.z;
    size_t localOffset = superVoxelIdx.x + samplingGridSize.x * 
        (superVoxelIdx.y + samplingGridSize.y * superVoxelIdx.z);

    float3 superVoxelCoordsPVCS {
        superVoxelSize.x * (superVoxelIdx.x + 0.5f),
        superVoxelSize.y * (superVoxelIdx.y + 0.5f),
        superVoxelSize.z * (superVoxelIdx.z + 0.5f) };

    for (int i=0; i<nBeamlets; i++) {
        const fd::d_BEAM_d beamlet = beamlets[i];
        const float3& source = beamlet.source;
        const float3& angles = beamlet.angles;

        float3 superVoxelMinusSourcePVCS = superVoxelCoordsPVCS - source;
        float3 superVoxelMinusSourceBEV = d_rotateBeamAtOriginRHS(
            superVoxelMinusSourcePVCS, angles.x, angles.y, angles.z);
        float dist_from_central_axis = sqrt(
            superVoxelMinusSourceBEV.x * superVoxelMinusSourceBEV.x + 
            superVoxelMinusSourceBEV.z * superVoxelMinusSourceBEV.z );
        if (dist_from_central_axis < extent + halfSuperVoxelDiag) {
            size_t globalOffset = i * samplingGridElements + localOffset;
            d_preSamplingArray[globalOffset] = true;
        }

    }
}


__global__ void
IMRT::d_InterpArrayPrep (
    float* d_BEVLinear,
    int nBeamlets,
    int2 packDim,
    int2 fmap_size,
    int maximum_dim_long,
    float** d_BEVSourceArray,
    size_t BEVPitch,  // in float, not byte
    int* d_beamletLongArray
) {
    int3 voxelIdx = make_int3(threadIdx + blockDim * blockIdx);
    int2 packIdx {
        voxelIdx.x / fmap_size.x,
        voxelIdx.y / fmap_size.y};
    int pack_idx = packIdx.x + packDim.x * packIdx.y;
    if (pack_idx >= nBeamlets)
        return;

    int current_beamlet_long = d_beamletLongArray[pack_idx];
    if (voxelIdx.z >= current_beamlet_long)
        return;
    
    int3 withinBeamletIdx {
        voxelIdx.x % fmap_size.x,
        voxelIdx.y % fmap_size.y,
        voxelIdx.z};
    
    size_t withinBeamlet_idx = withinBeamletIdx.x +
        withinBeamletIdx.y * fmap_size.x +
        withinBeamletIdx.z * BEVPitch;
    
    size_t voxel_idx = voxelIdx.x + packDim.x * fmap_size.x *
        (voxelIdx.y + packDim.y * fmap_size.y * voxelIdx.z);
    d_BEVLinear[voxel_idx] = d_BEVSourceArray[pack_idx][withinBeamlet_idx];
}