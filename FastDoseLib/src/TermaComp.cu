#include "fastdose.cuh"
#include "helper_math.cuh"
#include "device_launch_parameters.h"
#include <boost/filesystem.hpp>
#include <fstream>
namespace fs = boost::filesystem;

namespace fastdose{
    __constant__ float d_energy[MAX_KERNEL_NUM];
    __constant__ float d_fluence[MAX_KERNEL_NUM];
    __constant__ float d_mu[MAX_KERNEL_NUM];
    __constant__ float d_mu_en[MAX_KERNEL_NUM];

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
}

namespace fd = fastdose;

#define DIM2 8
#define SUPER_SAMPLING 4 // divide a a step into substeps

bool fd::SPECTRUM_h::bind_spectrum() {
    if (this->nkernels > MAX_KERNEL_NUM) {
        std::cerr << "The number of kernels included is more than MAX_KERNEL_NUM (" 
            << MAX_KERNEL_NUM << ")" << std::endl;
        return 1;
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_energy, this->energy.data(), this->nkernels*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_fluence, this->fluence.data(), this->nkernels*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_mu, this->mu.data(), this->nkernels*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_mu_en, this->mu_en.data(), this->nkernels*sizeof(float)));
    return 0;
}

__global__ void
fd::d_test_spectrum(float* output, int width, int idx) {
    int ii = threadIdx.x;
    if (ii >= width)
        return;
    if (idx == 0) {
        output[ii] = d_energy[ii];
    } else if (idx == 1) {
        output[ii] = d_fluence[ii];
    } else if (idx == 2) {
        output[ii] = d_mu[ii];
    } else if (idx == 3) {
        output[ii] = d_mu_en[ii];
    }
}


void fd::BEV2PVCS(
    BEAM_d& beam_d,
    DENSITY_d& density_d,
    cudaPitchedPtr& PitchedOutput,
    cudaTextureObject_t BEVTex,
    cudaStream_t stream
) {
    d_BEAM_d beam_input(beam_d);
    uint width = density_d.VolumeDim.x;
    uint height = density_d.VolumeDim.y;
    uint depth = density_d.VolumeDim.z;
    dim3 blockSize{DIM2, DIM2, DIM2};
    dim3 gridSize{
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        (depth + blockSize.z - 1) / blockSize.z,
    };
    d_BEV2PVCS<<<gridSize, blockSize, 0, stream>>>(
        beam_input,
        PitchedOutput,
        BEVTex,
        density_d.VolumeDim,
        density_d.VoxelSize
    );
}


__global__ void
fd::d_BEV2PVCS(
    d_BEAM_d beam_d,
    cudaPitchedPtr PitchedArray,
    cudaTextureObject_t BEVTex,
    uint3 ArrayDim,
    float3 voxel_size
) {
    uint3 idx = threadIdx + blockDim * blockIdx;
    if (idx.x >= ArrayDim.x || idx.y >= ArrayDim.y || idx.z >=ArrayDim.z)
        return;
    float3 coords = (make_float3(idx) + 0.5f) * voxel_size;
    float3 coords_minus_source_PVCS = coords - beam_d.source;
    float3 coords_minus_source_BEV = d_rotateBeamAtOriginRHS(coords_minus_source_PVCS, beam_d.angles.x, beam_d.angles.y, beam_d.angles.z);

    float2 voxel_size_at_this_point = beam_d.beamlet_size * (coords_minus_source_BEV.y / beam_d.sad);
    float3 coords_normalized {
        coords_minus_source_BEV.x / voxel_size_at_this_point.x,
        (coords_minus_source_BEV.y - beam_d.lim_min) / beam_d.long_spacing,
        coords_minus_source_BEV.z / voxel_size_at_this_point.y
    };
    size_t pitch = PitchedArray.pitch / sizeof(float);
    float* ptr = (float*)PitchedArray.ptr;
    size_t global_coords = idx.x + pitch * (idx.y + ArrayDim.y * idx.z);
    ptr[global_coords] = tex3D<float>(BEVTex, 
        coords_normalized.x, coords_normalized.z, coords_normalized.y);
}


bool fd::TermaComputeCollective(
    size_t fmap_npixels,
    size_t n_beams,
    d_BEAM_d* beams,
    float** fluence_array,
    float** TermaBEV_array,
    float** DensityBEV_array,
    DENSITY_d& density_d,
    SPECTRUM_h& spectrum_h,
    cudaStream_t stream
) {
    dim3 blockSize(((fmap_npixels + WARPSIZE - 1) / WARPSIZE) * WARPSIZE, 1, 1);
    dim3 gridSize(n_beams, 1, 1);
    d_TermaComputeCollective<<<gridSize, blockSize, 0, stream>>>(
        beams,
        fluence_array,
        TermaBEV_array,
        DensityBEV_array,
        density_d.densityTex,
        density_d.VoxelSize,
        spectrum_h.nkernels
    );
    return 0;
}


__global__ void
fd::d_TermaComputeCollective(
    d_BEAM_d* beams,
    float** fluence_maps,
    float** TermaBEV_array,
    float** DenseBEV_array,
    cudaTextureObject_t densityTex,
    float3 voxel_size,
    int nkern
) {
    int beam_idx = blockIdx.x;
    d_BEAM_d beam = beams[beam_idx];
    float* fluence_map = fluence_maps[beam_idx];
    float* TermaBEV = TermaBEV_array[beam_idx];
    float* DenseBEV = DenseBEV_array[beam_idx];

    int pixel_idx = threadIdx.x;
    int idx_x = pixel_idx % beam.fmap_size.x;
    int idx_y = pixel_idx / beam.fmap_size.x;
    if (idx_y >= beam.fmap_size.y)
        return;

    float fluence_value = fluence_map[pixel_idx];
    float3 pixel_center_minus_source_BEV {
        (idx_x + 0.5f - beam.fmap_size.x * 0.5f) * beam.beamlet_size.x,
        beam.sad,
        (idx_y + 0.5f - beam.fmap_size.y * 0.5f) * beam.beamlet_size.y
    };
    float3 pixel_center_minus_source_PVCS = d_inverseRotateBeamAtOriginRHS(
        pixel_center_minus_source_BEV, beam.angles.x, beam.angles.y, beam.angles.z);
    float3 step_size_PVCS = pixel_center_minus_source_PVCS *
        (beam.long_spacing / (beam.sad * SUPER_SAMPLING));
    float step_size_norm = length(step_size_PVCS);   // physical length

    // initialize to starting point
    float3 coords_PVCS = beam.source + pixel_center_minus_source_PVCS * (beam.lim_min / beam.sad);

    float3 step_size_PVCS_normalized = step_size_PVCS / voxel_size;
    float3 coords_PVCS_normalized = coords_PVCS / voxel_size;

    float radiological_path_length = 0.f;
    for (int i=0; i<beam.long_dim; i++) {
        float terma_avg = 0.;
        float density_avg = 0.;
        #pragma unroll
        for (int j=0; j<SUPER_SAMPLING; j++) {
            coords_PVCS_normalized += step_size_PVCS_normalized;
            float density = tex3D<float>(densityTex, coords_PVCS_normalized.x,
                coords_PVCS_normalized.y, coords_PVCS_normalized.z);
            radiological_path_length += density * step_size_norm;
            density_avg += density;

            float terma_local = 0.f;
            for (int e=0; e<nkern; e++) {
                float this_fluence = d_fluence[e] * fluence_value;
                float this_energy = d_energy[e];
                float this_mu = d_mu[e];
                terma_local += this_fluence * this_energy * this_mu *
                    __expf(- this_mu * radiological_path_length);
            }
            terma_avg += terma_local;
        }
        terma_avg /= SUPER_SAMPLING;
        density_avg /= SUPER_SAMPLING;
        size_t global_idx = pixel_idx + beam.TermaBEV_pitch * i / sizeof(float);
        TermaBEV[global_idx] = terma_avg;
        DenseBEV[global_idx] = density_avg;
    }
}


bool fd::test_TermaComputeCollective(
    std::vector<BEAM_d>& beams,
    DENSITY_d& density_d,
    SPECTRUM_h& spectrum_h,
    const std::string& outputFolder,
    cudaStream_t stream
) {
    // copy beams
    std::vector<d_BEAM_d> h_beams;
    h_beams.reserve(beams.size());
    for (int i=0; i<beams.size(); i++)
        h_beams.emplace_back(d_BEAM_d(beams[i]));
    d_BEAM_d* d_beams = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beams, beams.size()*sizeof(d_BEAM_d)));
    checkCudaErrors(cudaMemcpy(d_beams, h_beams.data(),
        beams.size()*sizeof(d_BEAM_d), cudaMemcpyHostToDevice));

    // copy fluence maps
    std::vector<float*> h_fluence_array;
    h_fluence_array.resize(beams.size());
    for (int i=0; i<beams.size(); i++)
        h_fluence_array[i] = beams[i].fluence;
    float** fluence_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)(&fluence_array), beams.size()*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(fluence_array, h_fluence_array.data(),
        beams.size()*sizeof(float*), cudaMemcpyHostToDevice));
    
    // allocate TermaBEV_array
    std::vector<float*> h_TermaBEV_array(beams.size(), nullptr);
    for (int i=0; i<beams.size(); i++)
        h_TermaBEV_array[i] = beams[i].TermaBEV;
    float** TermaBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)(&TermaBEV_array), beams.size()*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(TermaBEV_array, h_TermaBEV_array.data(),
        beams.size()*sizeof(float*), cudaMemcpyHostToDevice));
    
    // allocate DenseBEV_array
    std::vector<float*> h_DensityBEV_array(beams.size(), nullptr);
    for (int i=0; i<beams.size(); i++)
        h_DensityBEV_array[i] = beams[i].DensityBEV;
    float** DenseBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)(&DenseBEV_array), beams.size()*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(DenseBEV_array, h_DensityBEV_array.data(),
        beams.size()*sizeof(float*), cudaMemcpyHostToDevice));

    size_t fmap_npixels = beams[0].fmap_size.x * beams[0].fmap_size.y;

#if false
    // output the parameters of all beams
    for (int i=0; i<beams.size(); i++)
        std::cout << "beam " << i << ", fluence map dimensions: " << 
        beams[i].fmap_size << ", longitudinal dimension: " << beams[i].long_dim << std::endl;
#endif

    // for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    TermaComputeCollective(
        fmap_npixels,
        beams.size(),
        d_beams,
        fluence_array,
        TermaBEV_array,
        DenseBEV_array,
        density_d,
        spectrum_h,
        stream
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Terma time elapsed: " << milliseconds << " [ms]"<< std::endl;

    // retrieve result
    size_t nVoxels = beams[0].fmap_size.x * beams[0].fmap_size.y * beams[0].long_dim;
    size_t pitch_host = beams[0].fmap_size.x * beams[0].fmap_size.y;
    std::vector<float> TermaBEV_example(nVoxels, 0.);
    checkCudaErrors(cudaMemcpy2D(
        TermaBEV_example.data(), pitch_host*sizeof(float),
        beams[0].TermaBEV, beams[0].TermaBEV_pitch,
        pitch_host*sizeof(float), beams[0].long_dim, cudaMemcpyDeviceToHost));
    
    std::vector<float> DenseBEV_example(nVoxels, 0.);
    checkCudaErrors(cudaMemcpy2D(
        DenseBEV_example.data(), pitch_host*sizeof(float),
        beams[0].DensityBEV, beams[0].DensityBEV_pitch,
        pitch_host*sizeof(float), beams[0].long_dim, cudaMemcpyDeviceToHost));
    
    fs::path DensityFile = fs::path(outputFolder) / std::string("DensityBEV.bin");
    std::ofstream f(DensityFile.string());
    if (! f) {
        std::cerr << "Could not open file: " << DensityFile.string() << std::endl;
        return 1;
    }
    f.write((char*)(DenseBEV_example.data()), nVoxels*sizeof(float));
    f.close();

    fs::path TermaFile = fs::path(outputFolder) / std::string("TermaBEV.bin");
    f.open(TermaFile.string());
    if (! f.is_open()) {
        std::cerr << "Could not open file: " << TermaFile.string() << std::endl;
        return 1;
    }
    f.write((char*)(TermaBEV_example.data()), nVoxels*sizeof(float));
    f.close();
    
    // clean up
    checkCudaErrors(cudaFree(d_beams));
    checkCudaErrors(cudaFree(fluence_array));
    checkCudaErrors(cudaFree(TermaBEV_array));
    checkCudaErrors(cudaFree(DenseBEV_array));


    // transfer BEV to PVCS. Construct texture memory first
    cudaArray* TermaBEV_Arr;
    cudaTextureObject_t TermaBEV_Tex;
    cudaExtent volumeSize = make_cudaExtent(
        beams[0].fmap_size.x, beams[0].fmap_size.y, beams[0].long_dim);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&TermaBEV_Arr, &channelDesc, volumeSize));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = 
        make_cudaPitchedPtr(TermaBEV_example.data(),
            beams[0].fmap_size.x * sizeof(float),
            beams[0].fmap_size.x, beams[0].fmap_size.y);
    copyParams.dstArray = TermaBEV_Arr;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = TermaBEV_Arr;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&TermaBEV_Tex, &texRes, &texDescr, NULL));

    // Construct destination array
    cudaExtent TermaPVCS_Arr_extent = make_cudaExtent(
        density_d.VolumeDim.x * sizeof(float),
        density_d.VolumeDim.y, density_d.VolumeDim.z);
    cudaPitchedPtr TermaPVCS_Arr;
    checkCudaErrors(cudaMalloc3D(&TermaPVCS_Arr, TermaPVCS_Arr_extent));

    BEV2PVCS(
        beams[0],
        density_d,
        TermaPVCS_Arr,
        TermaBEV_Tex
    );

    // write result
    size_t pitchedVolume = TermaPVCS_Arr.pitch / sizeof(float) * 
        density_d.VolumeDim.y * density_d.VolumeDim.z;
    size_t volume = density_d.VolumeDim.x * density_d.VolumeDim.y * 
        density_d.VolumeDim.z;
    std::vector<float> TermaPVCS_pitched(pitchedVolume);
    std::vector<float> TermaPVCS(volume);
    checkCudaErrors(cudaMemcpy(TermaPVCS_pitched.data(), TermaPVCS_Arr.ptr,
        pitchedVolume*sizeof(float), cudaMemcpyDeviceToHost));
    pitched2contiguous(TermaPVCS, TermaPVCS_pitched,
        density_d.VolumeDim.x, density_d.VolumeDim.y, density_d.VolumeDim.z,
        TermaPVCS_Arr.pitch / sizeof(float)
    );
    fs::path TermaPVCSFile = fs::path(outputFolder) / std::string("TermaPVCS.bin");
    f.open(TermaPVCSFile.string());
    if (! f) {
        std::cerr << "Could not open file " << TermaPVCSFile.string() << std::endl;
        return 1;
    }
    f.write((char*)(TermaPVCS.data()), volume*sizeof(float));
    f.close();

    return 0;
}