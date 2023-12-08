#include "fastdose.cuh"
#include "helper_math.cuh"
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

#define DIM1 16
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

bool fd::TermaCompute(
    BEAM_d& beam_d, DENSITY_d& density_d, SPECTRUM_h& spectrum_h, cudaStream_t stream
) {
    dim3 blockSize(DIM1, DIM1);
    dim3 gridSize;
    gridSize.x = (beam_d.fmap_size.x + blockSize.x - 1) / blockSize.x;
    gridSize.y = (beam_d.fmap_size.y + blockSize.y - 1) / blockSize.y;
    d_BEAM_d beam_input(beam_d);

    // for safety
    if (beam_d.TermaBEVPitch.pitch != beam_d.DensityBEVPitch.pitch) {
        std::cerr << "The pitch of the density and the terma arrays are not equal." << std::endl;
        return 1;
    }

    d_TermaCompute<<<gridSize, blockSize, 0, stream>>>(
        beam_input,
        beam_d.fluence,
        beam_d.TermaBEVPitch,
        beam_d.DensityBEVPitch,
        density_d.densityTex,
        density_d.VoxelSize,
        spectrum_h.nkernels);
    return 0;
}

__global__ void
fd::d_TermaCompute(
        d_BEAM_d beam_d,
        float* fluence_map,
        cudaPitchedPtr TermaBEVPitch,
        cudaPitchedPtr DenseBEVPitch,
        cudaTextureObject_t densityTex,
        float3 voxel_size,
        int nkern
) {
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx_x >= beam_d.fmap_size.x || idx_y >= beam_d.fmap_size.y)
        return;

    int dest_pitch = TermaBEVPitch.pitch / sizeof(float);
    float* dest_ptr = (float*)TermaBEVPitch.ptr;
    float* dens_ptr = (float*)DenseBEVPitch.ptr;
    
    int fluence_idx = idx_x + idx_y * beam_d.fmap_size.x;
    float fluence_value = fluence_map[fluence_idx];

    float3 pixel_center_minus_source_BEV {
        (idx_x + 0.5f - beam_d.fmap_size.x * 0.5f) * beam_d.beamlet_size.x,
        beam_d.sad,
        (idx_y + 0.5f - beam_d.fmap_size.y * 0.5f) * beam_d.beamlet_size.y
    };
    float3 pixel_center_minus_source_PVCS = d_inverseRotateBeamAtOriginRHS(
        pixel_center_minus_source_BEV, beam_d.angles.x, beam_d.angles.y, beam_d.angles.z);

    float3 step_size_PVCS = pixel_center_minus_source_PVCS * (beam_d.long_spacing / (beam_d.sad * SUPER_SAMPLING));
    float step_size_norm = length(step_size_PVCS);  // physical length

    // initialize to starting point
    float3 coords_PVCS = beam_d.source + pixel_center_minus_source_PVCS * (beam_d.lim_min / beam_d.sad);

    float3 step_size_PVCS_normalized = step_size_PVCS / voxel_size;
    float3 coords_PVCS_normalized = coords_PVCS / voxel_size;

    double radiological_path_length = 0.;
    for (int i=0; i<beam_d.long_dim; i++) {
        float t_sum_avg = 0.;
        float density_avg = 0.;
        for (int j=0; j<SUPER_SAMPLING; j++) {
            coords_PVCS_normalized += step_size_PVCS_normalized;
            float density = tex3D<float>(densityTex, coords_PVCS_normalized.x, coords_PVCS_normalized.y, coords_PVCS_normalized.z);
            radiological_path_length += density * step_size_norm;
            density_avg += density;

            float tsum_local = 0.;
            for (int e=0; e<nkern; e++) {
                float this_fluence = d_fluence[e] * fluence_value;
                float this_energy = d_energy[e];
                // float this_mu_en = d_mu_en[e];
                float this_mu = d_mu[e];
                tsum_local += this_fluence * this_energy * this_mu * 
                    exp(- this_mu * radiological_path_length);
            }
            t_sum_avg += tsum_local;
        }
        t_sum_avg /= SUPER_SAMPLING;
        density_avg /= SUPER_SAMPLING;
        size_t global_idx = idx_x + dest_pitch * (idx_y + i * beam_d.fmap_size.y);
        dest_ptr[global_idx] = t_sum_avg;
        dens_ptr[global_idx] = density_avg;
    }
}


bool fd::test_TermaCompute(BEAM_d& beam_d, DENSITY_d& density_d, SPECTRUM_h& spectrum_h,
    const std::string& outputFolder) {
    TermaCompute(beam_d, density_d, spectrum_h);
    
    int width = beam_d.fmap_size.x;
    int height = beam_d.fmap_size.y;
    int depth = beam_d.long_dim;

    // copy data from device to host
    size_t pitch = beam_d.TermaBEVPitch.pitch / sizeof(float);
    size_t pitched_volume = depth * height *  pitch;
    std::vector<float> sample(pitched_volume);
    checkCudaErrors(cudaMemcpy(sample.data(), beam_d.TermaBEVPitch.ptr, 
        pitched_volume*sizeof(float), cudaMemcpyDeviceToHost));
    
    // copy data from pitched memory to contiguous memory
    size_t output_volume = depth * height * width;
    std::vector<float> output(output_volume, 0.);
    pitched2contiguous(output, sample, width, height, depth, pitch);

    fs::path outputFile = fs::path(outputFolder) / std::string("TermaBEV.bin");
    std::ofstream f(outputFile.string());
    if (! f) {
        std::cerr << "Could not open the file: " << outputFile.string() << std::endl;
        return 1;
    }
    f.write((char*)output.data(), output_volume*sizeof(float));
    f.close();


    // write terma to PVCS coords
    // firstly, create a texture for BEVTerma
    cudaArray* BEVTermaArray;
    cudaTextureObject_t BEVTermaTex;

    cudaExtent BEVTermaVolume = make_cudaExtent(
        beam_d.fmap_size.x, beam_d.fmap_size.y, beam_d.long_dim);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&BEVTermaArray, &channelDesc, BEVTermaVolume));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = 
        make_cudaPitchedPtr(output.data(),
            beam_d.fmap_size.x * sizeof(float),
            beam_d.fmap_size.x,
            beam_d.fmap_size.y
        );
    copyParams.dstArray = BEVTermaArray;
    copyParams.extent = BEVTermaVolume;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = BEVTermaArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&BEVTermaTex, &texRes, &texDescr, NULL));

    // prepare output
    cudaPitchedPtr TermaPVCSPiched;
    cudaExtent PVCSExtent = make_cudaExtent(density_d.VolumeDim.x*sizeof(float),
        density_d.VolumeDim.y, density_d.VolumeDim.z);
    checkCudaErrors(cudaMalloc3D(&TermaPVCSPiched, PVCSExtent));
    checkCudaErrors(cudaMemset3D(TermaPVCSPiched, 0., PVCSExtent));

    BEV2PVCS(
        beam_d,
        density_d,
        TermaPVCSPiched,
        BEVTermaTex
    );

    // output Terma PVCS
    size_t pitched_volume_PVCS = density_d.VolumeDim.y * density_d.VolumeDim.z 
        * TermaPVCSPiched.pitch / sizeof(float);
    std::vector<float> TermaPVCSPitched_h(pitched_volume_PVCS);
    checkCudaErrors(cudaMemcpy(TermaPVCSPitched_h.data(),
        (void*)TermaPVCSPiched.ptr, pitched_volume_PVCS*sizeof(float), cudaMemcpyDeviceToHost));
    size_t density_volume = density_d.VolumeDim.x *
        density_d.VolumeDim.y * density_d.VolumeDim.z;
    std::vector<float> TermaPVCS_h(density_volume);
    pitched2contiguous(TermaPVCS_h, TermaPVCSPitched_h, density_d.VolumeDim.x,
        density_d.VolumeDim.y, density_d.VolumeDim.z, TermaPVCSPiched.pitch/sizeof(float));
    outputFile = fs::path(outputFolder) / std::string("TermaPVCS.bin");
    f.open(outputFile.string());
    if (! f) {
        std::cerr << "Could not open the file: " << outputFile.string() << std::endl;
        return 1;
    }
    f.write((char*)TermaPVCS_h.data(), density_volume*sizeof(float));
    f.close();

    // clean up
    checkCudaErrors(cudaDestroyTextureObject(BEVTermaTex));
    checkCudaErrors(cudaFreeArray(BEVTermaArray));
    checkCudaErrors(cudaFree(TermaPVCSPiched.ptr));


    // output density
    checkCudaErrors(cudaMemcpy(sample.data(), beam_d.DensityBEVPitch.ptr,
        pitched_volume*sizeof(float), cudaMemcpyDeviceToHost));
    pitched2contiguous(output, sample, width, height, depth, pitch);
    outputFile = fs::path(outputFolder) / std::string("DensityBEV.bin");
    f.open(outputFile.string());
    if (! f) {
        std::cerr << "Could not open the file: " << outputFile.string() << std::endl;
        return 1;
    }
    f.write((char*)output.data(), output_volume*sizeof(float));
    f.close();

    return 0;
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


bool fd::profile_TermaCompute(
    std::vector<BEAM_d>& beams_d,
    DENSITY_d& density_d,
    SPECTRUM_h& spectrum,
    const std::string& outputFolder
) {
    std::vector<cudaStream_t> streams(beams_d.size());
    for (int i=0; i<beams_d.size(); i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i=0; i<beams_d.size(); i++) {
        if (TermaCompute(
            beams_d[i],
            density_d,
            spectrum,
            streams[i]
        ))
            return 1;
    }

    for (int i=0; i<beams_d.size(); i++) {
        cudaStreamSynchronize(streams[i]);
    }
    for (int i=0; i<beams_d.size(); i++) {
        cudaStreamDestroy(streams[i]);
    }
    return 0;
}


bool fd::TermaComputeCollective (
    std::vector<BEAM_d>& beams,
    DENSITY_d& density_d,
    SPECTRUM_h& spectrum_h,
    cudaStream_t stream
) {
    // prepare data
    std::vector<d_BEAM_d> beams_input;
    for (int i=0; i<beams.size(); i++) {
        beams_input.emplace_back(d_BEAM_d(beams[i]));
    }
    d_BEAM_d* d_beams_input;
    checkCudaErrors(cudaMalloc((void**)(&d_beams_input), beams.size()*sizeof(d_BEAM_d)));
    checkCudaErrors(cudaMemcpy(d_beams_input, beams_input.data(),
        beams.size()*sizeof(d_BEAM_d), cudaMemcpyHostToDevice));
    

    std::vector<float*> fluence_maps(beams.size());
    for (int i=0; i<beams.size(); i++) {
        fluence_maps[i] = beams[i].fluence;
    }
    float** fluence_maps_input;
    checkCudaErrors(cudaMalloc((void***)&fluence_maps_input, beams.size()*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(fluence_maps_input, fluence_maps.data(),
        beams.size()*sizeof(float*), cudaMemcpyHostToDevice));


    std::vector<cudaPitchedPtr> TermaBEVPitch_array(beams.size());
    std::vector<cudaPitchedPtr> DenseBEVPitch_array(beams.size());
    for (int i=0; i<beams.size(); i++) {
        TermaBEVPitch_array[i] = beams[i].TermaBEVPitch;
        DenseBEVPitch_array[i] = beams[i].DensityBEVPitch;
        if (beams[i].TermaBEVPitch.pitch != beams[i].DensityBEVPitch.pitch) {
            std::cerr << "The pitches of the BEV Terma array and density "
                "array are not equal for beam " << i << std::endl;
            return 1;
        }
    }
    cudaPitchedPtr* TermaBEVPitch_array_input;
    cudaPitchedPtr* DenseBEVPitch_array_input;
    checkCudaErrors(cudaMalloc((void**)&TermaBEVPitch_array_input, beams.size()*sizeof(cudaPitchedPtr)));
    checkCudaErrors(cudaMalloc((void**)&DenseBEVPitch_array_input, beams.size()*sizeof(cudaPitchedPtr)));
    checkCudaErrors(cudaMemcpy(TermaBEVPitch_array_input, TermaBEVPitch_array.data(),
        beams.size()*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(DenseBEVPitch_array_input, DenseBEVPitch_array.data(),
        beams.size()*sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice));

    dim3 blockSize{DIM1, DIM1, 1};
    dim3 gridSize{
        (beams[0].fmap_size.x + blockSize.x - 1) / blockSize.x,
        (beams[0].fmap_size.y + blockSize.y - 1) / blockSize.y,
        static_cast<uint>(beams.size())
    };
    d_TermaComputeCollective<<<gridSize, blockSize, 0, stream>>>(
        d_beams_input,
        fluence_maps_input,
        TermaBEVPitch_array_input,
        DenseBEVPitch_array_input,
        density_d.densityTex,
        density_d.VoxelSize,
        spectrum_h.nkernels
    );

    // check some results.
    size_t pitched_volume_size = TermaBEVPitch_array[0].pitch / sizeof(float) 
        * beams[0].fmap_size.y * beams[0].long_dim;
    std::vector<float> TermaSample(pitched_volume_size, 0.);
    std::vector<float> DenseSample(pitched_volume_size, 0.);
    checkCudaErrors(cudaMemcpy(TermaSample.data(), TermaBEVPitch_array[0].ptr,
        pitched_volume_size*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(DenseSample.data(), DenseBEVPitch_array[0].ptr,
        pitched_volume_size*sizeof(float), cudaMemcpyDeviceToHost));
    double Terma_sum = 0.;
    double Dense_sum = 0.;
    for (size_t i=0; i<pitched_volume_size; i++) {
        Terma_sum += TermaSample[i];
        Dense_sum += DenseSample[i];
    }
    std::cout << "Collective Terma calculation finished. Terma sum: " <<
        Terma_sum << ", Dense sum: " << Dense_sum << std::endl << std::endl;


    // clean up
    checkCudaErrors(cudaFree(d_beams_input));
    checkCudaErrors(cudaFree(fluence_maps_input));
    checkCudaErrors(cudaFree(TermaBEVPitch_array_input));
    checkCudaErrors(cudaFree(DenseBEVPitch_array_input));
    return 0;
}

__global__ void
fd::d_TermaComputeCollective(
    d_BEAM_d* beams,
    float** fluence_maps,
    cudaPitchedPtr* TermaBEVPitch_array,
    cudaPitchedPtr* DenseBEVPitch_array,
    cudaTextureObject_t densityTex,
    float3 voxel_size,
    int nkern
) {
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx_z = blockIdx.z;

    d_BEAM_d beam_d = beams[idx_z];
    float* fluence_map = fluence_maps[idx_z];
    cudaPitchedPtr TermaBEVPitch = TermaBEVPitch_array[idx_z];
    cudaPitchedPtr DenseBEVPitch = DenseBEVPitch_array[idx_z];

    if (idx_x >= beam_d.fmap_size.x || idx_y >= beam_d.fmap_size.y)
        return;

    int dest_pitch = TermaBEVPitch.pitch / sizeof(float);
    float* dest_ptr = (float*)TermaBEVPitch.ptr;
    float* dens_ptr = (float*)DenseBEVPitch.ptr;
    
    int fluence_idx = idx_x + idx_y * beam_d.fmap_size.x;
    float fluence_value = fluence_map[fluence_idx];

    float3 pixel_center_minus_source_BEV {
        (idx_x + 0.5f - beam_d.fmap_size.x * 0.5f) * beam_d.beamlet_size.x,
        beam_d.sad,
        (idx_y + 0.5f - beam_d.fmap_size.y * 0.5f) * beam_d.beamlet_size.y
    };
    float3 pixel_center_minus_source_PVCS = d_inverseRotateBeamAtOriginRHS(
        pixel_center_minus_source_BEV, beam_d.angles.x, beam_d.angles.y, beam_d.angles.z);

    float3 step_size_PVCS = pixel_center_minus_source_PVCS * (beam_d.long_spacing / (beam_d.sad * SUPER_SAMPLING));
    float step_size_norm = length(step_size_PVCS);  // physical length

    // initialize to starting point
    float3 coords_PVCS = beam_d.source + pixel_center_minus_source_PVCS * (beam_d.lim_min / beam_d.sad);

    float3 step_size_PVCS_normalized = step_size_PVCS / voxel_size;
    float3 coords_PVCS_normalized = coords_PVCS / voxel_size;

    double radiological_path_length = 0.;
    for (int i=0; i<beam_d.long_dim; i++) {
        float t_sum_avg = 0.;
        float density_avg = 0.;
        #pragma unroll
        for (int j=0; j<SUPER_SAMPLING; j++) {
            coords_PVCS_normalized += step_size_PVCS_normalized;
            float density = tex3D<float>(densityTex, coords_PVCS_normalized.x, coords_PVCS_normalized.y, coords_PVCS_normalized.z);
            radiological_path_length += density * step_size_norm;
            density_avg += density;

            float tsum_local = 0.;
            for (int e=0; e<nkern; e++) {
                float this_fluence = d_fluence[e] * fluence_value;
                float this_energy = d_energy[e];
                // float this_mu_en = d_mu_en[e];
                float this_mu = d_mu[e];
                tsum_local += this_fluence * this_energy * this_mu * 
                    exp(- this_mu * radiological_path_length);
            }
            t_sum_avg += tsum_local;
        }
        t_sum_avg /= SUPER_SAMPLING;
        density_avg /= SUPER_SAMPLING;
        size_t global_idx = idx_x + dest_pitch * (idx_y + i * beam_d.fmap_size.y);
        dest_ptr[global_idx] = t_sum_avg;
        dens_ptr[global_idx] = density_avg;
    }
}