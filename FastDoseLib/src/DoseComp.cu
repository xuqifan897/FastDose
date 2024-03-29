#include "fastdose.cuh"
#include "DoseComp.cuh"
#include "kernel.cuh"
#include "macros.h"

#include "helper_cuda.h"
#include "helper_math.cuh"
#include "math_constants.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

namespace fastdose {
    __constant__ float d_paramA[MAX_THETA_ANGLES];
    __constant__ float d_parama[MAX_THETA_ANGLES];
    __constant__ float d_paramB[MAX_THETA_ANGLES];
    __constant__ float d_paramb[MAX_THETA_ANGLES];
    __constant__ float d_theta[MAX_THETA_ANGLES];
    __constant__ float d_phi[MAX_THETA_ANGLES];
}

bool fastdose::KERNEL_h::bind_kernel() {
    if (this->nTheta > MAX_THETA_ANGLES || this->nPhi > MAX_THETA_ANGLES) {
        std::cerr << "The number of theta or phi angles included is more than MAX_THETA_ANGLES ("
            << MAX_THETA_ANGLES << ")" << std::endl;
        return 1;
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_paramA, this->paramA.data(), this->nTheta*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_parama, this->parama.data(), this->nTheta*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_paramB, this->paramB.data(), this->nTheta*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_paramb, this->paramb.data(), this->nTheta*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_theta, this->thetaMiddle.data(), this->nTheta*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_phi, this->phiAngles.data(), this->nPhi*sizeof(float)));
    return 0;
}

__global__ void
fastdose::d_test_kernel(float* output, int width, int idx) {
    int ii = threadIdx.x;
    if (ii >= width)
        return;
    if (idx == 0) {
        output[ii] = d_paramA[ii];
    } else if (idx == 1) {
        output[ii] = d_parama[ii];
    } else if (idx == 2) {
        output[ii] = d_paramB[ii];
    } else if (idx == 3) {
        output[ii] = d_paramb[ii];
    } else if (idx == 4) {
        output[ii] = d_theta[ii];
    } else if (idx == 5) {
        output[ii] = d_phi[ii];
    }
}


namespace fastdose {
    static __device__ float3 d_angle2Vector(
        float theta, float phi
    ) {
        float sin_theta, cos_theta, sin_phi, cos_phi;
        fast_sincosf(theta, &sin_theta, &cos_theta);
        fast_sincosf(phi, &sin_phi, &cos_phi);
        float3 result{
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta};
        return result;
    }

    
    static __device__ float3 d_nextPoint(
        const float3& currentLocation,
        const float3& direction,
        bool* flag
    ) {
        float3 stepSize;

        if (direction.x > 0) {
            stepSize.x = (ceilf(currentLocation.x + larger_eps_fastdose) - currentLocation.x) / direction.x;
        } else {
            stepSize.x = (floorf(currentLocation.x - larger_eps_fastdose) - currentLocation.x) / direction.x;
        }

        if (direction.y > 0) {
            stepSize.y = (ceilf(currentLocation.y + larger_eps_fastdose) - currentLocation.y) / direction.y;
        } else {
            stepSize.y = (floorf(currentLocation.y - larger_eps_fastdose) - currentLocation.y) / direction.y;
        }

        if (direction.z > 0) {
            stepSize.z = (ceilf(currentLocation.z + larger_eps_fastdose) - currentLocation.z) / direction.z;
        } else {
            stepSize.z = (floorf(currentLocation.z - larger_eps_fastdose) - currentLocation.z) / direction.z;
        }

        float stepTake = fmin(fmin(stepSize.x, stepSize.y), stepSize.z);
        float3 result = currentLocation + stepTake * direction;
        *flag = abs(result.z - roundf(result.z)) < eps_fastdose;

        if (*flag) {
            stepTake = stepSize.z;
            result = currentLocation + stepTake * direction;
            result.z = roundf(result.z);
        }
        return result;
    }


    static __device__ float d_calcLineSeg(
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
        return sqrtf(dot(diff, diff));
    }


    static __device__ int3 d_calcCoords(
        const float3& mid_point, const d_BEAM_d& beam
    ) {
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


    static __device__ float d_calcg(float x) {
        // This function calculates the approximation of Tayler expansion:
        // (1 - e^{-x}) / x = 1 - 1/2x + 1/6x^2 - 1/24x^3 ...
        float result = x * (- 1 / 24);
        result = x * ( 1 / 6 + result);
        result = x * (- 1 / 2 + result);
        result = 1 + result;
        return result;
    }
}


bool fastdose::DoseComputeCollective(
    size_t fmap_npixels,
    size_t n_beams,
    d_BEAM_d* d_beams,
    float** TermaBEV_array,
    float** DensityBEV_array,
    float** DoseBEV_array,
    int nTheta,
    int nPhi,
    cudaStream_t stream
) {
    dim3 blockSize(1, 1, 1);
    blockSize.x = ((fmap_npixels + WARPSIZE - 1) / WARPSIZE) * WARPSIZE;
    dim3 gridSize(n_beams, 1, 1);

    // shared memory contains the following parts:
    //     1. Terma value
    //     2. Density value
    //     3. Dose value
    //     4. cumulative_pi
    //     4. X value (times the number of phi angles, for A and B, respectively)
    //     5. convolution directions (times the number of phi angles)
    //     6. base coordinates, which are the coordinates of the
    //        ray originating from the base voxel (times the number of phi angles)
    int sharedMemorySize = (1 + 1 + 1 + 1 + 2 * nPhi) * fmap_npixels * sizeof(float)
        + nPhi * sizeof(float3) + nPhi * sizeof(float3);
    d_DoseComputeCollective<<<gridSize, blockSize, sharedMemorySize, stream>>>(
        d_beams,
        TermaBEV_array,
        DensityBEV_array,
        DoseBEV_array,
        nTheta,
        nPhi
    );
    return 0;
}


__global__ void
fastdose::d_DoseComputeCollective(
    d_BEAM_d* beams,
    float** TermaBEV_array,
    float** DensityBEV_array,
    float** DoseBEV_array,
    int nTheta,
    int nPhi
) {
    int beam_idx = blockIdx.x;
    d_BEAM_d beam = beams[beam_idx];
    float* TermaBEV = TermaBEV_array[beam_idx];
    float* DensityBEV = DensityBEV_array[beam_idx];
    float* DoseBEV = DoseBEV_array[beam_idx];

    int pixel_idx = threadIdx.x;
    int idx_x = pixel_idx % beam.fmap_size.x;
    int idx_y = pixel_idx / beam.fmap_size.x;
    if (idx_y >= beam.fmap_size.y)
        return;

    // shared memory contains the following parts:
    //     1. Terma value
    //     2. Density value
    //     3. Dose value
    //     4. cumulative_pi
    //     4. X value (times the number of phi angles, for A and B, respectively)
    //     5. convolution directions (times the number of phi angles)
    //     6. base coordinates, which are the coordinates of the
    //        ray originating from the base voxel (times the number of phi angles)
    int fmap_npixels = beam.fmap_size.x * beam.fmap_size.y;
    extern __shared__ float sharedData[];
    float* SharedTerma = sharedData;
    float* SharedDensity = SharedTerma + fmap_npixels;
    float* SharedDose = SharedDensity + fmap_npixels;
    float* cumu_p_i = SharedDose + fmap_npixels;
    float* SharedXA = cumu_p_i + fmap_npixels;
    float* SharedXB = SharedXA + nPhi * fmap_npixels;
    float3* directions = (float3*)(SharedXB + nPhi * fmap_npixels);
    float3* baseCoords = directions + nPhi;

    auto block = cooperative_groups::this_thread_block();
    // Here we assume beam.TermaBEV_pitch == beam.DoseBEV_pitch == beam.DensityBEV_pitch
    size_t pitch_float = beam.TermaBEV_pitch / sizeof(float);

    for (int thetaIdx=0; thetaIdx<nTheta; thetaIdx++) {
        float thetaAngle = d_theta[thetaIdx];
        // for now, we only consider forward scattering
        if (thetaAngle > CUDART_PIO2)
            continue;

        float A = d_paramA[thetaIdx];
        float a = d_parama[thetaIdx];
        float B = d_paramB[thetaIdx];
        float b = d_paramb[thetaIdx];

        // Initialize SharedX
        for (int phiIdx=0; phiIdx<nPhi; phiIdx++) {
            SharedXA[phiIdx * fmap_npixels + pixel_idx] = 0.f;
            SharedXB[phiIdx * fmap_npixels + pixel_idx] = 0.f;
        }

        // Initialize directions
        if (pixel_idx < nPhi) {
            // physical direction
            directions[pixel_idx] = d_angle2Vector(thetaAngle, d_phi[pixel_idx]);
            // convert physical direction to normalized direction w.r.t. voxel size
            directions[pixel_idx] = make_float3(
                directions[pixel_idx].x / beam.beamlet_size.x,
                directions[pixel_idx].y / beam.beamlet_size.y,
                directions[pixel_idx].z / beam.long_spacing
            );
            // normalize direction
            directions[pixel_idx] = normalize(directions[pixel_idx]);
        
            // Initialize baseCoords
            baseCoords[pixel_idx] = make_float3(0.5f, 0.5f, 0.0f);
        }
        __syncthreads();


        while (true) {
            // load Terma and Density of the current slice, and initialize the Dose
            int sliceIdx = (int)roundf(baseCoords[0].z);
            if (sliceIdx >= beam.long_dim)
                break;
            
            size_t global_idx = sliceIdx * pitch_float;
            cooperative_groups::memcpy_async(block, SharedTerma,
                TermaBEV+global_idx, fmap_npixels*sizeof(float));
            cooperative_groups::memcpy_async(block, SharedDensity,
                DensityBEV+global_idx, fmap_npixels*sizeof(float));
            cooperative_groups::wait(block);  // synchronize immediately
            
            SharedDose[pixel_idx] = 0.0f;

            for (int phi_idx=0; phi_idx<nPhi; phi_idx++) {
                // process the phi_idx
                float* SharedXA_local = SharedXA + phi_idx * fmap_npixels;
                float* SharedXB_local = SharedXB + phi_idx * fmap_npixels;
                float3& directions_local = directions[phi_idx];
                float3& baseCoords_local = baseCoords[phi_idx];

                // initialize cumu_p_i
                cumu_p_i[pixel_idx] = 0.;

                // do the dose calculation
                while (true) {
                    // compute the next interaction point
                    bool flag;
                    float3 np = d_nextPoint(baseCoords_local, directions_local, &flag);
                    
                    float3 baseCoords_thread {
                        baseCoords_local.x + idx_x,
                        baseCoords_local.y + idx_y,
                        baseCoords_local.z
                    };

                    float3 np_thread {
                        np.x + idx_x,
                        np.y + idx_y,
                        np.z
                    };

                    // determine the current voxel idx
                    float3 mid_point = (baseCoords_thread + np_thread) * 0.5f;
                    int3 mid_idx = d_calcCoords(mid_point, beam);
                    int mid_idx_linear = mid_idx.x + mid_idx.y * beam.fmap_size.x;

                    float localTerma = SharedTerma[mid_idx_linear];
                    float localDensity = SharedDensity[mid_idx_linear];
                    float localXA = SharedXA_local[pixel_idx];
                    float localXB = SharedXB_local[pixel_idx];

                    float p_i = d_calcLineSeg(baseCoords_thread, np_thread, beam) * localDensity;
                    float ap_i = a * p_i;
                    float bp_i = b * p_i;
                    float exp_minus_ap_i = __expf(-ap_i);
                    float exp_minus_bp_i = __expf(-bp_i);
                    float ga_i = d_calcg(ap_i);
                    float gb_i = d_calcg(bp_i);

                    // float lineSegDose = B / b * ((1 - gb_i) * localTerma + gb_i * localXB);
                    // float lineSegDose = A / a * ((1 - ga_i) * localTerma + ga_i * localXA);
                    float lineSegDose = 
                        A / a * ((1 - ga_i) * localTerma + ga_i * localXA) +
                        B / b * ((1 - gb_i) * localTerma + gb_i * localXB);
                    SharedDose[mid_idx_linear] += lineSegDose * p_i;
                    cumu_p_i[mid_idx_linear] += p_i;
                    
                    // update SharedX
                    SharedXA_local[pixel_idx] = localTerma * (1 - exp_minus_ap_i) + exp_minus_ap_i * localXA;
                    SharedXB_local[pixel_idx] = localTerma * (1 - exp_minus_bp_i) + exp_minus_bp_i * localXB;

                    // to tell if the ray has crossed a boundary
                    if (floorf(baseCoords_thread.x / beam.fmap_size.x) != floorf(np_thread.x / beam.fmap_size.x) ||
                        floorf(baseCoords_thread.y / beam.fmap_size.y) != floorf(np_thread.y / beam.fmap_size.y)
                    ) {
                        SharedXA_local[pixel_idx] = 0.0f;
                        SharedXB_local[pixel_idx] = 0.0f;
                    }

                    // update baseCoords_local
                    if (pixel_idx == 0)
                        baseCoords_local = np;
                    __syncthreads();

                    // reaches the next slice
                    if (flag)
                        break;
                }
            }
            DoseBEV[global_idx + pixel_idx] += SharedDose[pixel_idx] / (cumu_p_i[pixel_idx] + eps_fastdose);
        }
    }
}


bool fastdose::test_DoseComputeCollective(
    std::vector<BEAM_d>& beams, DENSITY_d& density_d,
    const std::string& outputFolder, int FmapOn,
    const KERNEL_h& kernel_h, cudaStream_t stream
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

#if false
    // Test the function calcLineSeg
    if (test_calcLineSeg(h_beams))
        return 1;
    if (test_calcCoords(h_beams))
        return 1;
    checkCudaErrors(cudaFree(d_beams));
    return 0;
#endif

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

    // allocate DoseBEV_array
    std::vector<float*> h_DoseBEV_array(beams.size(), nullptr);
    for (int i=0; i<beams.size(); i++)
        h_DoseBEV_array[i] = beams[i].DoseBEV;
    float** DoseBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void**)&DoseBEV_array, beams.size()*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(DoseBEV_array, h_DoseBEV_array.data(),
        beams.size()*sizeof(float*), cudaMemcpyHostToDevice));

    size_t fmap_npixels = beams[0].fmap_size.x * beams[0].fmap_size.y;

    // for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    DoseComputeCollective(
        fmap_npixels,
        beams.size(),
        d_beams,
        TermaBEV_array,
        DenseBEV_array,
        DoseBEV_array,
        kernel_h.nTheta,
        kernel_h.nPhi,
        stream
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Dose time elapsed: " << milliseconds << " [ms]"<< std::endl;

    // retrieve result
    size_t nVoxels = beams[0].fmap_size.x * beams[0].fmap_size.y * beams[0].long_dim;
    size_t pitch_host = beams[0].fmap_size.x * beams[0].fmap_size.y;
    std::vector<float> DoseBEV_example(nVoxels, 0.0f);
    checkCudaErrors(cudaMemcpy2D(
        DoseBEV_example.data(), pitch_host*sizeof(float),
        beams[0].DoseBEV, beams[0].DoseBEV_pitch,
        pitch_host*sizeof(float), beams[0].long_dim, cudaMemcpyDeviceToHost));
    
    fs::path DoseFile = fs::path(outputFolder) /
        (std::string("DoseBEVFmap") + std::to_string(FmapOn) + std::string(".bin"));
    std::ofstream f(DoseFile.string());
    if (! f.is_open()) {
        std::cerr << "Could not open file: " << DoseFile.string() << std::endl;
        return 1;
    }
    f.write((char*)DoseBEV_example.data(), nVoxels*sizeof(float));
    f.close();

    // clean up
    checkCudaErrors(cudaFree(d_beams));
    checkCudaErrors(cudaFree(TermaBEV_array));
    checkCudaErrors(cudaFree(DenseBEV_array));
    checkCudaErrors(cudaFree(DoseBEV_array));


    // transfer BEV to PVCS. Construct texture memory first
    cudaArray* DoseBEV_Arr;
    cudaTextureObject_t DoseBEV_Tex;
    cudaExtent volumeSize = make_cudaExtent(
        beams[0].fmap_size.x, beams[0].fmap_size.y, beams[0].long_dim);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&DoseBEV_Arr, &channelDesc, volumeSize));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = 
        make_cudaPitchedPtr(DoseBEV_example.data(),
            beams[0].fmap_size.x * sizeof(float),
            beams[0].fmap_size.x, beams[0].fmap_size.y);
    copyParams.dstArray = DoseBEV_Arr;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = DoseBEV_Arr;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&DoseBEV_Tex, &texRes, &texDescr, NULL));

    // Construct destination array
    cudaExtent DosePVCS_Arr_extent = make_cudaExtent(
        density_d.VolumeDim.x * sizeof(float),
        density_d.VolumeDim.y, density_d.VolumeDim.z);
    cudaPitchedPtr DosePVCS_Arr;
    checkCudaErrors(cudaMalloc3D(&DosePVCS_Arr, DosePVCS_Arr_extent));

    std::cout << "beams[0] angles: " << beams[0].angles << std::endl;
    std::cout << "beams[0] source: " << beams[0].source << std::endl;
    BEV2PVCS(beams[0], density_d, DosePVCS_Arr, DoseBEV_Tex, stream);

    // write result
    size_t pitchedVolume = DosePVCS_Arr.pitch / sizeof(float) * 
        density_d.VolumeDim.y * density_d.VolumeDim.z;
    size_t volume = density_d.VolumeDim.x * density_d.VolumeDim.y *
        density_d.VolumeDim.z;
    std::vector<float> DosePVCS_pitched(pitchedVolume);
    std::vector<float> DosePVCS(volume);
    checkCudaErrors(cudaMemcpy(DosePVCS_pitched.data(), DosePVCS_Arr.ptr,
        pitchedVolume*sizeof(float), cudaMemcpyDeviceToHost));
    pitched2contiguous(DosePVCS, DosePVCS_pitched,
        density_d.VolumeDim.x, density_d.VolumeDim.y, density_d.VolumeDim.z,
        DosePVCS_Arr.pitch / sizeof(float));
    fs::path DosePVCSFile = fs::path(outputFolder) /
        (std::string("DosePVCSFmap") + std::to_string(FmapOn) + std::string(".bin"));
    f.open(DosePVCSFile.string());
    if (! f.is_open()) {
        std::cerr << "Could not open file " << DosePVCSFile.string() << std::endl;
        return 1;
    }
    f.write((char*)DosePVCS.data(), volume*sizeof(float));
    f.close();

    return 0;
}