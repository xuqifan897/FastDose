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
    //     4. X value (times the number of phi angles)
    //     5. convolution directions (times the number of phi angles)
    //     6. base coordinates, which are the coordinates of the
    //        ray originating from the base voxel (times the number of phi angles)
    int sharedMemorySize = (1 + 1 + 1 + nPhi) * fmap_npixels * sizeof(float)
        + nPhi * sizeof(float3) + nPhi * sizeof(float3);
    d_DoseComputeCollective<<<gridSize, blockSize, sharedMemorySize, stream>>>(
        d_beams,
        TermaBEV_array,
        DensityBEV_array,
        DoseBEV_array,
        nTheta,
        nPhi
    );
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

    int pixel_idx = threadIdx.x;
    int idx_x = pixel_idx % beam.fmap_size.x;
    int idx_y = pixel_idx / beam.fmap_size.x;
    if (idx_y >= beam.fmap_size.y)
        return;

    // shared memory contains the following parts:
    //     1. Terma value
    //     2. Density value
    //     3. Dose value
    //     4. X value (times the number of phi angles)
    //     5. convolution directions (times the number of phi angles)
    //     6. base coordinates (times the number of phi angles)
    int fmap_npixels = beam.fmap_size.x * beam.fmap_size.y;
    extern __shared__ float sharedData[];
    float* SharedTerma = sharedData;
    float* SharedDensity = SharedTerma + fmap_npixels;
    float* SharedDose = SharedDensity + fmap_npixels;
    float* SharedX = SharedDose + fmap_npixels;
    float3* directions = (float3*)(SharedX + nPhi * fmap_npixels);
    float3* baseCoords = directions + nPhi;

    auto block = cooperative_groups::this_thread_block();
    // Here we assume beam.TermaBEV_pitch == beam.DoseBEV_pitch == beam.DensityBEV_pitch
    size_t pitch_float = beam.TermaBEV_pitch / sizeof(float);

    for (int thetaIdx=0; thetaIdx<nTheta; thetaIdx++) {
        float thetaAngle = d_theta[thetaIdx];
        // for now, we only consider forward scattering
        if (thetaAngle > CUDART_PIO2)
            continue;

        // Initialize SharedX
        for (int phiIdx=0; phiIdx<nPhi; phiIdx++) {
            SharedX[phiIdx * fmap_npixels + pixel_idx] = 0.f;
        }
        // __syncthreads();

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
            baseCoords[pixel_idx] = make_float3(.5f, .5f, 0.f);
        }


        while (true) {
            // load Terma and Density of the current slice
            int sliceIdx = (int)roundf(baseCoords[0].z);
            size_t global_idx = sliceIdx * pitch_float;
            cooperative_groups::memcpy_async(block, SharedTerma,
                TermaBEV+global_idx, fmap_npixels*sizeof(float));
            cooperative_groups::memcpy_async(block, SharedDensity,
                DensityBEV+global_idx, fmap_npixels*sizeof(float));

            for (int phi_idx=0; phi_idx<nPhi; phi_idx++) {
                // process the phi_idx
                float* SharedX_local = SharedX + phi_idx * fmap_npixels;
                float3& directions_local = directions[phi_idx];
                float3& baseCoords_local = baseCoords[phi_idx];

                // do the dose calculation
                while (true) {
                    // compute the next interaction point
                    bool flag;
                    __syncthreads();  // synchronization between computataion
                    float3 np = d_nextPoint(baseCoords_local, directions_local, &flag);
                    
                    // float lineSeg = 

                    // reaches the next slice
                    if (flag)
                        break;
                }
            }
            
            // stop condition
            if (baseCoords[0].z + eps_fastdose >= beam.long_dim)
                break;
        }

        // int iter_x_prev = -1;
        // int iter_y_prev = -1;
        // int iter_z_prev = -1;
        // int iter_x = 0;
        // int iter_y = 0;
        // int iter_z = 0;
        // while (iter_z < beam.long_dim) {
        //     if (iter_z != iter_z_prev) {
        //         // At a new slice now, initialize the dose matrix
        //         size_t global_idx = iter_z * pitch_float;
        //         cooperative_groups::memcpy_async(block, SharedTerma,
        //             TermaBEV+global_idx, fmap_npixels*sizeof(float));
        //         cooperative_groups::memcpy_async(block, SharedDensity,
        //             DensityBEV+global_idx, fmap_npixels*sizeof(float));
        //         SharedDose[pixel_idx] = 0.f;
        //     }
        // }
    }
}


bool fastdose::test_DoseComputeCollective(std::vector<BEAM_d>& beams,
        const std::string& outputFolder, const KERNEL_h& kernel_h, cudaStream_t stream
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

    
    if (DoseComputeCollective (
        fmap_npixels,
        beams.size(),
        d_beams,
        TermaBEV_array,
        DenseBEV_array,
        DoseBEV_array,
        kernel_h.nTheta,
        kernel_h.nPhi,
        stream
    ))
        return 1;


    // clean up
    checkCudaErrors(cudaFree(d_beams));
    checkCudaErrors(cudaFree(TermaBEV_array));
    checkCudaErrors(cudaFree(DenseBEV_array));
    checkCudaErrors(cudaFree(DoseBEV_array));
    return 0;
}