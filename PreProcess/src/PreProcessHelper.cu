#include "PreProcessHelper.h"
#include "PreProcessArgs.h"
#include <unistd.h>
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_math.h"

std::string PreProcess::get_username() {
    char* username = getlogin();
    if (username == nullptr) {
        return std::string("root");
    } else {
        return std::string(username);
    }
}

bool PreProcess::CreateIsoDensity(
    const FloatVolume& source, FloatVolume& target,
    CTLUT* ctlut, bool verbose=false) {
    float voxelSize = getarg<float>("voxelSize");

    /*    Build the input 3D texture    */
    cudaArray_t imgArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent imgSize = make_cudaExtent(source.size.x, source.size.y, source.size.z);
    checkCudaErrors(cudaMalloc3DArray(&imgArray, &channelDesc, imgSize));

    cudaMemcpy3DParms imgParams = {0};
    imgParams.srcPtr = make_cudaPitchedPtr((void*)source.data(),
        imgSize.width*sizeof(float), imgSize.width, imgSize.height);
    imgParams.dstArray = imgArray;
    imgParams.extent = imgSize;
    imgParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&imgParams);

    cudaResourceDesc texImgRes;
    memset(&texImgRes, 0, sizeof(cudaResourceDesc));
    texImgRes.resType = cudaResourceTypeArray;
    texImgRes.res.array.array = imgArray;

    cudaTextureDesc texImgDescr;
    memset(&texImgDescr, 0, sizeof(cudaTextureDesc));
    texImgDescr.normalizedCoords = false;
    texImgDescr.filterMode = cudaFilterModeLinear;
    texImgDescr.addressMode[0] = cudaAddressModeBorder;
    texImgDescr.addressMode[1] = cudaAddressModeBorder;
    texImgDescr.addressMode[2] = cudaAddressModeBorder;

    cudaTextureObject_t texImg;
    checkCudaErrors(cudaCreateTextureObject(&texImg, &texImgRes, &texImgDescr, NULL));

    
    // calculate new dimensions
    uint3 iso_size = float2uint_ceil(source.voxsize*(make_float3(source.size)/make_float3(voxelSize)));
    uint iso_count = product(iso_size);
    uint iso_memsize = iso_count * sizeof(float);

    float* iso_matrix = nullptr;
    checkCudaErrors(cudaMalloc((void**)&iso_matrix, iso_memsize));
    checkCudaErrors(cudaMemset(iso_matrix, 0, iso_memsize) );


    // allocate LUT in cuda
    size_t lutsize = ctlut->points.size();
    std::vector<float> h_hunits(lutsize, 0.0f);
    std::vector<float> h_massdens(lutsize, 0.0f);
    for (int i=0; i<lutsize; i++) {
        h_hunits[i] = ctlut->points[i].hunits;
        h_massdens[i] = ctlut->points[i].massdens;
    }
    float* d_hunits = nullptr;
    float* d_massdens = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_hunits, lutsize*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_massdens, lutsize*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_hunits, h_hunits.data(), lutsize*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_massdens, h_massdens.data(), lutsize*sizeof(float), cudaMemcpyHostToDevice));
    size_t isoShared = 2 * sizeof(float) * lutsize;

    dim3 blockSize(256, 1, 1);
    dim3 gridSize((iso_count + blockSize.x - 1) / blockSize.x, 1, 1);
    cudaMakeIsotropicWithLUT<<<gridSize, blockSize, isoShared, 0>>>(
        iso_matrix, source.voxsize, voxelSize, iso_size,
        d_hunits, d_massdens, lutsize, texImg);
    getLastCudaError("cudaMakeIsotropic()");

    target.size = iso_size;
    target.voxsize = make_float3(voxelSize);
    target.start = source.start;
    target.set_data(target.nvoxels());
    checkCudaErrors(cudaMemcpy(target.data(), iso_matrix, iso_memsize, cudaMemcpyDeviceToHost));

    // clean-up
    checkCudaErrors(cudaFree(d_hunits));
    checkCudaErrors(cudaFree(d_massdens));
    checkCudaErrors(cudaDestroyTextureObject(texImg));
    checkCudaErrors(cudaFreeArray(imgArray));
    return 0;
}

__global__ void
PreProcess::cudaMakeIsotropicWithLUT( float *iso, float3 voxelSize, float iso_voxel, uint3 iso_size, 
    float* lut_hunits, float* lut_massdens, int nlut, cudaTextureObject_t texObj) {
    extern __shared__ float s[];
    // find the overall ID number of this thread, tid (thread index)
    unsigned int bidx = blockIdx.x + gridDim.x * ( blockIdx.y + gridDim.y * blockIdx.z );
    unsigned int tidx = threadIdx.x + blockDim.x * bidx;
    float* s_lut_hunits = s;
    float* s_lut_massdens = &s[nlut];

    if (blockDim.x >= nlut) {
        if (threadIdx.x < nlut) {
            s_lut_hunits[threadIdx.x] = lut_hunits[threadIdx.x];
            s_lut_massdens[threadIdx.x] = lut_massdens[threadIdx.x];
        }
    } else {
        for (int jj=0; jj<ceilf((float)nlut/blockDim.x); jj++) {
            int ii = jj*blockDim.x + threadIdx.x;
            if (ii < nlut) {
                s_lut_hunits[ii] = lut_hunits[ii];
                s_lut_massdens[ii] = lut_massdens[ii];
            }
        }
    }
    __syncthreads();

	// convert tid into 3D coordinates based on the size of the isotropic volume
    unsigned int slicesize = (iso_size.x*iso_size.y);
    unsigned int X = (tidx % slicesize) % iso_size.x;
    unsigned int Y = (tidx % slicesize) / iso_size.x;
    unsigned int Z = (tidx / slicesize);
    if (X >= iso_size.x || Y >= iso_size.y || Z >= iso_size.z) { return; }

	// convert that to physical distance in mm
    float3 pos = make_float3(X, Y, Z) * make_float3(iso_voxel);

    // convert from physical point within the isotropic volume
    // to a set of coordinates in the original data
    pos /= voxelSize;

    // same the point from the texture memory
    // 0.5f is added to each coordinate as a shift to the center of the voxel
    // float value = tex3D( texImg, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
    float value = tex3D<float>(texObj, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);

    // find low
    if (value <= s_lut_hunits[0]) {
        // lower bound
        value = s_lut_massdens[0];
    }
    else if (value >= s_lut_hunits[nlut-1]) {
        // extrapolate (just based on fit from last 2 data points)
        value = s_lut_massdens[nlut-2] +
            (value - s_lut_hunits[nlut-2]) *
            (s_lut_massdens[nlut-1]-s_lut_massdens[nlut-2])/(s_lut_hunits[nlut-1]-s_lut_hunits[nlut-2]);
    }
    else {
        // interpolate
        int lowidx = 0;
        while(s_lut_hunits[lowidx]<value && lowidx < nlut-1) {lowidx++;}
        lowidx--;
        value = s_lut_massdens[lowidx] +
            (value - s_lut_hunits[lowidx]) *
            (s_lut_massdens[lowidx+1]-s_lut_massdens[lowidx])/(s_lut_hunits[lowidx+1]-s_lut_hunits[lowidx]);
    }

	// write to output in DCS
    iso[tidx] = value;
}