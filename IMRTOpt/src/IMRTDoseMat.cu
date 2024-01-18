#include <fstream>
#include <string>
#include <iomanip>
#include <limits>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "IMRTDoseMat.cuh"
#include "IMRTArgs.h"
#include "IMRTgeom.cuh"

namespace fd = fastdose;

bool IMRT::MatCSR::dense2sparse(
    float* d_dense, size_t num_rows, size_t num_cols, size_t ld
) {
    checkCudaErrors(cudaMalloc((void**)(&this->d_csr_offsets),
        (num_rows + 1) * sizeof(int)));
    
    cusparseHandle_t handle = nullptr;
    cusparseDnMatDescr_t matDense;
    void* dBufferConstruct = nullptr;
    size_t bufferSize = 0;

    checkCusparse(cusparseCreate(&handle))

    checkCusparse(cusparseCreateDnMat(
        &matDense, num_rows, num_cols, ld,
        d_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW))

    checkCusparse(cusparseCreateCsr(
        &(this->matA), num_rows, num_cols, 0,
        d_csr_offsets, nullptr, nullptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    
    // allocate an external buffer if needed
    checkCusparse(cusparseDenseToSparse_bufferSize(
        handle, matDense, this->matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &bufferSize))
    checkCudaErrors(cudaMalloc((void**) &dBufferConstruct, bufferSize));
    
    // execute Dense to Sparse conversion
    checkCusparse(cusparseDenseToSparse_analysis(
        handle, matDense, this->matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBufferConstruct))

    // get the number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp;
    checkCusparse(cusparseSpMatGetSize(
        this->matA, &num_rows_tmp, &num_cols_tmp, &(this->nnz)))
    
    // allocate CSR column indices and values
    checkCudaErrors(cudaMalloc((void**) &(this->d_csr_columns), nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &(this->d_csr_values), nnz*sizeof(float)));
    // reset offsets, column indices, and values pointers
    checkCusparse(cusparseCsrSetPointers(this->matA,
        this->d_csr_offsets, this->d_csr_columns, this->d_csr_values))
    
    // execute Dense to Sparse conversion
    checkCusparse(cusparseDenseToSparse_convert(handle, matDense, this->matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBufferConstruct))
    
    checkCudaErrors(cudaFree(dBufferConstruct));
    checkCusparse(cusparseDestroyDnMat(matDense))
    checkCusparse(cusparseDestroy(handle))
    return 0;
}

bool IMRT::DoseMatConstruction(
    std::vector<BeamBundle>& beam_bundles,
    fd::DENSITY_d& density_d,
    fd::SPECTRUM_h& spectrum_h,
    fd::KERNEL_h& kernel_h,
    cudaStream_t stream
) {
    // prepare the sparse dose matrices
    std::vector<MatCSR> SparseMatArray;

    cudaEvent_t start, stop, globalStart, globalStop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&globalStart));
    checkCudaErrors(cudaEventCreate(&globalStop));
    float milliseconds;

    int concurrency = getarg<int>("concurrency");
    float extent = getarg<float>("extent");
    int iterations = (beam_bundles.size() + concurrency - 1) / concurrency;
    SparseMatArray.resize(iterations);

    // get the maximum number of beamlets, and the maximum beamlet length
    int maxNumBeamletsPerBatch = 0;
    int maxBeamletLength = 0;
    for (int i=0; i<iterations; i++) {
        int localNumBeamlets = 0;
        int beam_bundle_idx_begin = i * concurrency;
        int beam_bundle_idx_end = (i + 1) * concurrency;
        beam_bundle_idx_end = min(beam_bundle_idx_end, (int)(beam_bundles.size()));
        for (int j=beam_bundle_idx_begin; j<beam_bundle_idx_end; j++) {
            localNumBeamlets += beam_bundles[j].beams_h.size();
            for (int k=0; k<beam_bundles[j].beams_h.size(); k++)
                maxBeamletLength = max(maxBeamletLength, beam_bundles[j].beams_h[k].long_dim);
        }
        maxNumBeamletsPerBatch = max(maxNumBeamletsPerBatch, localNumBeamlets);
    }
    std::cout << std::endl << "Maximum number of beamlets per batch: " << maxNumBeamletsPerBatch
        << ", maximum beamlet length: " << maxBeamletLength << std::endl << std::endl;
    
    // allocate working buffers
    uint3 densityDim = density_d.VolumeDim;
    // for safty check
    unsigned long long denseDoseMatSize_ = maxNumBeamletsPerBatch * densityDim.x * densityDim.y * densityDim.z;
    if (denseDoseMatSize_ > std::numeric_limits<size_t>::max()) {
        std::cerr << "The size of the dense dose matrix is " << denseDoseMatSize_
            << ", which is beyond the range size_t can represent. "
            "Please reduce the concurrency parameter" << std::endl;
        return 1;
    }
    size_t denseDoseMatSize = maxNumBeamletsPerBatch * densityDim.x * densityDim.y * densityDim.z;
    float* d_denseDoseMat = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_denseDoseMat, denseDoseMatSize*sizeof(float)));

    fd::d_BEAM_d* d_BeamletsBuffer = nullptr;
    float* d_FluenceBuffer = nullptr;
    float* d_DensityBEVBuffer = nullptr;
    float* d_TeramBEVBuffer = nullptr;
    float* d_DoseBEVBuffer = nullptr;
    int subFluenceDim = getarg<int>("subFluenceDim");
    int subFluenceOn = getarg<int>("subFluenceOn");
    size_t bufferSize = maxNumBeamletsPerBatch * subFluenceDim * subFluenceDim * maxBeamletLength;
    checkCudaErrors(cudaMalloc((void**)(&d_BeamletsBuffer),
        maxNumBeamletsPerBatch * sizeof(fd::d_BEAM_d)));
    checkCudaErrors(cudaMalloc((void**)(&d_FluenceBuffer), maxNumBeamletsPerBatch
        * subFluenceDim * subFluenceDim * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_DensityBEVBuffer, bufferSize*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_TeramBEVBuffer, bufferSize*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_DoseBEVBuffer, bufferSize*sizeof(float)));

    // Fluence buffer can be initialized before-hands
    std::vector<float> h_FluenceBuffer(maxNumBeamletsPerBatch * subFluenceDim * subFluenceDim);
    std::vector<float> h_SingleFluence(subFluenceDim * subFluenceDim, 0.0f);
    int FmapLeadingX = static_cast<int>((subFluenceDim - subFluenceOn) * 0.5f);
    int FmapLeadingY = FmapLeadingX;
    for (int j=FmapLeadingY; j<FmapLeadingY + subFluenceOn; j++) {
        for (int i=FmapLeadingX; i<FmapLeadingX + subFluenceOn; i++) {
            int idx = i + j * subFluenceDim;
            h_SingleFluence[idx] = 1.0f;
        }
    }
    for (int j=0; j<maxNumBeamletsPerBatch; j++) {
        size_t globalOffset = j * subFluenceDim * subFluenceDim;
        for (int i=0; i<subFluenceDim*subFluenceDim; i++) {
            h_FluenceBuffer[i + globalOffset] = h_SingleFluence[i];
        }
    }
    checkCudaErrors(cudaMemcpy(d_FluenceBuffer, h_FluenceBuffer.data(),
        maxNumBeamletsPerBatch*subFluenceDim*subFluenceDim*sizeof(float),
        cudaMemcpyHostToDevice));


    // buffer array
    std::vector<float*> h_FluenceArray(maxNumBeamletsPerBatch, nullptr);
    for (int i=0; i<maxNumBeamletsPerBatch; i++)
        h_FluenceArray[i] = d_FluenceBuffer + i * subFluenceDim * subFluenceDim;
    std::vector<float*> h_DensityArray(maxNumBeamletsPerBatch, nullptr);
    std::vector<float*> h_TermaArray(maxNumBeamletsPerBatch, nullptr);
    std::vector<float*> h_DoseArray(maxNumBeamletsPerBatch, nullptr);
    for (int i=0; i<maxNumBeamletsPerBatch; i++) {
        size_t offset = i * subFluenceDim * subFluenceDim * maxBeamletLength;
        h_DensityArray[i] = d_DensityBEVBuffer + offset;
        h_TermaArray[i] = d_TeramBEVBuffer + offset;
        h_DoseArray[i] = d_DoseBEVBuffer + offset;
    }
    float** d_FluenceArray = nullptr;
    float** d_DensityArray = nullptr;
    float** d_TermaArray = nullptr;
    float** d_DoseArray = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_FluenceArray, maxNumBeamletsPerBatch*sizeof(float*)));
    checkCudaErrors(cudaMalloc((void***)&d_DensityArray, maxNumBeamletsPerBatch*sizeof(float*)));
    checkCudaErrors(cudaMalloc((void***)&d_TermaArray, maxNumBeamletsPerBatch*sizeof(float*)));
    checkCudaErrors(cudaMalloc((void***)&d_DoseArray, maxNumBeamletsPerBatch*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_FluenceArray, h_FluenceArray.data(),
        maxNumBeamletsPerBatch*sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_DensityArray, h_DensityArray.data(),
        maxNumBeamletsPerBatch*sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_TermaArray, h_TermaArray.data(),
        maxNumBeamletsPerBatch*sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_DoseArray, h_DoseArray.data(),
        maxNumBeamletsPerBatch*sizeof(float*), cudaMemcpyHostToDevice));

    // pack array
    int2 packDim;
    packDim.x = (int)ceilf(sqrtf((float)maxNumBeamletsPerBatch));
    packDim.y = (int)ceilf((float)maxNumBeamletsPerBatch / packDim.x);
    int3 packArrayDim {
        packDim.x * subFluenceDim,
        packDim.y * subFluenceDim,
        maxBeamletLength };
    size_t packArraySize = packArrayDim.x * packArrayDim.y * packArrayDim.z;
    float* packArray = nullptr;
    checkCudaErrors(cudaMalloc((void**)&packArray, packArraySize*sizeof(float)));

    // for sampling
    dim3 samplingBlockSize{8, 8, 8};
    dim3 samplingGridSize {
        (densityDim.x - 1 + samplingBlockSize.x) / samplingBlockSize.x,
        (densityDim.y - 1 + samplingBlockSize.y) / samplingBlockSize.y,
        (densityDim.z - 1 + samplingBlockSize.z) / samplingBlockSize.z
    };
    size_t preSamplingGridSize = samplingGridSize.x * samplingGridSize.y
        * samplingGridSize.z * maxNumBeamletsPerBatch;
    bool* d_preSamplingArray = nullptr;
    checkCudaErrors(cudaMalloc((void**)(&d_preSamplingArray), preSamplingGridSize*sizeof(bool)));

    // the array to store the long_dim of each beamlets
    int* d_beamletLongArray = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beamletLongArray, maxNumBeamletsPerBatch*sizeof(int)));

    // prepare texture components
    cudaArray* DoseBEV_Arr;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize = make_cudaExtent(packArrayDim.x, packArrayDim.y, packArrayDim.z);
    cudaMalloc3DArray(&DoseBEV_Arr, &channelDesc, volumeSize);

    for (int i=0; i<iterations; i++) {
        // firstly, prepare beamlet information
        int localNumBeamlets = 0;
        int beam_bundle_idx_begin = i * concurrency;
        int beam_bundle_idx_end = (i + 1) * concurrency;
        beam_bundle_idx_end = min(beam_bundle_idx_end, (int)(beam_bundles.size()));
        for (int j=beam_bundle_idx_begin; j<beam_bundle_idx_end; j++)
            localNumBeamlets += beam_bundles[j].beams_h.size();
        
        std::vector<fd::d_BEAM_d> h_BeamletsBuffer;
        std::vector<int> h_beamletLongArray;
        h_BeamletsBuffer.reserve(localNumBeamlets);
        h_beamletLongArray.reserve(localNumBeamlets);
        size_t pitch = subFluenceDim * subFluenceDim * sizeof(float);
        for (int j=beam_bundle_idx_begin; j<beam_bundle_idx_end; j++) {
            for (int k=0; k<beam_bundles[j].beams_h.size(); k++) {
                const fd::BEAM_h& h_source = beam_bundles[j].beams_h[k];
                h_BeamletsBuffer.push_back(fd::d_BEAM_d(h_source, pitch, pitch));
                h_beamletLongArray.push_back(h_source.long_dim);
            }
        }
        checkCudaErrors(cudaMemcpyAsync(d_BeamletsBuffer, h_BeamletsBuffer.data(),
            localNumBeamlets*sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_beamletLongArray, h_beamletLongArray.data(),
            localNumBeamlets*sizeof(int), cudaMemcpyHostToDevice, stream));
        
        // Calculate Terma
        size_t fmap_npixels = subFluenceDim * subFluenceDim;
        fd::TermaComputeCollective(
            fmap_npixels,
            localNumBeamlets,
            d_BeamletsBuffer,
            d_FluenceArray,
            d_TermaArray,
            d_DensityArray,
            density_d,
            spectrum_h,
            stream
        );
        
        // Calculate Dose
        fd::DoseComputeCollective(
            fmap_npixels,
            localNumBeamlets,
            d_BeamletsBuffer,
            d_TermaArray,
            d_DensityArray,
            d_DoseArray,
            kernel_h.nTheta,
            kernel_h.nPhi,
            stream);
        
        // Copy the result to packed array
        BEV2PVCSInterp(
            d_denseDoseMat,
            denseDoseMatSize,
            d_BeamletsBuffer,
            localNumBeamlets,
            density_d,
            d_DoseArray,
            pitch / sizeof(float),
            d_preSamplingArray,
            preSamplingGridSize,
            packArray,
            packDim,
            make_int2(subFluenceDim, subFluenceDim),
            packArrayDim,
            &DoseBEV_Arr,
            d_beamletLongArray,
            extent,
            stream
        );
    }

    // clean up
    checkCudaErrors(cudaFreeArray(DoseBEV_Arr));
    checkCudaErrors(cudaFree(d_beamletLongArray));
    checkCudaErrors(cudaFree(d_preSamplingArray));
    checkCudaErrors(cudaFree(packArray));

    checkCudaErrors(cudaFree(d_DoseArray));
    checkCudaErrors(cudaFree(d_TermaArray));
    checkCudaErrors(cudaFree(d_DensityArray));
    checkCudaErrors(cudaFree(d_FluenceArray));

    checkCudaErrors(cudaFree(d_DoseBEVBuffer));
    checkCudaErrors(cudaFree(d_TeramBEVBuffer));
    checkCudaErrors(cudaFree(d_DensityBEVBuffer));
    checkCudaErrors(cudaFree(d_FluenceBuffer));
    checkCudaErrors(cudaFree(d_BeamletsBuffer));

    checkCudaErrors(cudaFree(d_denseDoseMat));
    return 0;
}