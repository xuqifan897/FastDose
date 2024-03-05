#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <random>
#include <limits>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "IMRTDoseMat.cuh"
#include "IMRTArgs.h"
#include "IMRTgeom.cuh"
#include "IMRTOptBench.cuh"

namespace fd = fastdose;

bool IMRT::DoseMatConstruction(
    std::vector<BeamBundle>& beam_bundles,
    fd::DENSITY_d& density_d,
    fd::SPECTRUM_h& spectrum_h,
    fd::KERNEL_h& kernel_h,
    MatCSREnsemble** matEns,
    cudaStream_t stream
) {
    cudaEvent_t start, stop, globalStart, globalStop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&globalStart));
    checkCudaErrors(cudaEventCreate(&globalStop));
    float milliseconds;

    int concurrency = getarg<int>("concurrency");
    float extent = getarg<float>("extent");
    int iterations = (beam_bundles.size() + concurrency - 1) / concurrency;
    std::vector<size_t> numRowsPerMat(iterations, 0);

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
        numRowsPerMat[i] = localNumBeamlets;
    }
    std::cout << std::endl << "Maximum number of beamlets per batch: " << maxNumBeamletsPerBatch
        << ", maximum beamlet length: " << maxBeamletLength << std::endl << std::endl;

    // prepare the sparse dose matrices
    uint3 densityDim = density_d.VolumeDim;
    size_t numDensityVoxels = densityDim.x * densityDim.y * densityDim.z;
    size_t EstNonZeroElementsPerMat = getarg<size_t>("EstNonZeroElementsPerMat");
    size_t estBufferSize = EstNonZeroElementsPerMat * beam_bundles.size();
    *matEns = new MatCSREnsemble(numRowsPerMat, numDensityVoxels, estBufferSize);

    // allocate working buffers
    // for safty check
    unsigned long long denseDoseMatSize_ = maxNumBeamletsPerBatch * densityDim.x * densityDim.y * densityDim.z;
    if (denseDoseMatSize_ > std::numeric_limits<uint>::max()) {
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

    // create a stream for memory reset, so that it can overlap computation and memory operations
    cudaStream_t memsetStream;
    checkCudaErrors(cudaStreamCreate(&memsetStream));

    cudaEventRecord(globalStart);
    // for debug purposes
    for (int i=0; i<iterations; i++) {
        cudaEventRecord(start);

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
            stream,
            memsetStream
        );

        (**matEns).addMat(d_denseDoseMat, localNumBeamlets, numDensityVoxels);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Iteration: " << i << ", beam bundle indices: "
            << beam_bundle_idx_begin << " ~ " << beam_bundle_idx_end - 1
            << " / " << beam_bundles.size() << ", time elapsed: "
            << milliseconds << " [ms]" << std::endl;
    }
    checkCudaErrors(cudaStreamDestroy(memsetStream));

    cudaEventRecord(globalStop);
    cudaEventSynchronize(globalStop);
    cudaEventElapsedTime(&milliseconds, globalStart, globalStop);
    std::cout << "Dose calculation time: " << milliseconds * 0.001f << " s" << std::endl;

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

    checkCudaErrors(cudaEventDestroy(globalStop));
    checkCudaErrors(cudaEventDestroy(globalStart));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));

    return 0;
}

#define SLICING_ROW_DEBUG false
bool IMRT::MatCSR64::slicing_row(const std::vector<size_t>& rowIndices) {
    std::vector<size_t> h_csr_offsets(this->numRows+1, 0);
    checkCudaErrors(cudaMemcpy(h_csr_offsets.data(), this->d_csr_offsets,
        (this->numRows+1)*sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t numRows_new = rowIndices.size();

    // in the format <new_idx, old_idx, size>
    std::vector<std::tuple<size_t, size_t, size_t>> mapping(numRows_new);
    std::vector<size_t> h_csr_offsets_new(numRows_new+1, 0);
    for (size_t i=0; i<numRows_new; i++) {
        size_t row_idx = rowIndices[i];
        size_t numElementsThisRow = h_csr_offsets[row_idx+1] - h_csr_offsets[row_idx];
        h_csr_offsets_new[i+1] = h_csr_offsets_new[i] + numElementsThisRow;

        size_t source_address = h_csr_offsets[row_idx];
        size_t target_address = h_csr_offsets_new[i];
        mapping[i] = std::make_tuple(target_address, source_address, numElementsThisRow);
    }
    checkCudaErrors(cudaMemcpyAsync(this->d_csr_offsets, h_csr_offsets_new.data(),
        (numRows_new + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    
    // coalease consecutive rows
    size_t target_idx = std::get<0>(mapping[0]);
    size_t source_idx = std::get<1>(mapping[0]);
    size_t size = std::get<2>(mapping[0]);
    std::vector<size_t> source_offsets_h;
    std::vector<size_t> target_offsets_h;
    std::vector<size_t> block_sizes_h;
    #if SLICING_ROW_DEBUG
        std::vector<size_t> rows{rowIndices[0]};
    #endif
    for (size_t i=1; i<numRows_new; i++) {
        size_t row_prev = rowIndices[i-1];
        size_t row_now = rowIndices[i];
        const auto& mapping_entry = mapping[i];
        if (row_now == row_prev + 1) {
            size += std::get<2>(mapping_entry);
            #if SLICING_ROW_DEBUG
                rows.push_back(row_now);
            #endif
            continue;
        } else {
            source_offsets_h.push_back(source_idx);
            target_offsets_h.push_back(target_idx);
            block_sizes_h.push_back(size);
            #if SLICING_ROW_DEBUG
                std::cout << "Consecutive rows: ";
                for (size_t j=0; j<rows.size(); j++)
                    std::cout << rows[j] << "  ";
                std::cout << "\n(target_idx, source_idx, size): ("
                    << target_idx << ", " << source_idx << ", " << size << ")\n\n";
                rows.clear();

                target_idx = std::get<0>(mapping_entry);
                source_idx = std::get<1>(mapping_entry);
                size = std::get<2>(mapping_entry);
                rows.push_back(row_now);
            #endif
        }
    }
    source_offsets_h.push_back(source_idx);
    target_offsets_h.push_back(target_idx);
    block_sizes_h.push_back(size);
    #if SLICING_ROW_DEBUG
        std::cout << "Consecutive rows: ";
        for (size_t j=0; j<rows.size(); j++)
            std::cout << rows[j] << "  ";
        std::cout << "\n(target_idx, source_idx, size): ("
                    << target_idx << ", " << source_idx << ", " << size << ")\n\n";
        rows.clear();
    #endif
    
    size_t* source_offsets_d;
    size_t* target_offsets_d;
    size_t* block_sizes_d;
    checkCudaErrors(cudaMalloc((void**)&source_offsets_d, source_offsets_h.size()*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&target_offsets_d, target_offsets_h.size()*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&block_sizes_d, block_sizes_h.size()*sizeof(size_t)));

    checkCudaErrors(cudaMemcpy(source_offsets_d, source_offsets_h.data(),
        source_offsets_h.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(target_offsets_d, target_offsets_h.data(),
        target_offsets_h.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(block_sizes_d, block_sizes_h.data(),
        block_sizes_h.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    
    memcpy_kernel(this->d_csr_columns, source_offsets_d,
        this->d_csr_columns, target_offsets_d,
        block_sizes_d, source_offsets_h.size());
    memcpy_kernel(this->d_csr_values, source_offsets_d,
        this->d_csr_values, target_offsets_d,
        block_sizes_d, source_offsets_h.size());
    
    this->numRows = numRows_new;
    this->nnz = h_csr_offsets_new.back();
    checkCusparse(cusparseCreateCsr(
        &this->matA, this->numRows, this->numCols, this->nnz,
        this->d_csr_offsets, this->d_csr_columns, this->d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    checkCudaErrors(cudaFree(source_offsets_d));
    checkCudaErrors(cudaFree(target_offsets_d));
    checkCudaErrors(cudaFree(block_sizes_d));
    return 0;
}
#undef SLICING_ROW_DEBUG


bool IMRT::memcpy_kernel(size_t* source_ptr, size_t* source_offsets,
    size_t* target_ptr, size_t* target_offsets,
    size_t* block_sizes, size_t num_blocks) {
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(1024, 1, 1);
    size_t sharedSize = blockSize.x * sizeof(size_t);
    d_memcpy_kernel<<<gridSize, blockSize, sharedSize, 0>>>(
        source_ptr, source_offsets, target_ptr, target_offsets,
        block_sizes, num_blocks);
    return 0;
}


bool IMRT::memcpy_kernel(float* source_ptr, size_t* source_offsets,
    float* target_ptr, size_t* target_offsets,
    size_t* block_sizes, size_t num_blocks) {
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(1024, 1, 1);
    size_t sharedSize = blockSize.x * sizeof(float);
    d_memcpy_kernel<<<gridSize, blockSize, sharedSize, 0>>>(
        source_ptr, source_offsets, target_ptr, target_offsets,
        block_sizes, num_blocks);
    return 0;
}


__global__ void IMRT::d_memcpy_kernel(
    size_t* source_ptr, size_t* source_offsets,
    size_t* target_ptr, size_t* target_offsets,
    size_t* block_sizes, size_t num_blocks
) {
    extern __shared__ size_t sharedData_size_t[];
    size_t idx = threadIdx.x;
    size_t batch_size = blockDim.x;
    for (size_t i=0; i<num_blocks; i++) {
        size_t current_source_offset = source_offsets[i];
        size_t current_target_offset = target_offsets[i];
        size_t current_block_size = block_sizes[i];
        size_t num_batches = (current_block_size + batch_size - 1) / batch_size;
        for (size_t j=0; j<num_batches; j++) {
            size_t j_times_batch_size = j * batch_size;
            size_t source_idx_start = current_source_offset + j_times_batch_size;
            size_t target_idx_start = current_target_offset + j_times_batch_size;
            size_t numActiveThreads = min(batch_size, current_block_size - j_times_batch_size);
            if (idx < numActiveThreads)
                sharedData_size_t[idx] = source_ptr[source_idx_start + idx];
            __syncthreads();
            if (idx < numActiveThreads)
                target_ptr[target_idx_start + idx] = sharedData_size_t[idx];
            __syncthreads();
        }
        if (idx == 0) {
            printf("progress: %lu / %lu\n", i, num_blocks);
        }
    }
}


__global__ void IMRT::d_memcpy_kernel(
    float* source_ptr, size_t* source_offsets,
    float* target_ptr, size_t* target_offsets,
    size_t* block_sizes, size_t num_blocks
) {
    extern __shared__ float sharedData_float[];
    size_t idx = threadIdx.x;
    size_t batch_size = blockDim.x;
    for (size_t i=0; i<num_blocks; i++) {
        size_t current_source_offset = source_offsets[i];
        size_t current_target_offset = target_offsets[i];
        size_t current_block_size = block_sizes[i];
        size_t num_batches = (current_block_size + batch_size - 1) / batch_size;
        for (size_t j=0; j<num_batches; j++) {
            size_t j_times_batch_size = j * batch_size;
            size_t source_idx_start = current_source_offset + j_times_batch_size;
            size_t target_idx_start = current_target_offset + j_times_batch_size;
            size_t numActiveThreads = min(batch_size, current_block_size - j_times_batch_size);
            if (idx < numActiveThreads)
                sharedData_float[idx] = source_ptr[source_idx_start + idx];
            __syncthreads();
            if (idx < numActiveThreads)
                target_ptr[target_idx_start + idx] = sharedData_float[idx];
            __syncthreads();
        }
    }
}


bool IMRT::benchmark_slicing_row() {
    size_t numRows = 100;
    size_t numCols = 100;
    std::srand(10086);
    MatCSR_Eigen matrix_Eigen;
    randomize_MatCSR_Eigen(matrix_Eigen, numRows, numCols);

    size_t numRows_selected = 50;
    std::vector<size_t> active_rows(numRows, 0);
    for (size_t i=0; i<numRows; i++)
        active_rows[i] = i;
    std::mt19937 rng(10086);
    std::shuffle(active_rows.begin(), active_rows.end(), rng);
    active_rows.resize(numRows_selected);
    std::sort(active_rows.begin(), active_rows.end());
    
    std::cout << "Active rows:\n";
    for (size_t i=0; i<numRows_selected; i++)
        std::cout << active_rows[i] << "  ";
    std::cout << std::endl;

    // construct row-selection matrix
    MatCSR_Eigen row_selection;
    EigenIdxType* row_selection_offsets = (EigenIdxType*)malloc(
        (numRows_selected+1)*sizeof(EigenIdxType));
    EigenIdxType* row_selection_columns = new EigenIdxType[numRows_selected];
    float* row_selection_values = new float[numRows_selected];
    row_selection_offsets[0] = 0;
    for (size_t i=0; i<numRows_selected; i++) {
        row_selection_offsets[i+1] = i + 1;
        row_selection_columns[i] = active_rows[i];
        row_selection_values[i] = 1.0f;
    }
    row_selection.customInit(numRows_selected, numCols, numRows_selected,
        row_selection_offsets, row_selection_columns, row_selection_values);
    MatCSR_Eigen result_Eigen = row_selection * matrix_Eigen;
    
    #if false
    // for debug purposes
        std::cout << "Original matrix:\n" << matrix_Eigen << "\n\nRow selection matrix:\n"
            << row_selection << "\n\nResult matrix:\n" << result_Eigen << std::endl;
    #endif

    MatCSR64 matrix_cu;
    Eigen2Cusparse(matrix_Eigen, matrix_cu);
    matrix_cu.slicing_row(active_rows);

    std::vector<size_t> offsets_test(numRows_selected + 1);
    std::vector<size_t> columns_test(matrix_cu.nnz);
    std::vector<float> values_test(matrix_cu.nnz);
    checkCudaErrors(cudaMemcpy(offsets_test.data(), matrix_cu.d_csr_offsets,
        (numRows_selected + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(columns_test.data(), matrix_cu.d_csr_columns,
        matrix_cu.nnz * sizeof(size_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(values_test.data(), matrix_cu.d_csr_values,
        matrix_cu.nnz * sizeof(float), cudaMemcpyDeviceToHost));

    EigenIdxType* offsets_ref = *result_Eigen.getOffset();
    const EigenIdxType* columns_ref = result_Eigen.getIndices();
    const float* values_ref = result_Eigen.getValues();

    for (size_t i=0; i<numRows_selected+1; i++) {
        if (offsets_test[i] != offsets_ref[i]) {
            std::cerr << "Offset mismatch at index " << i << std::endl;
            return 1;
        }
    }

    for (size_t i=0; i<matrix_cu.nnz; i++) {
        if (columns_test[i] != columns_ref[i]) {
            std::cerr << "Column mismatch at index " << i << std::endl;
            return 1;
        }
        if (std::abs(values_test[i] - values_ref[i]) > eps_fastdose) {
            std::cerr << "Value mismatch at index " << i << std::endl;
            return 1;
        }
    }
    std::cout << "Function slicing_row passed the test!" << std::endl;
    return 0;
}