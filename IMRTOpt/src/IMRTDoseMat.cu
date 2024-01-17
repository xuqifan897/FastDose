#include <fstream>
#include <iomanip>
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

    // firstly, calculate Terma and Dose
    int concurrency = getarg<int>("concurrency");
    float extent = getarg<float>("extent");
    int iterations = (beam_bundles.size() + concurrency - 1) / concurrency;
    SparseMatArray.resize(iterations);

    cudaEventRecord(globalStart);
    for (int i=0; i<iterations; i++) {
        cudaEventRecord(start);

        std::vector<fd::BEAM_d> beamlets;
        int beam_bundle_idx_begin = i * concurrency;
        int beam_bundle_idx_end = (i + 1) * concurrency;
        beam_bundle_idx_end = min(beam_bundle_idx_end, (int)(beam_bundles.size()));

        int nBeamlets = 0;
        for (int j=beam_bundle_idx_begin; j<beam_bundle_idx_end; j++) {
            nBeamlets += beam_bundles[j].beams_h.size();
        }
        beamlets.resize(nBeamlets);
        int count = 0;
        for (int j=beam_bundle_idx_begin; j<beam_bundle_idx_end; j++) {
            auto& current = beam_bundles[j];
            for (int k=0; k<current.beams_h.size(); k++) {
                fd::beam_h2d(current.beams_h[k], beamlets[count]);
                count ++;
            }
        }

        // preparation
        std::vector<fd::d_BEAM_d> h_beams;
        h_beams.reserve(nBeamlets);
        for (int j=0; j<nBeamlets; j++) {
            h_beams.push_back(fd::d_BEAM_d(beamlets[j]));
        }
        fd::d_BEAM_d* d_beams = nullptr;
        checkCudaErrors(cudaMalloc((void**)(&d_beams), nBeamlets*sizeof(fd::d_BEAM_d)));
        checkCudaErrors(cudaMemcpy(d_beams, h_beams.data(),
            nBeamlets*sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice));

        // allocate fluence array
        std::vector<float*> h_fluence_array(nBeamlets, nullptr);
        for (int j=0; j<nBeamlets; j++)
            h_fluence_array[j] = beamlets[j].fluence;
        float** d_fluence_array = nullptr;
        checkCudaErrors(cudaMalloc((void***)(&d_fluence_array), nBeamlets*sizeof(float*)));
        checkCudaErrors(cudaMemcpy(d_fluence_array, h_fluence_array.data(),
            nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

        // allocate Terma_array
        std::vector<float*> h_TermaBEV_array(nBeamlets, nullptr);
        for (int j=0; j<nBeamlets; j++)
            h_TermaBEV_array[j] = beamlets[j].TermaBEV;
        float** d_TermaBEV_array = nullptr;
        checkCudaErrors(cudaMalloc((void***)(&d_TermaBEV_array), nBeamlets*sizeof(float*)));
        checkCudaErrors(cudaMemcpy(d_TermaBEV_array, h_TermaBEV_array.data(),
            nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

        // allocate DenseBEV_array
        std::vector<float*> h_DensityBEV_array(nBeamlets, nullptr);
        for (int j=0; j<nBeamlets; j++)
            h_DensityBEV_array[j] = beamlets[j].DensityBEV;
        float** d_DensityBEV_array = nullptr;
        checkCudaErrors(cudaMalloc((void***)(&d_DensityBEV_array), nBeamlets*sizeof(float*)));
        checkCudaErrors(cudaMemcpy(d_DensityBEV_array, h_DensityBEV_array.data(),
            nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));
        
        // allocate DoseBEV_array
        std::vector<float*> h_DoseBEV_array(nBeamlets, nullptr);
        for (int j=0; j<nBeamlets; j++)
            h_DoseBEV_array[j] = beamlets[j].DoseBEV;
        float** d_DoseBEV_array = nullptr;
        checkCudaErrors(cudaMalloc((void***)(&d_DoseBEV_array), nBeamlets*sizeof(float*)));
        checkCudaErrors(cudaMemcpy(d_DoseBEV_array, h_DoseBEV_array.data(),
            nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

        size_t fmap_npixels = beamlets[0].fmap_size.x * beamlets[0].fmap_size.y;

        fd::TermaComputeCollective(
            fmap_npixels,
            nBeamlets,
            d_beams,
            d_fluence_array,
            d_TermaBEV_array,
            d_DensityBEV_array,
            density_d,
            spectrum_h,
            stream
        );

        fd::DoseComputeCollective(
            fmap_npixels,
            nBeamlets,
            d_beams,
            d_TermaBEV_array,
            d_DensityBEV_array,
            d_DoseBEV_array,
            kernel_h.nTheta,
            kernel_h.nPhi,
            stream
        );

        // clean up
        checkCudaErrors(cudaFree(d_DoseBEV_array));
        checkCudaErrors(cudaFree(d_DensityBEV_array));
        checkCudaErrors(cudaFree(d_TermaBEV_array));
        checkCudaErrors(cudaFree(d_fluence_array));
        
        float* d_dense_dose;
        size_t denseDoseSize = density_d.VolumeDim.x
            * density_d.VolumeDim.y * density_d.VolumeDim.z
            * beamlets.size();
        checkCudaErrors(cudaMalloc((void**)&d_dense_dose, denseDoseSize*sizeof(float)));
        checkCudaErrors(cudaMemset(d_dense_dose, 0, denseDoseSize*sizeof(float)));
        BEV2PVCSInterp(&d_dense_dose, beamlets, d_beams, density_d, 5, extent, stream);
        checkCudaErrors(cudaFree(d_beams));

        MatCSR& currentMat = SparseMatArray[i];
        size_t nColumns = density_d.VolumeDim.x * density_d.VolumeDim.y * density_d.VolumeDim.z;
        currentMat.dense2sparse(d_dense_dose, beamlets.size(), nColumns, nColumns);
        checkCudaErrors(cudaFree(d_dense_dose));

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        size_t free_bytes, total_bytes;
        checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
        float free_GB = (float)free_bytes * (1.0f / 1024) * (1.0f / 1024) * (1.0f / 1024);
        float total_GB = (float)total_bytes * (1.0f / 1024) * (1.0f / 1024) * (1.0f / 1024);
        std::cout << std::fixed << std::setprecision(2)
            << "Beams " << beam_bundle_idx_begin + 1 << " ~ " << beam_bundle_idx_end << " / " << beam_bundles.size()
            << ", number of beamlets: " << nBeamlets << ", time elapsed: " << milliseconds
            << "[ms], free memory: " << free_GB << " / " << total_GB << " GB" << std::endl;
    }
    cudaEventRecord(globalStop);
    cudaEventSynchronize(globalStop);
    cudaEventElapsedTime(&milliseconds, globalStart, globalStop);
    std::cout << std::endl << "Concurrency: " << concurrency
        << ", total dose calculation time: "
        << milliseconds / 1000 << "s." << std::endl;
    return 0;
}