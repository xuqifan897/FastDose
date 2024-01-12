#include "IMRTDoseMat.cuh"
#include "IMRTArgs.h"

namespace fd = fastdose;

bool IMRT::MatCSR::dense2sparse(
    float* d_dense, int num_rows, int num_cols, int ld
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
    
    // execute Sparse to Dense conversion
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
    
    // execute Sparse to Dense conversion
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

    // firstly, calculate Terma and Dose
    int concurrency = getarg<int>("concurrency");
    float extent = getarg<float>("extent");
    int iterations = (beam_bundles.size() + concurrency - 1) / concurrency;
    for (int i=0; i<iterations; i++) {
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

        #if TIMING
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        #endif

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

        #if TIMING
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Terma time elapsed: " << milliseconds << " [ms]" << std::endl;
        #endif

        #if TIMING
            cudaEventRecord(start);
        #endif

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

        #if TIMING
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Dose time elapsed: " << milliseconds << " [ms]" << std::endl;
        #endif

        // clean up
        checkCudaErrors(cudaFree(d_DoseBEV_array));
        checkCudaErrors(cudaFree(d_DensityBEV_array));
        checkCudaErrors(cudaFree(d_TermaBEV_array));
        checkCudaErrors(cudaFree(d_fluence_array));
        checkCudaErrors(cudaFree(d_beams));

        #if false
            size_t freeBytes, totalBytes;
            cudaMemGetInfo(&freeBytes, &totalBytes);
            std::cout << "Free memory: " << (float)freeBytes / (1<<30) << "GB \n"
                "Total memory: " << (float)totalBytes / (1<<30) << "GB." << std::endl;
        #endif


        #if false
            #if TIMING
                cudaEventRecord(start);
            #endif
            // convert BEV dose to PVCS dose
            float* DosePVCSCollective = nullptr;
            size_t singleMatrixSize = density_d.VolumeDim.x *
                density_d.VolumeDim.y * density_d.VolumeDim.z;
            size_t totalMatrixSize = singleMatrixSize * nBeamlets;
            size_t totalMemorySize = totalMatrixSize * sizeof(float);
            std::cout << "Total matrix size: " << (float)totalMemorySize / (1<<30) << "GB" << std::endl;
            checkCudaErrors(cudaMalloc((void**)&DosePVCSCollective, totalMatrixSize*sizeof(float)));
            checkCudaErrors(cudaMemset(DosePVCSCollective, 0, totalMatrixSize*sizeof(float)));
            for (int j=0; j<nBeamlets; j++) {
                fd::BEAM_d& current_beamlet = beamlets[j];
                
                if (current_beamlet.fmap_size.x * current_beamlet.fmap_size.y != 
                    current_beamlet.DoseBEV_pitch / sizeof(float)
                ) {
                    std::cout << "The fluence map size " << current_beamlet.fmap_size 
                        << " does not match the pitch size: " << current_beamlet.DoseBEV_pitch / sizeof(float)
                        << " Please adjust the fluence map dimension" << std::endl;
                    return 1;
                }
                cudaArray* DoseBEV_Arr;
                cudaTextureObject_t DoseBEV_Tex;
                cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
                cudaExtent volumeSize = make_cudaExtent(current_beamlet.fmap_size.x,
                    current_beamlet.fmap_size.y, current_beamlet.long_dim);
                cudaMalloc3DArray(&DoseBEV_Arr, &channelDesc, volumeSize);
                // copy to cudaArray
                cudaMemcpy3DParms copyParams = {0};
                copyParams.srcPtr = make_cudaPitchedPtr(
                    (void*)(current_beamlet.DoseBEV),
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
                memset(&texDescr, 0, sizeof(texDescr));
                texDescr.normalizedCoords = false;
                texDescr.addressMode[0] = cudaAddressModeBorder;
                texDescr.addressMode[1] = cudaAddressModeBorder;
                texDescr.addressMode[2] = cudaAddressModeBorder;
                checkCudaErrors(cudaCreateTextureObject(&DoseBEV_Tex, &texRes, &texDescr, nullptr));

                // prepare DosePVCS_Arr
                cudaPitchedPtr DosePVCS_Arr;
                DosePVCS_Arr.ptr = DosePVCSCollective + i * singleMatrixSize;
                DosePVCS_Arr.pitch = density_d.VolumeDim.x * sizeof(float);

                fd::BEV2PVCS_SuperSampling(current_beamlet,
                    density_d, DosePVCS_Arr, DoseBEV_Tex, 5, extent);
                
                // clean up
                checkCudaErrors(cudaDestroyTextureObject(DoseBEV_Tex));
                checkCudaErrors(cudaFreeArray(DoseBEV_Arr));
            }
            #if TIMING
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&milliseconds, start, stop);
                std::cout << "Interpolation time elapsed: " << milliseconds << " [ms]" << std::endl;
            #endif
        #endif

        // for debug purposes
        break;
    }
    return 0;
}