#include <cusparse.h>
#include "helper_cuda.h"

#include "PlanOptmTestCase.cuh"

namespace fd = fastdose;

bool PlanOptm::beamBundleTestCaseSparse(
    std::vector<BeamBundle> beam_bundles,
    fastdose::DENSITY_h& density_h,
    fastdose::DENSITY_d& density_d,
    fastdose::SPECTRUM_h& spectrum_h,
    fastdose::KERNEL_h& kernel_h,
    cudaStream_t stream
) {
    BeamBundle& first_beam_bundle = beam_bundles[0];
    int nBeamlets = first_beam_bundle.fluenceDim.x * first_beam_bundle.fluenceDim.y;
    first_beam_bundle.beams_d.resize(nBeamlets);
    for (int i=0; i<nBeamlets; i++) {
        fd::beam_h2d(first_beam_bundle.beams_h[i], first_beam_bundle.beams_d[i]);
    }

    // preparation
    std::vector<fd::d_BEAM_d> h_beams;
    h_beams.reserve(nBeamlets);
    for (int i=0; i<nBeamlets; i++)
        h_beams.push_back(fd::d_BEAM_d(first_beam_bundle.beams_d[i]));
    fd::d_BEAM_d* d_beams = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beams, nBeamlets*sizeof(fd::d_BEAM_d)));
    checkCudaErrors(cudaMemcpy(d_beams, h_beams.data(),
        nBeamlets*sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice));
    
    // allocate fluence array
    std::vector<float*> h_fluence_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_fluence_array[i] = first_beam_bundle.beams_d[i].fluence;
    float** fluence_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&fluence_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(fluence_array, h_fluence_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));
    
    // allocate Terma_array
    std::vector<float*> h_TermaBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_TermaBEV_array[i] = first_beam_bundle.beams_d[i].TermaBEV;
    float** TermaBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&TermaBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(TermaBEV_array, h_TermaBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));
    
    // allocate DenseBEV_array
    std::vector<float*> h_DensityBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_DensityBEV_array[i] = first_beam_bundle.beams_d[i].DensityBEV;
    float** DensityBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&DensityBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(DensityBEV_array, h_DensityBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    // allocate DoseBEV_array
    std::vector<float*> h_DoseBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_DoseBEV_array[i] = first_beam_bundle.beams_d[i].DoseBEV;
    float** DoseBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&DoseBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(DoseBEV_array, h_DoseBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));
    
    size_t fmap_npixels = first_beam_bundle.subFluenceDim.x *
        first_beam_bundle.subFluenceDim.y;

    // for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    fd::TermaComputeCollective(
        fmap_npixels,
        nBeamlets,
        d_beams,
        fluence_array,
        TermaBEV_array,
        DensityBEV_array,
        density_d,
        spectrum_h,
        stream
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Terma time elapsed: " << milliseconds << " [ms]" << std::endl;


    cudaEventRecord(start);

    fd::DoseComputeCollective(
        fmap_npixels,
        nBeamlets,
        d_beams,
        TermaBEV_array,
        DensityBEV_array,
        DoseBEV_array,
        kernel_h.nTheta,
        kernel_h.nPhi,
        stream
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Dose time elapsed: " << milliseconds << " [ms]" << std::endl;

    // destination array
    // the density matrices are used to construct the sparse matrices.
    float* DosePVCSCollective = nullptr;
    size_t pitch_in_elements = ((density_d.VolumeDim.x + pitchModule - 1) 
        / pitchModule) * pitchModule;
    size_t singleMatrixSize = pitch_in_elements * density_d.VolumeDim.y * density_d.VolumeDim.z;
    size_t totalMatrixSize = singleMatrixSize * nBeamlets;
    checkCudaErrors(cudaMalloc((void**)&DosePVCSCollective, totalMatrixSize*sizeof(float)));
    checkCudaErrors(cudaMemset(DosePVCSCollective, 0.f, totalMatrixSize*sizeof(float)));

    for (int i=0; i<nBeamlets; i++) {
        fd::BEAM_d& current_beamlet = first_beam_bundle.beams_d[i];

        // for safety check
        if (current_beamlet.fmap_size.x * current_beamlet.fmap_size.y
            != current_beamlet.DoseBEV_pitch / sizeof(float)) {
            std::cerr << "The DoseBEV pitch value " << current_beamlet.DoseBEV_pitch / sizeof(float)
                << " does not equal to the size of the fluence: " << current_beamlet.fmap_size
                << std::endl;
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
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = false;
        texDescr.addressMode[0] = cudaAddressModeBorder;
        texDescr.addressMode[1] = cudaAddressModeBorder;
        texDescr.addressMode[2] = cudaAddressModeBorder;
        checkCudaErrors(cudaCreateTextureObject(&DoseBEV_Tex, &texRes, &texDescr, NULL));

        // prepare DosePVCS_Arr
        cudaPitchedPtr DosePVCS_Arr;
        DosePVCS_Arr.ptr = DosePVCSCollective + i * singleMatrixSize;
        DosePVCS_Arr.pitch = pitch_in_elements * sizeof(float);

        fd::BEV2PVCS_SuperSampling(current_beamlet,
            density_d, DosePVCS_Arr, DoseBEV_Tex, 5, 2.0f, stream);

        // clean up
        checkCudaErrors(cudaDestroyTextureObject(DoseBEV_Tex));
        checkCudaErrors(cudaFreeArray(DoseBEV_Arr));
    }

    #if false
        // non-transpose
        // sparsification
        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matSparse;
        cusparseDnMatDescr_t matDense;
        void* dBuffer = nullptr;
        size_t bufferSize = 0;

        // Device memory management
        int* d_csr_offsets;
        int* d_csr_columns;
        float* d_csr_values;

        CHECK_CUDA(cudaMalloc((void**)&d_csr_offsets,(nBeamlets+1)*sizeof(int)))
        CHECK_CUSPARSE(cusparseCreate(&handle))
        // create dense matrix:
        CHECK_CUSPARSE(cusparseCreateDnMat(
            &matDense, nBeamlets, singleMatrixSize, singleMatrixSize,
            DosePVCSCollective, CUDA_R_32F, CUSPARSE_ORDER_ROW))
        // create sparse matrix;
        CHECK_CUSPARSE(cusparseCreateCsr(
            &matSparse, nBeamlets, singleMatrixSize, 0,
            d_csr_offsets, nullptr, nullptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
        // allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
            handle, matDense, matSparse,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            &bufferSize));
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
        // execute dense to sparse conversion
        CHECK_CUSPARSE(cusparseDenseToSparse_analysis(
            handle, matDense, matSparse,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            dBuffer))
        int64_t num_rows_tmp, num_cols_tmp, nnz;
        CHECK_CUSPARSE(cusparseSpMatGetSize(
            matSparse, &num_rows_tmp, &num_cols_tmp, &nnz))
        // allocate CSF column indices and values
        CHECK_CUDA(cudaMalloc((void**)&d_csr_columns, nnz*sizeof(int)))
        CHECK_CUDA(cudaMalloc((void**)&d_csr_values, nnz*sizeof(float)));
        // reset offsets, column indices, and value pointers
        CHECK_CUSPARSE(cusparseCsrSetPointers(
            matSparse, d_csr_offsets, d_csr_columns, d_csr_values))
        CHECK_CUSPARSE(cusparseDenseToSparse_convert(
            handle, matDense, matSparse,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            dBuffer))
        CHECK_CUDA(cudaFree(dBuffer))
        std::cout << "Number of non-zero elements: " << nnz << std::endl;


        #if false
            // sparse to dense test
            float* d_dense_ref;
            checkCudaErrors(cudaMalloc((void**)&d_dense_ref,
                totalMatrixSize*sizeof(float)));
            cusparseDnMatDescr_t matDenseRef;
            CHECK_CUSPARSE(cusparseCreateDnMat(
                &matDenseRef, nBeamlets, singleMatrixSize, singleMatrixSize,
                d_dense_ref, CUDA_R_32F, CUSPARSE_ORDER_ROW))
            CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(
                handle, matSparse, matDenseRef,
                CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                &bufferSize))
            CHECK_CUDA(cudaMalloc((void**)&dBuffer, bufferSize))
            CHECK_CUSPARSE(cusparseSparseToDense(
                handle, matSparse, matDenseRef,
                CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                dBuffer))
            // compare results
            std::vector<float> h_dense_ref(totalMatrixSize, 0.0f);
            std::vector<float> h_DosePVCSCollective(totalMatrixSize, 0.0f);
            CHECK_CUDA(cudaMemcpy(h_dense_ref.data(), d_dense_ref,
                totalMatrixSize*sizeof(float), cudaMemcpyDeviceToHost))
            CHECK_CUDA(cudaMemcpy(h_DosePVCSCollective.data(),
                DosePVCSCollective, totalMatrixSize*sizeof(float),
                cudaMemcpyDeviceToHost))
            double diff_abs = 0.0;
            for (size_t i=0; i<totalMatrixSize; i++)
                diff_abs += std::abs(h_dense_ref[i] - h_DosePVCSCollective[i]);
            std::cout << "Absolute difference: " << diff_abs << std::endl;
            CHECK_CUSPARSE(cusparseDestroyDnMat(matDenseRef))
            CHECK_CUDA(cudaFree(d_dense_ref))
            CHECK_CUDA(cudaFree(dBuffer))
        #endif


        #if true
            // matrix multiplication test
            float alpha = 1.0f;
            float beta = 1.0f;
            cusparseDnVecDescr_t vecX, vecY;
            float* dX, *dY;
            CHECK_CUDA(cudaMalloc((void**)&dX, singleMatrixSize * sizeof(float)))
            CHECK_CUDA(cudaMalloc((void**)&dY, nBeamlets * sizeof(float)))
            std::vector<float> hX(singleMatrixSize, 1.0f);
            checkCudaErrors(cudaMemcpy(dX, hX.data(),
                singleMatrixSize*sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemset(dY, 0.0f, nBeamlets*sizeof(float)))
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, singleMatrixSize, dX, CUDA_R_32F))
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, nBeamlets, dY, CUDA_R_32F))
            CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matSparse, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
            CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

            cudaEventRecord(start);
            CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matSparse, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Transpose computation time: " << milliseconds << " [ms]" << std::endl;
            
            // read result
            std::vector<float> hY(nBeamlets, 0.0f);
            checkCudaErrors(cudaMemcpy(hY.data(), dY,
                nBeamlets*sizeof(float), cudaMemcpyDeviceToHost));

            // calculate reference value
            std::vector<float> h_DosePVCSCollective(totalMatrixSize, 0.0f);
            checkCudaErrors(cudaMemcpy(h_DosePVCSCollective.data(),
                DosePVCSCollective, totalMatrixSize*sizeof(float),
                cudaMemcpyDeviceToHost));
            std::vector<float> hYRef(nBeamlets, 0.0f);
            for (int i=0; i<nBeamlets; i++) {
                for (int j=0; j<singleMatrixSize; j++) {
                    hYRef[i] += h_DosePVCSCollective[i*singleMatrixSize+j];
                }
            }
            float absolute_diff = 0.0f;
            double scale = 0.0;
            for (int i=0; i<nBeamlets; i++) {
                absolute_diff += std::abs(hYRef[i] - hY[i]);
                scale += hY[i];
            }
            std::cout << "Absolute difference: " << absolute_diff << 
                " / Scale: " << scale << std::endl;

            CHECK_CUDA(cudaFree(dBuffer))
            CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
            CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
            CHECK_CUDA(cudaFree(dX))
            CHECK_CUDA(cudaFree(dY))
        #endif
        CHECK_CUSPARSE(cusparseDestroyDnMat(matDense))
        CHECK_CUSPARSE(cusparseDestroySpMat(matSparse))
        CHECK_CUDA(cudaFree(d_csr_offsets))
        CHECK_CUDA(cudaFree(d_csr_columns))
        CHECK_CUDA(cudaFree(d_csr_values))
        CHECK_CUSPARSE(cusparseDestroy(handle))

    #elif true
        // transpose
        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matSparse;
        int* d_csr_offsets;
        CHECK_CUDA(cudaMalloc((void**)&d_csr_offsets, (singleMatrixSize+1)*sizeof(float)))
        int* d_csr_columns;
        float* d_csr_values;
        cusparseDnMatDescr_t matDense;
        void* dBuffer = nullptr;
        size_t bufferSize = 0;
        CHECK_CUSPARSE(cusparseCreate(&handle))
        CHECK_CUSPARSE(cusparseCreateDnMat(
            &matDense, singleMatrixSize, nBeamlets, singleMatrixSize, DosePVCSCollective,
            CUDA_R_32F, CUSPARSE_ORDER_COL))

        CHECK_CUSPARSE(cusparseCreateCsr(
            &matSparse, singleMatrixSize, nBeamlets, 0,
            d_csr_offsets, nullptr, nullptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
        
        CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
            handle, matDense, matSparse,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            &bufferSize))
        
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

        CHECK_CUSPARSE(cusparseDenseToSparse_analysis(
            handle, matDense, matSparse,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            dBuffer))
        
        int64_t num_rows_tmp, num_cols_tmp, nnz;
        CHECK_CUSPARSE(cusparseSpMatGetSize(matSparse, &num_rows_tmp, &num_cols_tmp,
            &nnz))
        
        CHECK_CUDA(cudaMalloc((void**)&d_csr_columns, nnz*sizeof(float)))
        CHECK_CUDA(cudaMalloc((void**)&d_csr_values, nnz*sizeof(float)))
        CHECK_CUSPARSE(cusparseCsrSetPointers(matSparse, d_csr_offsets, d_csr_columns, d_csr_values))
        CHECK_CUSPARSE(cusparseDenseToSparse_convert(
            handle, matDense, matSparse,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
            dBuffer))
        CHECK_CUDA(cudaFree(dBuffer))
        
        
        // matrix-vector multiplication experiment
        cusparseDnVecDescr_t vecX, vecY;
        float* dX;
        float* dY;
        float alpha = 1.0f;
        float beta = 1.0f;
        std::vector<float> hX(nBeamlets, 0.0f);
        CHECK_CUDA(cudaMalloc((void**)&dX, nBeamlets*sizeof(float)))
        CHECK_CUDA(cudaMalloc((void**)&dY, singleMatrixSize*sizeof(float)))
        CHECK_CUDA(cudaMemset(dY, 0.0f, singleMatrixSize*sizeof(float)))
        CHECK_CUDA(cudaMemcpy(dX, hX.data(), nBeamlets*sizeof(float), cudaMemcpyHostToDevice))
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, nBeamlets, dX, CUDA_R_32F))
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, singleMatrixSize, dY, CUDA_R_32F))
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matSparse, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

        cudaEventRecord(start);
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matSparse, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Number of non-zero elements: " << nnz << std::endl;
        std::cout << "Non-transpose computation time: " << milliseconds << " [ms]" << std::endl;

        CHECK_CUDA(cudaFree(dBuffer))

        CHECK_CUDA(cudaFree(dX))
        CHECK_CUDA(cudaFree(dY))

        CHECK_CUSPARSE(cusparseDestroyDnMat(matDense))
        CHECK_CUSPARSE(cusparseDestroySpMat(matSparse))
        CHECK_CUSPARSE(cusparseDestroy(handle))
        CHECK_CUDA(cudaFree(d_csr_columns))
        CHECK_CUDA(cudaFree(d_csr_offsets))
        CHECK_CUDA(cudaFree(d_csr_values))
    #endif

    CHECK_CUDA(cudaFree(DosePVCSCollective))

    // clean up
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(DoseBEV_array));
    checkCudaErrors(cudaFree(DensityBEV_array));
    checkCudaErrors(cudaFree(TermaBEV_array));
    checkCudaErrors(cudaFree(fluence_array));
    checkCudaErrors(cudaFree(d_beams));

    return 0;
}