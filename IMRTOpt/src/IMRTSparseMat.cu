#include "IMRTDoseMat.cuh"


bool IMRT::MatCSR64::dense2sparse(
    float* d_dense, size_t num_rows, size_t num_cols, size_t ld
) {
    checkCudaErrors(cudaMalloc((void**)(&this->d_csr_offsets),
        (num_rows + 1) * sizeof(size_t)));
    
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
    checkCudaErrors(cudaMalloc((void**) &(this->d_csr_columns), nnz*sizeof(size_t)));
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


bool IMRT::MatCSR64::fuseEnsemble(MatCSREnsemble& matEns) {
    // directly move d_columnsBuffer and d_valuesBuffer from matEns
    this->d_csr_columns = matEns.d_columnsBuffer;
    matEns.d_columnsBuffer = nullptr;

    this->d_csr_values = matEns.d_valuesBuffer;
    matEns.d_valuesBuffer = nullptr;

    // then we deal with offsets buffer. Since it is small
    // in size compared to the above two arrays, we do it on cpu.
    const std::vector<size_t>& numRowsPerMat = matEns.numRowsPerMat;
    size_t numRowsTotal = matEns.CumuNumRowsPerMat.back();
    std::vector<size_t> h_csr_offsets(numRowsTotal + 1);
    this->numRows = numRowsTotal;
    this->numCols = matEns.numColsPerMat;
    
    size_t ensembleOffsetSize = matEns.OffsetBufferIdx.back() + numRowsPerMat.back() + 1;
    std::vector<size_t> h_offsetsBuffer(ensembleOffsetSize, 0);
    checkCudaErrors(cudaMemcpy(h_offsetsBuffer.data(), matEns.d_offsetsBuffer,
        ensembleOffsetSize*sizeof(size_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(matEns.d_offsetsBuffer));
    matEns.d_offsetsBuffer = nullptr;
    
    size_t destOffset = 0;
    size_t sourceOffset = 0;
    size_t valueOffset = 0;
    for (size_t i=0; i<numRowsPerMat.size(); i++) {
        size_t currentNumRows = numRowsPerMat[i];
        for (int j=0; j<currentNumRows; j++) {
            h_csr_offsets[destOffset] = h_offsetsBuffer[sourceOffset] + valueOffset;
            destOffset ++;
            sourceOffset ++;
        }
        valueOffset += h_offsetsBuffer[sourceOffset];
        sourceOffset ++;
    }
    h_csr_offsets[destOffset] = valueOffset;

    this->nnz = valueOffset;
    checkCudaErrors(cudaMalloc((void**)&this->d_csr_offsets, (numRowsTotal+1)*sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(this->d_csr_offsets, h_csr_offsets.data(),
        h_csr_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    checkCusparse(cusparseCreateCsr(
        &(this->matA), numRowsTotal, matEns.numColsPerMat, this->nnz,
        this->d_csr_offsets, this->d_csr_columns, this->d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    return 0;
}


bool IMRT::MatOARSlicing(
    const MatCSR64& fullMat, MatCSR64& sliceMat, MatCSR64& sliceMatT,
    const std::vector<StructInfo>& structs
) {
    if (sliceMat.d_csr_offsets != nullptr
        || sliceMat.d_csr_columns != nullptr
        || sliceMat.d_csr_values != nullptr
        || sliceMat.d_buffer_spmv != nullptr
        || sliceMatT.d_csr_offsets != nullptr
        || sliceMatT.d_csr_columns != nullptr
        || sliceMatT.d_csr_values != nullptr
        || sliceMatT.d_buffer_spmv != nullptr
    ) {
        std::cerr << "The input matrices are supposed to be empty, "
            "but non-nullptr pointers detected." << std::endl;
        return 1;
    }

    // firstly, construct the sparse matrix that filters
    // out the OAR voxels from the full volume
    std::vector<size_t> nonZeroVoxels;
    size_t totalCount = 0;
    size_t nVoxels = fullMat.numCols;
    for (int i=0; i<structs.size(); i++) {
        const StructInfo& currentStruct = structs[i];
        if (currentStruct.maxWeights < eps_fastdose &&
            currentStruct.minDoseTargetWeights < eps_fastdose &&
            currentStruct.OARWeights < eps_fastdose) {
            std::cout << "Structure: " << currentStruct.name
                << " is irrelevant in the optimization, skip." << std::endl;
            continue;
        }

        size_t localCount = 0;
        if (nVoxels != currentStruct.size.x
            * currentStruct.size.y * currentStruct.size.z
        ) {
            std::cerr << "Number of voxels not consistent across structures." << std::endl;
            return 1;
        }
        for (size_t j=0; j<nVoxels; j++)
            localCount += (currentStruct.mask[j] > 0);
        nonZeroVoxels.push_back(localCount);
        totalCount += localCount;
        std::cout << "Structure: " << currentStruct.name
            << ", non-zero voxels: " << localCount << std::endl;
    }
    std::cout << "Total number of non-zero voxels: " << totalCount << std::endl;

    std::vector<size_t> h_filterOffsets(totalCount+1, 0);
    std::vector<size_t> h_filterColumns(totalCount, 0);
    std::vector<float> h_filterValues(totalCount, 1.0f);
    size_t idx = 0;
    for (int i=0; i<structs.size(); i++) {
        const StructInfo& currentStruct = structs[i];
        if (currentStruct.maxWeights < eps_fastdose &&
            currentStruct.minDoseTargetWeights < eps_fastdose &&
            currentStruct.OARWeights < eps_fastdose) {
            continue;
        }
        const std::vector<uint8_t>& mask = currentStruct.mask;
        for (size_t j=0; j<nVoxels; j++) {
            if (mask[j] > 0) {
                h_filterColumns[idx] = j;
                idx++;
            }
        }
    }
    for (size_t i=0; i<totalCount+1; i++)
        h_filterOffsets[i] = i;

    cusparseSpMatDescr_t matFilter;
    size_t* d_filterOffsets = nullptr;
    size_t* d_filterColumns = nullptr;
    float* d_filterValues = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_filterOffsets, (totalCount+1)*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&d_filterColumns, totalCount*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&d_filterValues, totalCount*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_filterOffsets, h_filterOffsets.data(),
        (totalCount+1)*sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filterColumns, h_filterColumns.data(),
        totalCount*sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filterValues, h_filterValues.data(),
        totalCount*sizeof(float), cudaMemcpyHostToDevice));
    checkCusparse(cusparseCreateCsr(
        &matFilter, totalCount, nVoxels, totalCount,
        d_filterOffsets, d_filterColumns, d_filterValues,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    
    // initialize sliceMatT
    sliceMatT.numRows = fullMat.numRows;
    sliceMatT.numCols = totalCount;
    checkCudaErrors(cudaMalloc((void**)&sliceMatT.d_csr_offsets,
        (sliceMatT.numRows+1)*sizeof(size_t)));
    checkCusparse(cusparseCreateCsr(
        &sliceMatT.matA, sliceMatT.numRows, sliceMatT.numCols, 0,
        sliceMatT.d_csr_offsets, nullptr, nullptr,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    checkCusparse(cusparseSpGEMM_createDescr(&spgemmDesc));
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    void *dBuffer1 = nullptr, *dBuffer2 = nullptr;
    float alpha = 1.0f, beta = 0.0f;
    cusparseHandle_t handle = nullptr;
    checkCusparse(cusparseCreate(&handle));

    // ask bufferSize1 bytes for external memory
    checkCusparse(cusparseSpGEMM_workEstimation(handle, opA, opB,
        &alpha, fullMat.matA, matFilter, &beta, sliceMatT.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1, nullptr));
    checkCudaErrors(cudaMalloc((void**)&dBuffer1, bufferSize1));

    // inspect the matrices A and B to understand the
    // memory requirement for the next step
    checkCusparse(cusparseSpGEMM_workEstimation(handle, opA, opB,
        &alpha, fullMat.matA, matFilter, &beta, sliceMatT.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1, dBuffer1));

    // ask bufferSize2 bytes for external memory
    checkCusparse(cusparseSpGEMM_compute(handle, opA, opB,
        &alpha, fullMat.matA, matFilter, &beta, sliceMatT.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, nullptr));
    checkCudaErrors(cudaMalloc((void**)&dBuffer2, bufferSize2));

    // compute the intermediate product of A * B
    checkCusparse(cusparseSpGEMM_compute(handle, opA, opB,
        &alpha, fullMat.matA, matFilter, &beta, sliceMatT.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, dBuffer2));

    int64_t C_num_rows1, C_num_cols1;
    checkCusparse(cusparseSpMatGetSize(sliceMatT.matA,
        &C_num_rows1, &C_num_cols1, &sliceMatT.nnz));
    checkCudaErrors(cudaMalloc((void**)&sliceMatT.d_csr_columns, sliceMatT.nnz*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&sliceMatT.d_csr_values, sliceMatT.nnz*sizeof(float)));
    
    // update matC with new pointers
    checkCusparse(cusparseCsrSetPointers(sliceMatT.matA, sliceMatT.d_csr_offsets,
        sliceMatT.d_csr_columns, sliceMatT.d_csr_values));
    
    // copy the final products to the matrix C
    checkCusparse(cusparseSpGEMM_copy(handle, opA, opB,
        &alpha, fullMat.matA, matFilter, &beta, sliceMatT.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));


    
    // initialize sliceMat
    sliceMat.numRows = totalCount;
    sliceMat.numCols = fullMat.numRows;
    checkCudaErrors(cudaMalloc((void**)&sliceMat.d_csr_offsets,
        (sliceMat.numRows+1)*sizeof(size_t)));
    checkCusparse(cusparseCreateCsr(
        &sliceMat.matA, sliceMat.numRows, sliceMat.numCols, 0,
        sliceMat.d_csr_offsets, nullptr, nullptr,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    checkCusparse(cusparseSpGEMM_createDescr(&spgemmDesc));
    opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    opB = CUSPARSE_OPERATION_TRANSPOSE;
    computeType = CUDA_R_32F;
    size_t bufferSize1_new = 0, bufferSize2_new = 0;
    alpha = 1.0f, beta = 0.0f;

    checkCusparse(cusparseSpGEMM_workEstimation(handle, opA, opB,
        &alpha, matFilter, fullMat.matA, &beta, sliceMat.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1_new, nullptr));
    if (bufferSize1_new > bufferSize1) {
        checkCudaErrors(cudaFree(dBuffer1));
        checkCudaErrors(cudaMalloc((void**)&dBuffer1, bufferSize1_new));
    }

    checkCusparse(cusparseSpGEMM_workEstimation(handle, opA, opB,
        &alpha, matFilter, fullMat.matA, &beta, sliceMat.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1_new, dBuffer1));

    checkCusparse(cusparseSpGEMM_compute(handle, opA, opB,
        &alpha, matFilter, fullMat.matA, &beta, sliceMat.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2_new, nullptr));
    if (bufferSize2_new > bufferSize2) {
        checkCudaErrors(cudaFree(dBuffer2));
        checkCudaErrors(cudaMalloc((void**)&dBuffer2, bufferSize2_new));
    }

    checkCusparse(cusparseSpGEMM_compute(handle, opA, opB,
        &alpha, matFilter, fullMat.matA, &beta, sliceMat.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2_new, dBuffer2));

    checkCusparse(cusparseSpMatGetSize(sliceMat.matA,
        &C_num_rows1, &C_num_cols1, &sliceMat.nnz));
    checkCudaErrors(cudaMalloc((void**)&sliceMat.d_csr_columns, sliceMat.nnz*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&sliceMat.d_csr_values, sliceMat.nnz*sizeof(float)));

    checkCusparse(cusparseCsrSetPointers(sliceMat.matA, sliceMat.d_csr_offsets,
        sliceMat.d_csr_columns, sliceMat.d_csr_values));

    checkCusparse(cusparseSpGEMM_copy(handle, opA, opB,
        &alpha, matFilter, fullMat.matA, &beta, sliceMat.matA,
        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    // clean-up
    checkCudaErrors(cudaFree(dBuffer2));
    checkCudaErrors(cudaFree(dBuffer1));
    checkCusparse(cusparseDestroy(handle));
    checkCusparse(cusparseDestroySpMat(matFilter));
    checkCudaErrors(cudaFree(d_filterValues));
    checkCudaErrors(cudaFree(d_filterColumns));
    checkCudaErrors(cudaFree(d_filterOffsets));

    return 0;
}