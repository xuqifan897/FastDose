#include "IMRTDoseMat.cuh"


bool IMRT::MatCSR::dense2sparse(
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


bool IMRT::MatCSR::fuseEnsemble(MatCSREnsemble& matEns) {
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
    checkCudaErrors(cudaMalloc((void**)&this->d_csr_offsets, numRowsTotal*sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(this->d_csr_offsets, h_csr_offsets.data(),
        (h_csr_offsets.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

    checkCusparse(cusparseCreateCsr(
        &(this->matA), numRowsTotal, matEns.numColsPerMat, this->nnz,
        this->d_csr_offsets, this->d_csr_columns, this->d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    return 1;
}