#include "IMRTDoseMat.cuh"
#include <iostream>
#include <chrono>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


bool IMRT::MatCSR32_fromfile(const std::string resultFolder,
    size_t numColsPerMat, const std::vector<StructInfo>& structs
) {
    #if TIMING
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif

    std::vector<std::pair<std::vector<size_t>, std::string>> inputQueue {
        {std::vector<size_t>(), std::string("NonZeroElements")},
        {std::vector<size_t>(), std::string("numRowsPerMat")}
    };

    size_t numMatrices = 0;
    for (auto& a: inputQueue) {
        std::vector<size_t>& array = a.first;
        const std::string& name = a.second;
        fs::path fullFile = fs::path(resultFolder) / (name + std::string(".bin"));
        std::ifstream f(fullFile.string());
        if (! f.is_open()) {
            std::cerr << "Cannot open file: " << fullFile << std::endl;
            return 1;
        }
        if (numMatrices == 0) {
            f.seekg(0, std::ios::end);
            numMatrices = f.tellg() / sizeof(size_t);
            f.seekg(0, std::ios::beg);
        }
        array.resize(numMatrices);
        f.read((char*)array.data(), numMatrices*sizeof(size_t));
        f.close();
    }

    // fill offsetsBuffer
    fs::path offsetsBufferFile = fs::path(resultFolder) / std::string("offsetsBuffer.bin");
    std::ifstream f(offsetsBufferFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << offsetsBufferFile << std::endl;
        return 1;
    }
    f.seekg(0, std::ios::end);
    size_t offsetsBufferSize = f.tellg() / sizeof(size_t);
    f.seekg(0, std::ios::beg);
    std::vector<size_t> h_offsetsBuffer(offsetsBufferSize, 0);
    f.read((char*)(h_offsetsBuffer.data()), offsetsBufferSize*sizeof(size_t));
    f.close();

    size_t nnz = 0;
    fs::path columnsBufferFile = fs::path(resultFolder) / std::string("columnsBuffer.bin");
    f.open(columnsBufferFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << columnsBufferFile << std::endl;
        return 1;
    }
    f.seekg(0, std::ios::end);
    nnz = f.tellg() / sizeof(size_t);
    f.seekg(0, std::ios::beg);
    std::vector<size_t> h_columnsBuffer(nnz, 0);
    f.read((char*)(h_columnsBuffer.data()), nnz*sizeof(size_t));
    f.close();

    fs::path valuesBufferFile = fs::path(resultFolder) / std::string("valuesBuffer.bin");
    f.open(valuesBufferFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << valuesBufferFile << std::endl;
        return 1;
    }
    std::vector<float> m_values(nnz, 0);
    f.read((char*)m_values.data(), nnz*sizeof(float));
    f.close();

    #if TIMING
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Time elapsed to load data from file: " << std::fixed
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    // convert the size_t array to int array in parallel
    std::vector<int> m_offsets(offsetsBufferSize, 0);
    std::vector<int> m_columns(nnz, 0);
    for (size_t i=0; i<offsetsBufferSize; i++)
        m_offsets[i] = static_cast<int>(h_offsetsBuffer[i]);
    for (size_t i=0; i<nnz; i++) {
        m_columns[i] = static_cast<int>(h_columnsBuffer[i]);
    }

    #if TIMING
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
        std::cout << "Time elapsed to convert size_t to int: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif


    // load the data to device
    int* d_offsets = nullptr;
    int* d_columns = nullptr;
    float* d_values = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_offsets, offsetsBufferSize*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_columns, nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_values, nnz*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_offsets, h_offsetsBuffer.data(),
        offsetsBufferSize*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_columns, m_columns.data(),
        nnz*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_values, m_values.data(),
        nnz*sizeof(float), cudaMemcpyHostToDevice));

    #if TIMING
        auto time3 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2);
        std::cout << "Time elapsed to copy host data to device: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    // create matrices
    std::vector<MatCSR32> matricesT(numMatrices);
    size_t cumuRows = 0;
    size_t cumuNnz = 0;
    for (int i=0; i<numMatrices; i++) {
        // set the pointers
        int* d_currentOffsets = d_offsets + cumuRows;
        int* d_currentColumns = d_columns + cumuNnz;
        float* d_currentValues = d_values + cumuNnz;

        size_t currentNumRows = inputQueue[1].first[i];
        size_t currentNnz = inputQueue[0].first[i];
        cumuRows += currentNumRows + 1;
        cumuNnz += currentNnz;

        MatCSR32& currentMat = matricesT[i];
        currentMat.d_csr_offsets = d_currentOffsets;
        currentMat.d_csr_columns = d_currentColumns;
        currentMat.d_csr_values = d_currentValues;
        currentMat.numRows = static_cast<int>(currentNumRows);
        currentMat.numCols = static_cast<int>(numColsPerMat);
        currentMat.nnz = static_cast<int64_t>(currentNnz);
        checkCusparse(cusparseCreateCsr(
            &currentMat.matA, currentNumRows, numColsPerMat, currentNnz,
            d_currentOffsets, d_currentColumns, d_currentValues,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }
    #if TIMING
        auto time4 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3);
        std::cout << "Time elapsed to construct the sparse matrices: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    // construct OAR filtering matrix
    MatCSR32 matFilter, matFilterT;
    getOARFilter(matFilter, matFilterT, structs, numColsPerMat);
    #if TIMING
        auto time5 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time5 - time4);
        std::cout << "Time elapsed to construct the OAR filtering matrices: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    #if false
        test_MatFilter(matFilter, matFilterT);
    #endif

    std::vector<MatCSR32> OARMatricesT(numMatrices);
    int* d_bufferOffsets = nullptr;
    int* d_bufferColumns = nullptr;
    float* d_bufferValues = nullptr;
    OARFiltering(OARMatricesT, matricesT, matFilter, matFilterT,
        &d_bufferOffsets, &d_bufferColumns, &d_bufferValues);
    #if TIMING
        auto time6 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time6 - time5);
        std::cout << "Time elapsed to construct the dose loading matrix "
            "restricted to OAR volumes: " << duration.count() * 0.001f
            << " [s]" << std::endl; 
    #endif

    // clean up
    for (int i=0; i<numMatrices; i++) {
        MatCSR32& currentMatOAR = OARMatricesT[i];
        checkCusparse(cusparseDestroySpMat(currentMatOAR.matA));
        currentMatOAR.matA = nullptr;
        currentMatOAR.d_csr_offsets = nullptr;
        currentMatOAR.d_csr_columns = nullptr;
        currentMatOAR.d_csr_values = nullptr;

        MatCSR32& currentMat = matricesT[i];
        checkCusparse(cusparseDestroySpMat(currentMat.matA));
        currentMat.matA = nullptr;
        currentMat.d_csr_offsets = nullptr;
        currentMat.d_csr_columns = nullptr;
        currentMat.d_csr_values = nullptr;
    }
    checkCudaErrors(cudaFree(d_bufferOffsets));
    checkCudaErrors(cudaFree(d_bufferColumns));
    checkCudaErrors(cudaFree(d_bufferValues));
    checkCudaErrors(cudaFree(d_offsets));
    checkCudaErrors(cudaFree(d_columns));
    checkCudaErrors(cudaFree(d_values));
    return 0;
}


bool IMRT::getOARFilter(MatCSR32& matFilter, MatCSR32& matFilterT,
    const std::vector<StructInfo>& structs, size_t nVoxels
) {
    std::vector<int> nonZeroVoxels;
    int totalCount = 0;
    for (int i=0; i<structs.size(); i++) {
        const StructInfo& currentStruct = structs[i];
        if (currentStruct.maxWeights < eps_fastdose &&
            currentStruct.minDoseTargetWeights < eps_fastdose &&
            currentStruct.OARWeights < eps_fastdose) {
            std::cout << "Structure: " << currentStruct.name
                << " is irrelevant in the optimization, skip." << std::endl;
            continue;
        }

        int localCount = 0;
        if (nVoxels != currentStruct.size.x
            * currentStruct.size.y * currentStruct.size.z
        ) {
            std::cerr << "Number of voxels not consistent across structures." << std::endl;
            return 1;
        }
        for (int j=0; j<nVoxels; j++)
            localCount += (currentStruct.mask[j] > 0);
        nonZeroVoxels.push_back(localCount);
        totalCount += localCount;
        std::cout << "Structure: " << currentStruct.name
            << ", non-zero voxels: " << localCount << std::endl;
    }
    std::cout << "OAR filtering matrix, number of non-zero voxels: " << totalCount << std::endl;

    std::vector<int> h_filterOffsets(totalCount+1, 0);
    std::vector<int> h_filterColumns(totalCount, 0);
    std::vector<float> h_filterValues(totalCount, 1.0f);
    int idx = 0;
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
    for (int i=0; i<totalCount+1; i++)
        h_filterOffsets[i] = i;
    checkCudaErrors(cudaMalloc((void**)&matFilterT.d_csr_offsets, (totalCount+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&matFilterT.d_csr_columns, totalCount*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&matFilterT.d_csr_values, totalCount*sizeof(float)));
    checkCudaErrors(cudaMemcpy(matFilterT.d_csr_offsets, h_filterOffsets.data(),
        h_filterOffsets.size()*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(matFilterT.d_csr_columns, h_filterColumns.data(),
        h_filterColumns.size()*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(matFilterT.d_csr_values, h_filterValues.data(),
        h_filterValues.size()*sizeof(float), cudaMemcpyHostToDevice));
    matFilterT.numRows = totalCount;
    matFilterT.numCols = nVoxels;
    matFilterT.nnz = totalCount;
    checkCusparse(cusparseCreateCsr(&matFilterT.matA, totalCount, nVoxels, totalCount,
        matFilterT.d_csr_offsets, matFilterT.d_csr_columns, matFilterT.d_csr_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));


    // initialize matFilter
    checkCudaErrors(cudaMalloc((void**)&matFilter.d_csr_offsets, (nVoxels+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&matFilter.d_csr_columns, totalCount*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&matFilter.d_csr_values, totalCount*sizeof(float)));
    matFilter.numRows = nVoxels;
    matFilter.numCols = totalCount;
    matFilter.nnz = totalCount;
    checkCusparse(cusparseCreateCsr(&matFilter.matA, nVoxels, totalCount, totalCount,
        matFilter.d_csr_offsets, matFilter.d_csr_columns, matFilter.d_csr_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    size_t transposeBufferSize;
    void* transposeBuffer = nullptr;
    cusparseHandle_t handle = nullptr;
    checkCusparse(cusparseCreate(&handle));
    checkCusparse(cusparseCsr2cscEx2_bufferSize(
        handle, matFilterT.numRows, matFilterT.numCols, matFilterT.nnz,
        matFilterT.d_csr_values, matFilterT.d_csr_offsets, matFilterT.d_csr_columns,
        matFilter.d_csr_values, matFilter.d_csr_offsets, matFilter.d_csr_columns,
        CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
        &transposeBufferSize));
    checkCudaErrors(cudaMalloc((void**)&transposeBuffer, transposeBufferSize));
    checkCusparse(cusparseCsr2cscEx2(
        handle, matFilterT.numRows, matFilterT.numCols, matFilter.nnz,
        matFilterT.d_csr_values, matFilterT.d_csr_offsets, matFilterT.d_csr_columns,
        matFilter.d_csr_values, matFilter.d_csr_offsets, matFilter.d_csr_columns,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT,
        transposeBuffer));
    checkCudaErrors(cudaFree(transposeBuffer));
    checkCusparse(cusparseDestroy(handle));
    return 0;
}


bool IMRT::OARFiltering(std::vector<MatCSR32>& OARMatricesT,
    const std::vector<MatCSR32>& matricesT,
    const MatCSR32& matFilter, const MatCSR32& matFilterT,
    int** d_bufferOffsets, int** d_bufferColumns, float** d_bufferValues
) {
    // preparation
    #if true
        int numMatrices = matricesT.size();
        if (numMatrices != OARMatricesT.size()) {
            std::cerr << "The size of the two vectors, matricesT and "
                "OARMatricesT, are not the same." << std::endl;
            return 1;
        }
    #else
        int numMatrices = 1;
    #endif
    std::vector<cudaStream_t> streams(numMatrices);
    std::vector<cusparseHandle_t> handles(numMatrices);
    std::vector<cusparseSpGEMMDescr_t> SpGEMMDescs(numMatrices);

    // first, we know the number of rows of the result.
    // So we can pre allocate d_bufferOffsets
    std::vector<int> offsetIdx(numMatrices);
    int offsetIdxPrev = 0;
    for (int i=0; i<numMatrices; i++) {
        offsetIdx[i] = offsetIdxPrev;
        offsetIdxPrev += matricesT[i].numRows + 1;
    }
    checkCudaErrors(cudaMalloc((void**)d_bufferOffsets, offsetIdxPrev*sizeof(int)));

    for (int i=0; i<numMatrices; i++) {
        MatCSR32& destMat = OARMatricesT[i];
        destMat.d_csr_offsets = *d_bufferOffsets + offsetIdx[i];
        int destNumRows = matricesT[i].numRows;
        checkCusparse(cusparseCreateCsr(&destMat.matA, destNumRows, matFilter.numCols, 0,
            destMat.d_csr_offsets, nullptr, nullptr,
            CUSPARSE_INDEX_32I,  CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        checkCusparse(cusparseSpGEMM_createDescr(&SpGEMMDescs[i]));
        checkCudaErrors(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        checkCusparse(cusparseCreate(&handles[i]));
        checkCusparse(cusparseSetStream(handles[i], streams[i]));
    }

    // ask bufferSize1 bytes for external memory
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0f;
    float beta = 0.0f;

        std::vector<size_t> bufferSize1(numMatrices, 0);
        void* buffer1 = nullptr;
        for (int i=0; i<numMatrices; i++) {
            checkCusparse(cusparseSpGEMM_workEstimation(handles[i], opA, opB,
                &alpha, matricesT[i].matA, matFilter.matA, &beta, OARMatricesT[i].matA,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescs[i], &bufferSize1[i], nullptr));
        }
        cudaDeviceSynchronize();
        size_t bufferSize1_total = 0;
        for (int i=0; i<numMatrices; i++)
            bufferSize1_total += bufferSize1[i];
        std::cout << "bufferSize1_total: " << bufferSize1_total << std::endl;
        checkCudaErrors(cudaMalloc(&buffer1, bufferSize1_total));

        size_t cumuBufferOffset1 = 0;
        for (int i=0; i<numMatrices; i++) {
            void* localBuffer1 = static_cast<char*>(buffer1) + cumuBufferOffset1;
            cumuBufferOffset1 += bufferSize1[i];
            checkCusparse(cusparseSpGEMM_workEstimation(handles[i], opA, opB,
                &alpha, matricesT[i].matA, matFilter.matA, &beta, OARMatricesT[i].matA,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescs[i], &bufferSize1[i], localBuffer1));
        }

        // ask bufferSize2 bytes for external memory
        std::vector<size_t> bufferSize2(numMatrices, 0);
        void* buffer2 = nullptr;
        for (int i=0; i<numMatrices; i++) {
            checkCusparse(cusparseSpGEMM_compute(handles[i], opA, opB,
                &alpha, matricesT[i].matA, matFilter.matA, &beta, OARMatricesT[i].matA,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                SpGEMMDescs[i], &bufferSize2[i], nullptr));
        }
        cudaDeviceSynchronize();
        size_t bufferSize2_total = 0;
        for (int i=0; i<numMatrices; i++)
            bufferSize2_total += bufferSize2[i];
        std::cout << "bufferSize2_total: " << bufferSize2_total << std::endl;
        checkCudaErrors(cudaMalloc(&buffer2, bufferSize2_total));

        // compute the intermediate product of A * B
        size_t cumuBufferOffset2 = 0;
        for (int i=0; i<numMatrices; i++) {
            void* localBuffer2 = static_cast<char*>(buffer2) + cumuBufferOffset2;
            cumuBufferOffset2 += bufferSize2[i];
            checkCusparse(cusparseSpGEMM_compute(handles[i], opA, opB,
                &alpha, matricesT[i].matA, matFilter.matA, &beta, OARMatricesT[i].matA,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                SpGEMMDescs[i], &bufferSize2[i], localBuffer2));
        }

        // get matrix C non-zero entries
        size_t totalNnz = 0;
        int64_t C_num_rows1, C_num_cols1;
        for (int i=0; i<numMatrices; i++) {
            checkCusparse(cusparseSpMatGetSize(OARMatricesT[i].matA,
                &C_num_rows1, &C_num_cols1, &OARMatricesT[i].nnz));
            totalNnz += OARMatricesT[i].nnz;
        }
        checkCudaErrors(cudaMalloc((void**)d_bufferColumns, totalNnz*sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)d_bufferValues, totalNnz*sizeof(float)));

        size_t cumuNnzOffset = 0;
        for (int i=0; i<numMatrices; i++) {
            // update matC with the new pointers
            int* localBufferColumns = *d_bufferColumns + cumuNnzOffset;
            float* localBufferValues = *d_bufferValues + cumuNnzOffset;
            cumuNnzOffset += OARMatricesT[i].nnz;
            checkCusparse(cusparseCsrSetPointers(OARMatricesT[i].matA,
                OARMatricesT[i].d_csr_offsets, localBufferColumns, localBufferValues));
        
            // copy the final products to the matirxC
            checkCusparse(cusparseSpGEMM_copy(handles[i], opA, opB,
                &alpha, matricesT[i].matA, matFilter.matA, &beta, OARMatricesT[i].matA,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, SpGEMMDescs[i]));
        }

        std::cout << "Total number of non-zero elements in OARMatrices vector: "
            << cumuNnzOffset << std::endl;

    // clean up
    checkCudaErrors(cudaFree(buffer2));
    checkCudaErrors(cudaFree(buffer1));
    for (int i=0; i<numMatrices; i++) {
        checkCusparse(cusparseDestroy(handles[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
        checkCusparse(cusparseSpGEMM_destroyDescr(SpGEMMDescs[i]));
    }
    return 0;
}