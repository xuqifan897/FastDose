#include <chrono>
#include "IMRTDoseMatEigen.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTInit.cuh"

bool IMRT::OARFiltering(
    const std::string& resultFolder,
    const std::vector<StructInfo>& structs,
    MatCSR64& SpOARmat, MatCSR64& SpOARmatT
) {
    MatCSR_Eigen filter, filterT;
    if (getStructFilter(filter, filterT, structs)) {
        std::cerr << "OAR filter and its transpose construction error." << std::endl;
        return 1;
    }
    std::vector<MatCSR_Eigen> OARMatrices;
    std::vector<MatCSR_Eigen> OARMatricesT;
    if (parallelSpGEMM(resultFolder, filter, filterT, OARMatrices, OARMatricesT)) {
        std::cerr << "CPU OAR dose loading matrices and their transpose "
            "construction error." << std::endl;
        return 1;
    }

    MatCSR_Eigen OARMat, OARMatT;
    if (parallelMatCoalease(OARMat, OARMatT, OARMatrices, OARMatricesT)) {
        std::cerr << "GPU OAR dose loading matrix and its transpose "
            "construction error." << std::endl;
        return 1;
    }

    #if false
        if(test_OARMat_OARMatT(OARMat, OARMatT)) {
            std::cerr << "OARMat and OARMatT test error." << std::endl;
            return 1;
        } else {
            std::cout << "OARMat and OARMatT match each other, test passed!" << std::endl;
        }
    #endif

    if (SpOARmatInit(SpOARmat, SpOARmatT, OARMat, OARMatT)) {
        std::cerr << "Error converting OARmat from CPU to GPU." << std::endl;
        return 1;
    }

    #if false
        if(test_SpMatOAR(SpOARmat, SpOARmatT, filter, OARMatrices)) {
            std::cerr << "SpOARmat and SpOARmatT validation error." << std::endl;
            return 1;
        }
    #endif

    return 0;
}


bool IMRT::getStructFilter (
    MatCSR_Eigen& filter, MatCSR_Eigen& filterT,
    const std::vector<StructInfo>& structs
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif

    std::vector<size_t> nonZeroVoxels;
    size_t totalCount = 0;
    size_t nVoxels = 0;
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
        if (nVoxels == 0)
            nVoxels = currentStruct.size.x
                * currentStruct.size.y * currentStruct.size.z;
        else if (nVoxels != currentStruct.size.x
            * currentStruct.size.y * currentStruct.size.z) {
            std::cerr << "Number of voxels inconsistent among structures." << std::endl;
            return 1;
        }
        for (size_t j=0; j<nVoxels; j++)
            localCount += (currentStruct.mask[j] > 0);
        nonZeroVoxels.push_back(localCount);
        totalCount += localCount;
        std::cout << "Structure: " << currentStruct.name
            << ", non-zero voxels: " << localCount << std::endl;
    }
    std::cout << "Total number of non-zero voxels: " << totalCount << std::endl << std::endl;

    EigenIdxType* h_filterOffsets = (EigenIdxType*)malloc((totalCount+1)*sizeof(EigenIdxType));
    EigenIdxType* h_filterColumns = new EigenIdxType[totalCount];
    float* h_filterValues = new float[totalCount];
    for (size_t i=0; i<totalCount; i++)
        h_filterValues[i] = 1.0f;

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

    filterT.customInit(totalCount, nVoxels, totalCount,
        h_filterOffsets, h_filterColumns, h_filterValues);
    
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<
            std::chrono::milliseconds>(time1 - time0);
        std::cout << "OAR filter initialization time: " << std::fixed
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif
    
    filter = filterT.transpose();
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<
            std::chrono::milliseconds>(time2 - time1);
        std::cout << "OAR filter transpose time: " << duration.count() * 0.001f
            << " [s]" << std::endl;
    #endif

    return 0;
}


bool IMRT::SpOARmatInit(MatCSR64& SpOARmat, MatCSR64& SpOARmatT,
    const MatCSR_Eigen& OARmat, const MatCSR_Eigen& OARmatT
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif

    size_t* SpOARmat_offsets = nullptr;
    size_t* SpOARmat_columns = nullptr;
    float* SpOARmat_values = nullptr;

    size_t* SpOARmatT_offsets = nullptr;
    size_t* SpOARmatT_columns = nullptr;
    float* SpOARmatT_values = nullptr;

    size_t SpOARmat_numRows = OARmat.getRows();
    size_t SpOARmatT_numRows = OARmatT.getRows();
    size_t nnz = OARmat.getNnz();
    if (nnz != OARmatT.getNnz()) {
        std::cerr << "The number of non-zero elements for OARmat "
            "and OARmatT should be equal" << std::endl;
        return 1;
    }

    checkCudaErrors(cudaMalloc((void**)&SpOARmat_offsets, (SpOARmat_numRows+1)*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&SpOARmat_columns, nnz * sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&SpOARmat_values, nnz * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&SpOARmatT_offsets, (SpOARmatT_numRows+1)*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&SpOARmatT_columns, nnz * sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&SpOARmatT_values, nnz * sizeof(float)));

    int nStreams = 6;
    std::vector<cudaStream_t> streams(nStreams);
    for (int i=0; i<nStreams; i++)
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    
    checkCudaErrors(cudaMemcpyAsync(SpOARmat_offsets, OARmat.getOffset(),
        (SpOARmat_numRows+1)*sizeof(size_t), cudaMemcpyHostToDevice, streams[0]));
    checkCudaErrors(cudaMemcpyAsync(SpOARmat_columns, OARmat.getIndices(),
        nnz * sizeof(size_t), cudaMemcpyHostToDevice, streams[1]));
    checkCudaErrors(cudaMemcpyAsync(SpOARmat_values, OARmat.getValues(),
        nnz * sizeof(float), cudaMemcpyHostToDevice, streams[2]));
    
    checkCudaErrors(cudaMemcpyAsync(SpOARmatT_offsets, OARmatT.getOffset(),
        (SpOARmatT_numRows + 1) * sizeof(size_t), cudaMemcpyHostToDevice, streams[3]));
    checkCudaErrors(cudaMemcpyAsync(SpOARmatT_columns, OARmatT.getIndices(),
        nnz * sizeof(size_t), cudaMemcpyHostToDevice, streams[4]));
    checkCudaErrors(cudaMemcpyAsync(SpOARmatT_values, OARmatT.getValues(),
        nnz * sizeof(float), cudaMemcpyHostToDevice, streams[5]));

    for (int i=0; i<nStreams; i++)
        cudaStreamSynchronize(streams[i]);
    for (int i=0; i<nStreams; i++)
        cudaStreamDestroy(streams[i]);
    
    checkCusparse(cusparseCreateCsr(
        &SpOARmat.matA, OARmat.getRows(), OARmat.getCols(), OARmat.getNnz(),
        SpOARmat_offsets, SpOARmat_columns, SpOARmat_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    checkCusparse(cusparseCreateCsr(
        &SpOARmatT.matA, OARmatT.getRows(), OARmatT.getCols(), OARmatT.getNnz(),
        SpOARmatT_offsets, SpOARmatT_columns, SpOARmatT_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    SpOARmat.d_csr_offsets = SpOARmat_offsets;
    SpOARmat.d_csr_columns = SpOARmat_columns;
    SpOARmat.d_csr_values = SpOARmat_values;
    SpOARmat.numRows = OARmat.getRows();
    SpOARmat.numCols = OARmat.getCols();
    SpOARmat.nnz = OARmat.getNnz();

    SpOARmatT.d_csr_offsets = SpOARmatT_offsets;
    SpOARmatT.d_csr_columns = SpOARmatT_columns;
    SpOARmatT.d_csr_values = SpOARmatT_values;
    SpOARmatT.numRows = OARmatT.getRows();
    SpOARmatT.numCols = OARmatT.getCols();
    SpOARmatT.nnz = OARmatT.getNnz();
    
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Cusparse matrices initialization time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif
    return 0;
}