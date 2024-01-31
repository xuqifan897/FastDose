#include <fstream>
#include <chrono>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include "IMRTDoseMatEigen.cuh"
#include "IMRTDoseMatEns.cuh"

bool IMRT::MatCSR_Eigen::customInit(
    EigenIdxType nRows, EigenIdxType nCols, EigenIdxType nnz,
    EigenIdxType* offsets, EigenIdxType* columns, float* values
) {
    this->m_outerSize = nRows;
    this->m_innerSize = nCols;
    this->m_outerIndex = offsets;
    this->m_innerNonZeros = nullptr;

    StorageTransparent data;
    *(data.getIndices()) = columns;
    *(data.getValues()) = values;
    data.getSize() = nnz;
    data.getAllocatedSize() = nnz;

    this->m_data.swap(data);
    return 0;
}


bool IMRT::MatCSR_Eigen::fromEnsemble(MatCSREnsemble& source) {
    const std::vector<size_t>& numRowsPerMat = source.numRowsPerMat;
    size_t numRowsTotal = source.CumuNumRowsPerMat.back();
    std::vector<size_t> h_csr_offsets(numRowsTotal + 1);
    this->m_outerSize = numRowsTotal;
    this->m_innerSize = source.numColsPerMat;

    size_t ensembleOffsetSize = source.OffsetBufferIdx.back() + numRowsPerMat.back() + 1;
    std::vector<size_t> h_offsetsBuffer(ensembleOffsetSize, 0);
    checkCudaErrors(cudaMemcpy(h_offsetsBuffer.data(), source.d_offsetsBuffer,
        ensembleOffsetSize*sizeof(size_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(source.d_offsetsBuffer));
    source.d_offsetsBuffer = nullptr;

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

    size_t nnz = source.CumuNonZeroElements.back();
    std::vector<size_t> h_csr_columns(nnz, 0);
    float* m_values = nullptr;
    m_values = (float*)malloc(nnz*sizeof(float));
    checkCudaErrors(cudaMemcpy(h_csr_columns.data(), source.d_columnsBuffer,
        nnz*sizeof(size_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_values, source.d_valuesBuffer,
        nnz*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(source.d_columnsBuffer));
    checkCudaErrors(cudaFree(source.d_valuesBuffer));
    source.d_columnsBuffer = nullptr;
    source.d_valuesBuffer = nullptr;

    // then, convert the size_t array to EigenIdxType array
    EigenIdxType* m_offsets = nullptr;
    EigenIdxType* m_columns = nullptr;
    m_offsets = (EigenIdxType*)malloc((numRowsTotal+1)*sizeof(EigenIdxType));
    m_columns = (EigenIdxType*)malloc(nnz*sizeof(EigenIdxType));
    for (size_t i=0; i<numRowsTotal+1; i++)
        m_offsets[i] = static_cast<EigenIdxType>(h_csr_offsets[i]);
    for (size_t i=0; i<nnz; i++)
        m_columns[i] = static_cast<EigenIdxType>(h_csr_columns[i]);
    
    this->m_outerIndex = m_offsets;
    this->m_innerNonZeros = nullptr;
    
    StorageTransparent data;
    *(data.getIndices()) = m_columns;
    *(data.getValues()) = m_values;
    data.getSize() = nnz;
    data.getAllocatedSize() = nnz;

    this->m_data.swap(data);

    return 0;
}


bool IMRT::MatCSR_Eigen::fromfile(const std::string& resultFolder, size_t numCols) {
    cudaEvent_t start, stop;
    float milliseconds;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    cudaEventRecord(start);

    std::vector<std::pair<std::vector<size_t>, std::string>> inputQueue {
        {std::vector<size_t>(), std::string("NonZeroElements")},
        {std::vector<size_t>(), std::string("numRowsPerMat")},
        {std::vector<size_t>(), std::string("OffsetBufferIdx")}
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
    std::vector<size_t> h_offsetsBuffer(offsetsBufferSize, 0);
    f.seekg(0, std::ios::beg);
    f.read((char*)(h_offsetsBuffer.data()), offsetsBufferSize*sizeof(size_t));
    f.close();

    size_t nnz = 0;
    fs::path columnsBufferFile = fs::path(resultFolder) / std::string("columnsBuffer.bin");
    f.open(columnsBufferFile.string());
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
    float* m_values = (float*)malloc(nnz*sizeof(float));
    f.read((char*)m_values, nnz*sizeof(float));
    f.close();

    size_t numRowsTotal = 0;
    for (size_t i=0; i<inputQueue[1].first.size(); i++)
        numRowsTotal += inputQueue[1].first[i];

    EigenIdxType* m_offsets = (EigenIdxType*)malloc((numRowsTotal+1)*sizeof(EigenIdxType));
    EigenIdxType* m_columns = (EigenIdxType*)malloc(nnz*sizeof(EigenIdxType));

    EigenIdxType sourceOffset = 0;
    EigenIdxType destOffset = 0;
    EigenIdxType valueOffset = 0;
    for (size_t i=0; i<inputQueue[1].first.size(); i++) {
        size_t currentNumRows = inputQueue[1].first[i];
        for (size_t j=0; j<currentNumRows; j++) {
            m_offsets[destOffset] = static_cast<EigenIdxType>(
                h_offsetsBuffer[sourceOffset] + valueOffset);
            destOffset ++;
            sourceOffset ++;
        }
        valueOffset += h_offsetsBuffer[sourceOffset];
        sourceOffset++;
    }
    m_offsets[destOffset] = valueOffset;

    for (size_t i=0; i<nnz; i++)
        m_columns[i] = static_cast<EigenIdxType>(h_columnsBuffer[i]);

    this->m_outerSize = numRowsTotal;
    this->m_innerSize = numCols;
    this->m_outerIndex = m_offsets;
    this->m_innerNonZeros = nullptr;

    StorageTransparent data;
    *(data.getIndices()) = m_columns;
    *(data.getValues()) = m_values;
    data.getSize() = nnz;
    data.getAllocatedSize() = nnz;

    this->m_data.swap(data);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    std::cout << "Sparse matrix initialization completed. Time elapsed: "
        << std::fixed << milliseconds * 0.001f << " [s]" << std::endl << std::endl;
    
    return 0;
}


IMRT::MatCSR_Eigen operator*(const IMRT::MatCSR_Eigen& a, const IMRT::MatCSR_Eigen& b) {
    const Eigen::SparseMatrix<float, Eigen::RowMajor, EigenIdxType>& a_base = a;
    const Eigen::SparseMatrix<float, Eigen::RowMajor, EigenIdxType>& b_base = b;
    Eigen::SparseMatrix<float, Eigen::RowMajor, EigenIdxType> intermediate = a_base * b_base;
    IMRT::MatCSR_Eigen result;
    result.swap(intermediate);
    return result;
}


IMRT::MatCSR_Eigen IMRT::MatCSR_Eigen::transpose() const {
    const Eigen::SparseMatrix<float, Eigen::RowMajor, EigenIdxType>* pointer = this;
    SparseMatrix<float, Eigen::RowMajor, EigenIdxType> intermediate = pointer->transpose();
    MatCSR_Eigen result;
    result.swap(intermediate);
    return result;
}


bool IMRT::MatOARSlicing(const MatCSR_Eigen& matrixT, MatCSR_Eigen& A,
    MatCSR_Eigen& AT, const std::vector<StructInfo>& structs
) {
    // firstly, construct the sparse matrix that filters
    // out the OAR voxels from the full volume
    std::vector<size_t> nonZeroVoxels;
    size_t totalCount = 0;
    size_t nVoxels = matrixT.cols();
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
    std::cout << "Total number of non-zero voxels: " << totalCount << std::endl << std::endl;


    EigenIdxType* h_filterOffsets = (EigenIdxType*)malloc((totalCount+1)*sizeof(EigenIdxType));
    EigenIdxType* h_filterColumns = (EigenIdxType*)malloc(totalCount*sizeof(EigenIdxType));
    float* h_filterValues = (float*)malloc(totalCount*sizeof(float));
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

    MatCSR_Eigen structFilter;
    structFilter.customInit(totalCount, nVoxels, totalCount,
        h_filterOffsets, h_filterColumns, h_filterValues);
    
    #if slicingTiming
        std::cout << "Transpose starts." << std::endl;
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    
    MatCSR_Eigen structTranspose = structFilter.transpose();
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<
            std::chrono::seconds>(time1 - time0);
        std::cout << "Transpose time: " << duration.count() << " [s]."
            << std::endl << "Multiplication 1 starts." << std::endl;
    #endif

    AT = matrixT * structTranspose;
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<
            std::chrono::seconds>(time2 - time1);
        std::cout << "Multiplication time: " << duration.count() << " [s]."
            << std::endl << "The second transpose starts." << std::endl;
    #endif

    A = AT.transpose();
    #if slicingTiming
        auto time3 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<
            std::chrono::seconds>(time3 - time2);
        std::cout << "The second transpose time: " << duration.count() << " [s]" << std::endl;
    #endif
    return 0;
}