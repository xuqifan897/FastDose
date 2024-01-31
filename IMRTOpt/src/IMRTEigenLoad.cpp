#include <iostream>
#include <omp.h>
#include <bitset>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include "IMRTDoseMatEigen.cuh"

bool IMRT::parallelSpGEMM(
    const std::string& resultFolder,
    const MatCSR_Eigen& filter,
    const MatCSR_Eigen& filterT,
    std::vector<MatCSR_Eigen>& OARMatrices,
    std::vector<MatCSR_Eigen>& OARMatricesT
) {
    #if true
        size_t number1 = 7195446;
        EigenIdxType number2 = 7195446;
        std::bitset<8> number1_bs(number1);
        std::bitset<8> number2_bs(number2);
        if (number1_bs != number2_bs) {
            std::cerr << "The assumption that size_t and int64_t "
                "have the same binary format on positive integers "
                "doesn't hold true." << std::endl;
            return 1;
        }
    #endif

    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif

    std::vector<std::pair<size_t*, std::string>> inputQueue {
        {nullptr, std::string("NonZeroElements")},
        {nullptr, std::string("numRowsPerMat")}
    };

    EigenIdxType numMatrices = 0;
    for (auto& a: inputQueue) {
        void** array = (void**)&a.first;
        const std::string& name = a.second;
        fs::path fullFile = fs::path(resultFolder) / (name + std::string(".bin"));
        if (readBlockParallel(fullFile.string(), array, &numMatrices)) {
            std::cerr << "Cannot load from file: " << fullFile << std::endl;
            return 1;
        }
    }
    numMatrices /= sizeof(size_t);

    // fill offsetsBuffer
    fs::path offsetsBufferFile = fs::path(resultFolder) / std::string("offsetsBuffer.bin");
    fs::path columnsBufferFile = fs::path(resultFolder) / std::string("columnsBuffer.bin");
    fs::path valuesBufferFile = fs::path(resultFolder) / std::string("valuesBuffer.bin");
    EigenIdxType* h_offsetsBuffer = nullptr;
    EigenIdxType* h_columnsBuffer = nullptr;
    float* h_valuesBuffer = nullptr;
    EigenIdxType offsetsBufferSize = 0;
    EigenIdxType nnz_size_t = 0;
    EigenIdxType nnz_float = 0;
    if (readBlockParallel(offsetsBufferFile.string(), (void**)(&h_offsetsBuffer), &offsetsBufferSize)
        || readBlockParallel(columnsBufferFile.string(), (void**)(&h_columnsBuffer), &nnz_size_t)
        || readBlockParallel(valuesBufferFile.string(), (void**)(&h_valuesBuffer), &nnz_float)
    ) {
        std::cerr << "Cannot open one of the three files: " << std::endl
            << offsetsBufferFile << std::endl << columnsBufferFile << std::endl
            << valuesBufferFile << std::endl;
        return 0;
    }
    if (nnz_float/sizeof(float) != nnz_size_t/sizeof(size_t)) {
        std::cerr << "The number of elements in columns and values "
            "are not equal." << std::endl;
        return 1;
    }

    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Data loading time elapsed: " << duration.count() * 0.001f
            << " [s]"<< std::endl;
    #endif

    std::vector<MatCSR_Eigen> matricesT(numMatrices);
    std::vector<size_t> cumuNnz(numMatrices, 0);
    std::vector<size_t> cumuNumRows(numMatrices, 0);
    for (int i=1; i<numMatrices; i++) {
        cumuNnz[i] = cumuNnz[i-1] + inputQueue[0].first[i-1];
        cumuNumRows[i] = cumuNumRows[i-1] + inputQueue[1].first[i-1] + 1;
    }
    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        size_t localNnz = inputQueue[0].first[i];
        size_t localNumRows = inputQueue[1].first[i];
        EigenIdxType* m_offsets = (EigenIdxType*)malloc((localNumRows+1)*sizeof(EigenIdxType));
        EigenIdxType* m_columns = new EigenIdxType[localNnz];
        float* m_values = new float[localNnz];

        size_t local_cumuNnz = cumuNnz[i];
        size_t local_cumuNumRows = cumuNumRows[i];

        std::copy(&h_offsetsBuffer[local_cumuNumRows],
            &h_offsetsBuffer[local_cumuNumRows+localNumRows+1], m_offsets);
        std::copy(&h_columnsBuffer[local_cumuNnz],
            &(h_columnsBuffer)[local_cumuNnz+localNnz], m_columns);
        std::copy(&h_valuesBuffer[local_cumuNnz],
            &h_valuesBuffer[local_cumuNnz+localNnz], m_values);

        matricesT[i].customInit(localNumRows, filter.getRows(), localNnz,
            m_offsets, m_columns, m_values);
    }
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
        std::cout << "Transpose full matrix initialization time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    // clean up
    delete[] inputQueue[0].first;
    delete[] inputQueue[1].first;
    delete[] h_offsetsBuffer;
    delete[] h_columnsBuffer;
    delete[] h_valuesBuffer;

    OARMatrices.resize(numMatrices);
    OARMatricesT.resize(numMatrices);
    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        OARMatricesT[i] = matricesT[i] * filter;
        OARMatrices[i] = OARMatricesT[i].transpose();
    }
    #if slicingTiming
        auto time3 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2);
        std::cout << "OAR dose loading matrices and their transpose construction time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    #if false
        test_parallelSpGEMM(OARMatrices, OARMatricesT, matricesT, filter);
    #endif

    return 0;
}


bool IMRT::readBlockParallel(
    const std::string& filename,
    void** pointer, EigenIdxType* size
) {
    std::ifstream f(filename);
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return 1;
    }
    f.seekg(0, std::ios::end);
    *size = f.tellg();
    f.close();
    *pointer = new char[*size];

    const int numThreads = 8;
    EigenIdxType blockPerThread = (*size + numThreads - 1) / numThreads;
    #pragma omp parallel num_threads(numThreads)
    {
        int threadIdx = omp_get_thread_num();
        EigenIdxType threadStart = threadIdx * blockPerThread;
        EigenIdxType threadEnd = std::min((threadIdx + 1) * blockPerThread, *size);
        readBlockParallelFunc(filename, (char*)(*pointer), threadStart, threadEnd);
    }
    return 0;
}

void IMRT::readBlockParallelFunc(const std::string& filename,
    char* buffer, size_t start, size_t end
) {
    std::ifstream f(filename);
    if (! f.is_open())
        std::cerr << "Cannot open file: " << filename << std::endl;
    
    f.seekg(start, std::ios::beg);
    f.read(buffer+start, end-start);
    f.close();
}

bool IMRT::parallelMatCoalease(
    MatCSR_Eigen& OARmat, MatCSR_Eigen& OARmatT,
    const std::vector<MatCSR_Eigen>& OARMatrices,
    const std::vector<MatCSR_Eigen>& OARMatricesT
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    // initialize OARmatT
    int numMatrices = OARMatricesT.size();
    size_t numRowsTotal_matT = 0;
    size_t nnzTotal_matT = 0;
    for (int i=0; i<numMatrices; i++) {
        numRowsTotal_matT += OARMatricesT[i].getRows();
        nnzTotal_matT += OARMatricesT[i].getNnz();
    }
    EigenIdxType* OARmatT_offsets = (EigenIdxType*)malloc((numRowsTotal_matT+1)*sizeof(EigenIdxType));
    OARmatT_offsets[0] = 0;
    size_t offsetsIdx = 0;
    for (int i=0; i<numMatrices; i++) {
        const MatCSR_Eigen& local_OARMatricesT = OARMatricesT[i];
        EigenIdxType local_numRows = local_OARMatricesT.getRows();
        const EigenIdxType* m_outerIndex = local_OARMatricesT.getOffset();
        for (EigenIdxType j=0; j<local_numRows; j++) {
            OARmatT_offsets[offsetsIdx+1] = OARmatT_offsets[offsetsIdx] +
                m_outerIndex[j+1] - m_outerIndex[j];
            offsetsIdx++;
        }
    }
    std::vector<EigenIdxType> cumuNnz(numMatrices, 0);
    for (int i=0; i<numMatrices-1; i++) {
        cumuNnz[i+1] = cumuNnz[i] + OARMatricesT[i].getNnz();
    }
    EigenIdxType* OARmatT_columns = new EigenIdxType[nnzTotal_matT];
    float* OARmatT_values = new float[nnzTotal_matT];
    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        EigenIdxType nnz_offset = cumuNnz[i];
        const MatCSR_Eigen& localOARMat = OARMatricesT[i];
        EigenIdxType localNnz = localOARMat.getNnz();
        const EigenIdxType* localColumns = localOARMat.getIndices();
        const float* localValues = localOARMat.getValues();
        std::copy(localColumns, localColumns + localNnz, OARmatT_columns + nnz_offset);
        std::copy(localValues, localValues + localNnz, OARmatT_values + nnz_offset);
    }
    
    OARmatT.customInit(numRowsTotal_matT, OARMatricesT[0].getCols(), nnzTotal_matT,
        OARmatT_offsets, OARmatT_columns, OARmatT_values);
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "OARmatT initialization time elapsed: " << duration.count() * 0.001f
            << " [s]" << std::endl;
    #endif

    
    // initialize OARmat
    Eigen::Index OARmat_numRows = OARMatrices[0].getRows();
    size_t OARmat_numCols = numRowsTotal_matT;
    EigenIdxType* m_offsets_OARmat = (EigenIdxType*)malloc((OARmat_numRows+1)*sizeof(EigenIdxType));
    EigenIdxType* m_columns_OARmat = new EigenIdxType[nnzTotal_matT];
    float* m_values_OARmat = new float[nnzTotal_matT];
    // the 2d array to store the starting of every numMatrix at every row of OARmat
    std::vector<std::vector<EigenIdxType>> offsets_copy(numMatrices,
        std::vector<EigenIdxType>(OARmat_numRows, 0));
    EigenIdxType currentOffset = 0;
    m_offsets_OARmat[0] = 0;
    for (EigenIdxType j=0; j<OARmat_numRows; j++) {
        for (EigenIdxType i=0; i<numMatrices; i++) {
            offsets_copy[i][j] = currentOffset;
            const MatCSR_Eigen& localOARMat = OARMatrices[i];
            const EigenIdxType* localOffsets = localOARMat.getOffset();
            EigenIdxType nnzThisRow = localOffsets[j+1] - localOffsets[j];
            currentOffset += nnzThisRow;
        }
        m_offsets_OARmat[j+1] = currentOffset;
    }
    if (currentOffset != nnzTotal_matT) {
        std::cerr << "Error, currentOffset is supposed to be equal to nnz, "
            "but currentOffset = " << currentOffset << ", nnz = "
            << nnzTotal_matT << std::endl;
        return 1;
    }
    std::vector<EigenIdxType> cumuNumCols(numMatrices, 0);
    for (int i=0; i<numMatrices-1; i++) {
        cumuNumCols[i + 1] = cumuNumCols[i] + OARMatrices[i].getCols();
    }

    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        const MatCSR_Eigen& localOARMat = OARMatrices[i];
        const EigenIdxType* localOffsets = localOARMat.getOffset();
        const EigenIdxType* localColumns = localOARMat.getIndices();
        const float* localValues = localOARMat.getValues();
        size_t column_offset = cumuNumCols[i];
        for (int j=0; j<OARmat_numRows; j++) {
            size_t nnzThisRow = localOffsets[j+1] - localOffsets[j];
            size_t startingIndex = offsets_copy[i][j];
            for (size_t k=0; k<nnzThisRow; k++) {
                m_values_OARmat[startingIndex + k] = localValues[localOffsets[j] + k];
                m_columns_OARmat[startingIndex + k] = column_offset
                    + localColumns[localOffsets[j] + k];
            }
        }
    }
    OARmat.customInit(OARmat_numRows, OARmat_numCols, nnzTotal_matT,
        m_offsets_OARmat, m_columns_OARmat, m_values_OARmat);
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
        std::cout << "OARmat initialization time elapsed: " << duration.count() * 0.001f
            << " [s]" << std::endl;
    #endif
    return 0;
}