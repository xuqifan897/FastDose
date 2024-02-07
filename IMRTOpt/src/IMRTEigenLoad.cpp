#include <iostream>
#include <iomanip>
#include <omp.h>
#include <bitset>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include "IMRTDoseMatEigen.cuh"

bool IMRT::parallelSpGEMM(
    const std::string& resultFolder,
    const MatCSR_Eigen& filter,
    const MatCSR_Eigen& filterT,
    std::vector<MatCSR_Eigen>& VOIMatrices,
    std::vector<MatCSR_Eigen>& VOIMatricesT
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

    VOIMatrices.resize(numMatrices);
    VOIMatricesT.resize(numMatrices);
    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        VOIMatricesT[i] = matricesT[i] * filter;
        VOIMatrices[i] = VOIMatricesT[i].transpose();
    }
    #if slicingTiming
        auto time3 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2);
        std::cout << "VOI dose loading matrices and their transpose construction time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    #if false
        test_parallelSpGEMM(VOIMatrices, VOIMatricesT, matricesT, filter);
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
    MatCSR_Eigen& VOImat, MatCSR_Eigen& VOImatT,
    const std::vector<MatCSR_Eigen>& VOIMatrices,
    const std::vector<MatCSR_Eigen>& VOIMatricesT
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    // initialize VOImatT
    int numMatrices = VOIMatricesT.size();
    size_t numRowsTotal_matT = 0;
    size_t nnzTotal_matT = 0;
    for (int i=0; i<numMatrices; i++) {
        numRowsTotal_matT += VOIMatricesT[i].getRows();
        nnzTotal_matT += VOIMatricesT[i].getNnz();
    }
    EigenIdxType* VOImatT_offsets = (EigenIdxType*)malloc((numRowsTotal_matT+1)*sizeof(EigenIdxType));
    VOImatT_offsets[0] = 0;
    size_t offsetsIdx = 0;
    for (int i=0; i<numMatrices; i++) {
        const MatCSR_Eigen& local_VOIMatricesT = VOIMatricesT[i];
        EigenIdxType local_numRows = local_VOIMatricesT.getRows();
        const EigenIdxType* m_outerIndex = local_VOIMatricesT.getOffset();
        for (EigenIdxType j=0; j<local_numRows; j++) {
            VOImatT_offsets[offsetsIdx+1] = VOImatT_offsets[offsetsIdx] +
                m_outerIndex[j+1] - m_outerIndex[j];
            offsetsIdx++;
        }
    }
    std::vector<EigenIdxType> cumuNnz(numMatrices, 0);
    for (int i=0; i<numMatrices-1; i++) {
        cumuNnz[i+1] = cumuNnz[i] + VOIMatricesT[i].getNnz();
    }
    EigenIdxType* VOImatT_columns = new EigenIdxType[nnzTotal_matT];
    float* VOImatT_values = new float[nnzTotal_matT];
    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        EigenIdxType nnz_offset = cumuNnz[i];
        const MatCSR_Eigen& localVOIMat = VOIMatricesT[i];
        EigenIdxType localNnz = localVOIMat.getNnz();
        const EigenIdxType* localColumns = localVOIMat.getIndices();
        const float* localValues = localVOIMat.getValues();
        std::copy(localColumns, localColumns + localNnz, VOImatT_columns + nnz_offset);
        std::copy(localValues, localValues + localNnz, VOImatT_values + nnz_offset);
    }
    
    VOImatT.customInit(numRowsTotal_matT, VOIMatricesT[0].getCols(), nnzTotal_matT,
        VOImatT_offsets, VOImatT_columns, VOImatT_values);
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "VOImatT initialization time elapsed: " << duration.count() * 0.001f
            << " [s]" << std::endl;
    #endif

    
    // initialize VOImat
    Eigen::Index VOImat_numRows = VOIMatrices[0].getRows();
    size_t VOImat_numCols = numRowsTotal_matT;
    EigenIdxType* m_offsets_VOImat = (EigenIdxType*)malloc((VOImat_numRows+1)*sizeof(EigenIdxType));
    EigenIdxType* m_columns_VOImat = new EigenIdxType[nnzTotal_matT];
    float* m_values_VOImat = new float[nnzTotal_matT];
    // the 2d array to store the starting of every numMatrix at every row of VOImat
    std::vector<std::vector<EigenIdxType>> offsets_copy(numMatrices,
        std::vector<EigenIdxType>(VOImat_numRows, 0));
    EigenIdxType currentOffset = 0;
    m_offsets_VOImat[0] = 0;
    for (EigenIdxType j=0; j<VOImat_numRows; j++) {
        for (EigenIdxType i=0; i<numMatrices; i++) {
            offsets_copy[i][j] = currentOffset;
            const MatCSR_Eigen& localVOIMat = VOIMatrices[i];
            const EigenIdxType* localOffsets = localVOIMat.getOffset();
            EigenIdxType nnzThisRow = localOffsets[j+1] - localOffsets[j];
            currentOffset += nnzThisRow;
        }
        m_offsets_VOImat[j+1] = currentOffset;
    }
    if (currentOffset != nnzTotal_matT) {
        std::cerr << "Error, currentOffset is supposed to be equal to nnz, "
            "but currentOffset = " << currentOffset << ", nnz = "
            << nnzTotal_matT << std::endl;
        return 1;
    }
    std::vector<EigenIdxType> cumuNumCols(numMatrices, 0);
    for (int i=0; i<numMatrices-1; i++) {
        cumuNumCols[i + 1] = cumuNumCols[i] + VOIMatrices[i].getCols();
    }

    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        const MatCSR_Eigen& localVOIMat = VOIMatrices[i];
        const EigenIdxType* localOffsets = localVOIMat.getOffset();
        const EigenIdxType* localColumns = localVOIMat.getIndices();
        const float* localValues = localVOIMat.getValues();
        size_t column_offset = cumuNumCols[i];
        for (int j=0; j<VOImat_numRows; j++) {
            size_t nnzThisRow = localOffsets[j+1] - localOffsets[j];
            size_t startingIndex = offsets_copy[i][j];
            for (size_t k=0; k<nnzThisRow; k++) {
                m_values_VOImat[startingIndex + k] = localValues[localOffsets[j] + k];
                m_columns_VOImat[startingIndex + k] = column_offset
                    + localColumns[localOffsets[j] + k];
            }
        }
    }
    VOImat.customInit(VOImat_numRows, VOImat_numCols, nnzTotal_matT,
        m_offsets_VOImat, m_columns_VOImat, m_values_VOImat);
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
        std::cout << "VOImat initialization time elapsed: " << duration.count() * 0.001f
            << " [s]" << std::endl;
    #endif
    return 0;
}


bool IMRT::DxyInit(IMRT::MatCSR_Eigen& Dxy, size_t size) {
    size_t Dxy_nnz = 2 * (size - 1);
    EigenIdxType* Dxy_offsets = (EigenIdxType*)malloc((size+1)*sizeof(EigenIdxType));
    EigenIdxType* Dxy_columns = new EigenIdxType[Dxy_nnz];
    float* Dxy_values = new float[Dxy_nnz];

    for (size_t i=0; i<size; i++) {
        Dxy_offsets[i] = 2 * i;
    }
    Dxy_offsets[size] = Dxy_offsets[size-1];    

    for (size_t i=0; i<size-1; i++) {
        Dxy_columns[2 * i] = i;
        Dxy_columns[2 * i + 1] = i + 1;
        Dxy_values[2 * i] = -1.0f;
        Dxy_values[2 * i + 1] = 1.0f;
    }
    Dxy.customInit(size, size, Dxy_nnz,
        Dxy_offsets, Dxy_columns, Dxy_values);
    return 0;
}

bool IMRT::IdentityInit(IMRT::MatCSR_Eigen& Id, size_t size) {
    size_t Id_nnz = size;
    EigenIdxType* Id_offsets = (EigenIdxType*)malloc((size+1)*sizeof(EigenIdxType));
    EigenIdxType* Id_columns = new EigenIdxType[Id_nnz];
    float* Id_values = new float[Id_nnz];
    for (size_t i=0; i<Id_nnz; i++) {
        Id_offsets[i] = i;
        Id_columns[i] = i;
        Id_values[i] = 1.0f;
    }
    Id_offsets[size] = Id_nnz;
    Id.customInit(size, size, size,
        Id_offsets, Id_columns, Id_values);
    return 0;
}


bool IMRT::KroneckerProduct(const MatCSR_Eigen& A,
    const MatCSR_Eigen& B, MatCSR_Eigen& C
) {
    size_t C_nnz = A.getNnz() * B.getNnz();
    size_t C_rows = A.getRows() * B.getRows();
    size_t C_cols = A.getCols() * B.getCols();

    EigenIdxType* C_offsets = (EigenIdxType*)malloc((C_rows + 1) * sizeof(EigenIdxType));
    C_offsets[0] = 0;
    EigenIdxType* C_columns = new EigenIdxType[C_nnz];
    float* C_values = new float[C_nnz];

    const EigenIdxType* A_offsets = A.getOffset();
    const EigenIdxType* A_columns = A.getIndices();
    const float* A_values = A.getValues();

    const EigenIdxType* B_offsets = B.getOffset();
    const EigenIdxType* B_columns = B.getIndices();
    const float* B_values = B.getValues();

    for (size_t A_row=0; A_row<A.getRows(); A_row++) {
        size_t row_base = A_row * B.getRows();
        size_t nnz_base = A_offsets[A_row] * B.getNnz();
        
        size_t A_idx_start = A_offsets[A_row];
        size_t A_idx_end = A_offsets[A_row + 1];
        size_t A_nnz_this_row = A_idx_end - A_idx_start;

        for (size_t B_row=0; B_row<B.getRows(); B_row++) {
            size_t B_idx_start = B_offsets[B_row];
            size_t B_idx_end = B_offsets[B_row + 1];
            size_t B_nnz_this_row = B_idx_end - B_idx_start;

            size_t C_nnz_this_row = A_nnz_this_row * B_nnz_this_row;
            size_t C_row = row_base + B_row;
            C_offsets[C_row + 1] = C_offsets[C_row] + C_nnz_this_row;

            for (size_t A_idx=A_idx_start; A_idx<A_idx_end; A_idx++) {
                EigenIdxType A_col = A_columns[A_idx];
                float A_val = A_values[A_idx];

                EigenIdxType C_col_base = A_col * B.getCols();

                for (size_t B_idx=B_idx_start; B_idx<B_idx_end; B_idx++) {
                    EigenIdxType B_col = B_columns[B_idx];
                    float B_val = B_values[B_idx];

                    EigenIdxType C_col = C_col_base + B_col;
                    float C_val = A_val * B_val;
                    C_columns[nnz_base] = C_col;
                    C_values[nnz_base] = C_val;
                    nnz_base ++;
                }
            }
        }
    }

    C.customInit(C_rows, C_cols, C_nnz,
        C_offsets, C_columns, C_values);
    return 0;
}


bool IMRT::test_KroneckerProduct() {
    MatCSR_Eigen A, B, AkB, BkA;

    EigenIdxType* A_offset = (EigenIdxType*)malloc(4*sizeof(EigenIdxType));
    EigenIdxType* A_columns = new EigenIdxType[3];
    float* A_values = new float[3];
    A_offset[0] = 0; A_offset[1] = 1; A_offset[2] = 2; A_offset[3] = 3;
    A_columns[0] = 1; A_columns[1] = 2; A_columns[2] = 0;
    A_values[0] = 2.0f; A_values[1] = 1.0f; A_values[2] = 3.0f; 
    A.customInit(3, 3, 3, A_offset, A_columns, A_values);

    EigenIdxType* B_offset = (EigenIdxType*)malloc(5*sizeof(EigenIdxType));
    EigenIdxType* B_columns = new EigenIdxType[4];
    float* B_values = new float[4];
    B_offset[0] = 0; B_offset[1] = 1; B_offset[2] = 2; B_offset[3] = 3; B_offset[4] = 4;
    B_columns[0] = 3; B_columns[1] = 2; B_columns[2] = 1; B_columns[3] = 0;
    B_values[0] = 1.0f; B_values[1] = 1.0f; B_values[2] = 1.0f; B_values[3] = 1.0f;
    B.customInit(4, 4, 4, B_offset, B_columns, B_values);

    KroneckerProduct(A, B, AkB);
    KroneckerProduct(B, A, BkA);

    std::cout << std::fixed << std::setprecision(0) << "A:" << std::endl << A << std::endl;
    std::cout << "B:" << std::endl << B << std::endl;
    std::cout << "Kronecker(A, B):" << std::endl << AkB << std::endl;
    std::cout << "Kronecker(B, A):" << std::endl << BkA << std::endl;
    return 0;
}


bool IMRT::filterConstruction(MatCSR_Eigen& filter, const std::vector<uint8_t>& array) {
    size_t totalNumElements = array.size();
    size_t BeamletFilter_nnz = 0;
    for (size_t i=0; i<totalNumElements; i++)
        BeamletFilter_nnz += (array[i] > 0);
    EigenIdxType* BeamletFilter_offsets = (EigenIdxType*)malloc(
        (totalNumElements+1)*sizeof(EigenIdxType));
    BeamletFilter_offsets[0] = 0;
    EigenIdxType* BeamletFilter_columns = new EigenIdxType[BeamletFilter_nnz];
    float* BeamletFilter_values = new float[BeamletFilter_nnz];
    
    size_t BeamletFilterIdx = 0;
    for (size_t i=0; i<totalNumElements; i++) {
        if(array[i] > 0) {
            BeamletFilter_columns[BeamletFilterIdx] = BeamletFilterIdx;
            BeamletFilter_values[BeamletFilterIdx] = 1.0f;
            BeamletFilterIdx ++;
        }
        BeamletFilter_offsets[i+1] = BeamletFilterIdx;
    }
    filter.customInit(totalNumElements, BeamletFilter_nnz, BeamletFilter_nnz,
        BeamletFilter_offsets, BeamletFilter_columns, BeamletFilter_values);
    return 0;
}

bool IMRT::test_filterConstruction() {
    std::vector<uint8_t> array{1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1};
    MatCSR_Eigen filter;
    filterConstruction(filter, array);
    std::cout << std::fixed << std::setprecision(0) << filter << std::endl;
    return 0;
}