#include <fstream>
#include <chrono>
#include "IMRTDoseMatEigen.cuh"
#include "IMRTArgs.h"


bool IMRT::fluenceGradInit(
    std::vector<IMRT::MatCSR_Eigen>& SpFluenceGrad,
    std::vector<IMRT::MatCSR_Eigen>& SpFluenceGradT,
    std::vector<uint8_t>& fluenceArray,
    const std::string& fluenceMapPath
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    int fluenceDim = getarg<int>("fluenceDim");
    size_t numElementsPerBB = fluenceDim * fluenceDim;
    std::ifstream f(fluenceMapPath);
    f.seekg(0, std::ios::end);
    size_t totalNumElements = f.tellg() / sizeof(uint8_t);
    f.seekg(0, std::ios::beg);
    fluenceArray.resize(totalNumElements);
    f.read((char*)fluenceArray.data(), totalNumElements*sizeof(uint8_t));
    f.close();

    if (totalNumElements % numElementsPerBB != 0) {
        std::cerr << "The total number of elements is supposed to be a "
            "multiple of the number of pixels in a beam, which is not satisfied." << std::endl;
        return 1;
    }
    size_t nBeams = totalNumElements / numElementsPerBB;

    MatCSR_Eigen Dxy, Id, Dx0, Dy0;
    DxyInit(Dxy, fluenceDim);
    IdentityInit(Id, fluenceDim);
    KroneckerProduct(Id, Dxy, Dx0);
    KroneckerProduct(Dxy, Id, Dy0);

    SpFluenceGrad.resize(nBeams);
    SpFluenceGradT.resize(nBeams);
    #pragma omp parallel for
    for (int i=0; i<nBeams; i++) {
        MatCSR_Eigen BeamletFilter;
        std::vector<uint8_t> localFluenceArray(numElementsPerBB);
        std::copy(&fluenceArray[numElementsPerBB * i],
            &fluenceArray[numElementsPerBB * (i + 1)], localFluenceArray.begin());
        filterConstruction(BeamletFilter, localFluenceArray);

        MatCSR_Eigen Dx_local = Dx0 * BeamletFilter;
        MatCSR_Eigen Dy_local = Dy0 * BeamletFilter;
        size_t numActiveBeamlets = BeamletFilter.getCols();
        Eigen::VectorXf sumVec(numActiveBeamlets);
        for (size_t j=0; j<numActiveBeamlets; j++)
            sumVec[j] = 1.0f;
        
        Eigen::VectorXf Dxsum1 = Dx_local * sumVec;
        Eigen::VectorXf Dxsum2 = Dx_local.cwiseAbs() * sumVec;
        std::vector<uint8_t> Dx_filter_array(Dxsum1.size());
        for (size_t j=0; j<Dxsum1.size(); j++)
            Dx_filter_array[j] = (std::abs(Dxsum1[i] < 1e-4f)) && (Dxsum2[i] > 1e-4f);
        MatCSR_Eigen Dx_filter;
        filterConstruction(Dx_filter, Dx_filter_array);
        Dx_local = Dx_filter.transpose() * Dx_local;

        Eigen::VectorXf Dysum1 = Dy_local * sumVec;
        Eigen::VectorXf Dysum2 = Dy_local.cwiseAbs() * sumVec;
        std::vector<uint8_t> Dy_filter_array(Dysum1.size());
        for (size_t j=0; j<Dysum1.size(); j++)
            Dy_filter_array[j] = (std::abs(Dysum1[i]) < 1e-4f) && (Dysum2[i] > 1e-4f);
        MatCSR_Eigen Dy_filter;
        filterConstruction(Dy_filter, Dy_filter_array);
        Dy_local = Dy_filter.transpose() * Dy_local;

        // concatenate Dx and Dy
        MatCSR_Eigen Dxy_concat;
        size_t Dxy_numRows = Dx_local.getRows() + Dy_local.getRows();
        size_t Dxy_numCols = Dx_local.getCols();
        size_t Dxy_nnz = Dx_local.getNnz() + Dy_local.getNnz();
        EigenIdxType* Dxy_concat_offset = (EigenIdxType*)malloc((Dxy_numRows+1)*sizeof(EigenIdxType));
        EigenIdxType* Dxy_concat_columns = new EigenIdxType[Dxy_nnz];
        float* Dxy_concat_values = new float[Dxy_nnz];

        EigenIdxType* Dx_offsets = *Dx_local.getOffset();
        EigenIdxType* Dy_offsets = *Dy_local.getOffset();
        std::copy(Dx_offsets, Dx_offsets + Dx_local.getRows() + 1, Dxy_concat_offset);
        size_t offsets_offset = Dxy_concat_offset[Dx_local.getRows()];
        for (size_t i=0; i<Dy_local.getRows(); i++) {
            Dxy_concat_offset[Dx_local.getRows() + i + 1] = offsets_offset + Dy_offsets[i + 1];
        }

        const EigenIdxType* Dx_columns = Dx_local.getIndices();
        const EigenIdxType* Dy_columns = Dy_local.getIndices();
        std::copy(Dx_columns, Dx_columns + Dx_local.getNnz(), Dxy_concat_columns);
        std::copy(Dy_columns, Dy_columns + Dy_local.getNnz(), Dxy_concat_columns + Dx_local.getNnz());
    
        const float* Dx_values = Dx_local.getValues();
        const float* Dy_values = Dy_local.getValues();
        std::copy(Dx_values, Dx_values + Dx_local.getNnz(), Dxy_concat_values);
        std::copy(Dy_values, Dy_values + Dy_local.getNnz(), Dxy_concat_values + Dx_local.getNnz());

        Dxy_concat.customInit(Dxy_numRows, Dxy_numCols, Dxy_nnz,
            Dxy_concat_offset, Dxy_concat_columns, Dxy_concat_values);
        
        SpFluenceGrad[i] = Dxy_concat;
        SpFluenceGradT[i] = Dxy_concat.transpose();
    }

    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Fluence gradient operator and its transpose construction time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    return 0;
}


bool IMRT::matFuseFunc(
    std::vector<MatCSR_Eigen*>& VOIMatrices,
    std::vector<MatCSR_Eigen*>& VOIMatricesT,
    std::vector<MatCSR_Eigen*>& SpFluenceGrad,
    std::vector<MatCSR_Eigen*>& SpFluenceGradT,
    MatCSR_Eigen& VOIMat_Eigen,
    MatCSR_Eigen& VOIMatT_Eigen,
    MatCSR_Eigen& D_Eigen,
    MatCSR_Eigen& DTrans_Eigen
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    // initialize VOIMatT_Eigen
    int numMatrices = VOIMatricesT.size();
    size_t numRowsTotal_matT = 0;
    size_t nnzTotal_matT = 0;
    for (int i=0; i<numMatrices; i++) {
        numRowsTotal_matT += VOIMatricesT[i]->getRows();
        nnzTotal_matT += VOIMatricesT[i]->getNnz();
    }
    EigenIdxType* VOImatT_offsets = (EigenIdxType*)malloc((numRowsTotal_matT + 1)
        * sizeof(EigenIdxType));
    VOImatT_offsets[0] = 0;
    size_t offsetsIdx = 0;
    for (int i=0; i<numMatrices; i++) {
        const MatCSR_Eigen& local_VOIMatricesT = *VOIMatricesT[i];
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
        cumuNnz[i+1] = cumuNnz[i] + VOIMatricesT[i]->getNnz();
    }
    EigenIdxType* VOImatT_columns = new EigenIdxType[nnzTotal_matT];
    float* VOImatT_values = new float[nnzTotal_matT];
    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        EigenIdxType nnz_offset = cumuNnz[i];
        const MatCSR_Eigen& localVOIMatT = *VOIMatricesT[i];
        EigenIdxType localNnz = localVOIMatT.getNnz();
        const EigenIdxType* localColumns = localVOIMatT.getIndices();
        const float* localValues = localVOIMatT.getValues();
        std::copy(localColumns, localColumns + localNnz, VOImatT_columns + nnz_offset);
        std::copy(localValues, localValues + localNnz, VOImatT_values + nnz_offset);
    }
    VOIMatT_Eigen.customInit(numRowsTotal_matT, VOIMatricesT[0]->getCols(), nnzTotal_matT,
        VOImatT_offsets, VOImatT_columns, VOImatT_values);
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "VOImatT initialization time elapsed: " << duration.count() * 0.001f
            << " [s]" << std::endl;
    #endif


    // initialize VOImat
    Eigen::Index VOImat_numRows = VOIMatrices[0]->getRows();
    size_t VOImat_numCols = numRowsTotal_matT;
    EigenIdxType* m_offsets_VOImat = (EigenIdxType*)malloc((VOImat_numRows+1)*sizeof(EigenIdxType));
    EigenIdxType* m_columns_VOImat = new EigenIdxType[nnzTotal_matT];
    float* m_values_VOImat = new float[nnzTotal_matT];

    m_offsets_VOImat[0] = 0;
    #pragma omp parallel for
    for (size_t row=0; row<VOImat_numRows; row++) {
        m_offsets_VOImat[row + 1] = 0;
        for (size_t i=0; i<numMatrices; i++) {
            EigenIdxType* localOffsets = *(VOIMatrices[i]->getOffset());
            m_offsets_VOImat[row + 1] += localOffsets[row + 1] - localOffsets[row];
        }
    }

    for (size_t row=0; row<VOImat_numRows; row++) {
        m_offsets_VOImat[row + 1] += m_offsets_VOImat[row];
    }

    #pragma omp parallel for
    for (size_t row=0; row<VOImat_numRows; row++) {
        size_t dest_idx = m_offsets_VOImat[row];
        size_t column_offset = 0;
        for (int i=0; i<numMatrices; i++) {
            EigenIdxType* localOffsets = *(VOIMatrices[i]->getOffset());
            const EigenIdxType* localColumns = VOIMatrices[i]->getIndices();
            const float* localValues = VOIMatrices[i]->getValues();

            EigenIdxType source_idx_begin = localOffsets[row];
            EigenIdxType source_idx_end = localOffsets[row + 1];
            for (size_t source_idx=source_idx_begin; source_idx<source_idx_end; source_idx++) {
                m_columns_VOImat[dest_idx] = localColumns[source_idx] + column_offset;
                m_values_VOImat[dest_idx] = localValues[source_idx];
                dest_idx ++;
            }
            column_offset += VOIMatrices[i]->getCols();
        }
    }

    VOIMat_Eigen.customInit(VOImat_numRows, VOImat_numCols, nnzTotal_matT,
        m_offsets_VOImat, m_columns_VOImat, m_values_VOImat);
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
        std::cout << "VOImat initialization time elapsed: " << duration.count() * 0.001f
            << " [s]" << std::endl;
    #endif

    #if false
        if(test_matFuseFunc(VOIMat_Eigen, VOIMatT_Eigen)) {
            std::cerr << "VOIMat_Eigen and VOIMatT_Eigen didn't pass the test." << std::endl;
            return 1;
        } else {
            std::cout << "VOIMat_Eigen and VOIMatT_Eigen passed the test!" << std::endl;
        }
    #endif

    diagBlock(D_Eigen, SpFluenceGrad);
    diagBlock(DTrans_Eigen, SpFluenceGradT);

    #if slicingTiming
        auto time3 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2);
        std::cout << "Fluence grad and its transpose time elapsed: " << duration.count() * 0.001f
            << " [s]" << std::endl;
    #endif
    return 0;
}


bool IMRT::diagBlock(MatCSR_Eigen& target, const std::vector<MatCSR_Eigen*>& source)
{
    // calculate the number of rows and the number of non-zero elements
    size_t target_numRows = 0;
    size_t target_numCols = 0;
    size_t target_nnz = 0;
    for (MatCSR_Eigen* mat : source) {
        target_numRows += mat->getRows();
        target_numCols += mat->getCols();
        target_nnz += mat->getNnz();
    }
    EigenIdxType* target_offsets = (EigenIdxType*)malloc(
        (target_numRows+1)*sizeof(EigenIdxType));
    EigenIdxType* target_indices = new EigenIdxType[target_nnz];
    float* target_values = new float[target_nnz];
    
    target_offsets[0] = 0;
    EigenIdxType row_offset = 0;
    EigenIdxType col_offset = 0;
    EigenIdxType nnz_offset = 0;
    for (MatCSR_Eigen* mat : source) {
        // update target_offsets
        EigenIdxType* local_offsets = *mat->getOffset();
        const EigenIdxType* local_columns = mat->getIndices();
        const float* local_values = mat->getValues();
        for (size_t i=0; i<mat->getRows(); i++) {
            target_offsets[row_offset + 1] = target_offsets[row_offset]
                + local_offsets[i + 1] - local_offsets[i];
            row_offset ++;
        }

        // update columns and values
        for (size_t i=0; i<mat->getNnz(); i++) {
            target_indices[nnz_offset] = col_offset + local_columns[i];
            target_values[nnz_offset] = local_values[i];
            nnz_offset ++;
        }
        col_offset += mat->getCols();
    }
    target.customInit(target_numRows, target_numCols, target_nnz,
        target_offsets, target_indices, target_values);
    return 0;
}