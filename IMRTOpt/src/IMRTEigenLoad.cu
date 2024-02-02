#include <iomanip>
#include <chrono>
#include <Eigen/Dense>
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

    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    if (Eigen2Cusparse(OARMat, SpOARmat) || Eigen2Cusparse(OARMatT, SpOARmatT)) {
        std::cerr << "Error loading OARmat and OARmatT from CPU to GPU." << std::endl;
    }
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Loading OARMat and OARMatT to device time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

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

bool IMRT::fluenceGradInit(
    MatCSR64& SpFluenceGrad, MatCSR64& SpFluenceGradT,
    const std::string& fluenceMapPath, int fluenceDim
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    size_t numElementsPerBB = fluenceDim * fluenceDim;
    std::ifstream f(fluenceMapPath);
    f.seekg(0, std::ios::end);
    size_t totalNumElements = f.tellg() / sizeof(uint8_t);
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> fluenceArray(totalNumElements);
    f.read((char*)fluenceArray.data(), totalNumElements*sizeof(uint8_t));
    f.close();
    
    if (totalNumElements % numElementsPerBB != 0) {
        std::cerr << "The total number of elements is supposed to be a "
            "multiple of the number of pixels in a beam, which is not satisfied." << std::endl;
        return 1;
    }
    size_t nBeams = totalNumElements / numElementsPerBB;

    MatCSR_Eigen Dxy, Id, IdNbeams, Dx0, Dy0, Dx, Dy;
    DxyInit(Dxy, fluenceDim);
    IdentityInit(Id, fluenceDim);
    IdentityInit(IdNbeams, nBeams);
    KroneckerProduct(Id, Dxy, Dx0);
    KroneckerProduct(Dxy, Id, Dy0);
    KroneckerProduct(IdNbeams, Dx0, Dx);
    KroneckerProduct(IdNbeams, Dy0, Dy);

    // construct filtering matrix
    MatCSR_Eigen BeamletFilter;
    filterConstruction(BeamletFilter, fluenceArray);
    Dx = Dx * BeamletFilter;
    Dy = Dy * BeamletFilter;

    size_t numActiveBeamlets = BeamletFilter.getCols();
    Eigen::VectorXf sumVec(numActiveBeamlets);
    for (size_t i=0; i<numActiveBeamlets; i++)
        sumVec[i] = 1.0f;

    Eigen::VectorXf Dxsum1 = Dx * sumVec;
    Eigen::VectorXf Dxsum2 = Dx.cwiseAbs() * sumVec;
    std::vector<uint8_t> Dx_filter_array(Dxsum1.size());
    for (size_t i=0; i<Dxsum1.size(); i++)
        Dx_filter_array[i] = (std::abs(Dxsum1[i]) < 1e-4f) && (Dxsum2[i] > 1e-4f);
    MatCSR_Eigen Dx_filter;
    filterConstruction(Dx_filter, Dx_filter_array);
    Dx = Dx_filter.transpose() * Dx;

    Eigen::VectorXf Dysum1 = Dy * sumVec;
    Eigen::VectorXf Dysum2 = Dy.cwiseAbs() * sumVec;
    std::vector<uint8_t> Dy_filter_array(Dysum1.size());
    for (size_t i=0; i<Dysum1.size(); i++)
        Dy_filter_array[i] = (std::abs(Dysum1[i]) < 1e-4f) && (Dysum2[i] > 1e-4f);
    MatCSR_Eigen Dy_filter;
    filterConstruction(Dy_filter, Dy_filter_array);
    Dy = Dy_filter.transpose() * Dy;

    // concatenate Dx and Dy
    MatCSR_Eigen Dxy_concat;
    size_t Dxy_numRows = Dx.getRows() + Dy.getRows();
    size_t Dxy_numCols = Dx.getCols();
    size_t Dxy_nnz = Dx.getNnz() + Dy.getNnz();
    EigenIdxType* Dxy_concat_offset = (EigenIdxType*)malloc((Dxy_numRows+1)*sizeof(EigenIdxType));
    EigenIdxType* Dxy_concat_columns = new EigenIdxType[Dxy_nnz];
    float* Dxy_concat_values = new float[Dxy_nnz];
    
    EigenIdxType* Dx_offsets = *Dx.getOffset();
    EigenIdxType* Dy_offsets = *Dy.getOffset();
    std::copy(Dx_offsets, Dx_offsets + Dx.getRows() + 1, Dxy_concat_offset);
    size_t offsets_offset = Dxy_concat_offset[Dx.getRows()];
    for (size_t i=0; i<Dy.getRows(); i++) {
        Dxy_concat_offset[Dx.getRows() + i + 1] = offsets_offset + Dy_offsets[i + 1];
    }

    const EigenIdxType* Dx_columns = Dx.getIndices();
    const EigenIdxType* Dy_columns = Dy.getIndices();
    std::copy(Dx_columns, Dx_columns + Dx.getNnz(), Dxy_concat_columns);
    std::copy(Dy_columns, Dy_columns + Dy.getNnz(), Dxy_concat_columns + Dx.getNnz());

    const float* Dx_values = Dx.getValues();
    const float* Dy_values = Dy.getValues();
    std::copy(Dx_values, Dx_values + Dx.getNnz(), Dxy_concat_values);
    std::copy(Dy_values, Dy_values + Dy.getNnz(), Dxy_concat_values + Dx.getNnz());

    Dxy_concat.customInit(Dxy_numRows, Dxy_numCols, Dxy_nnz,
        Dxy_concat_offset, Dxy_concat_columns, Dxy_concat_values);


    // load to GPU
    if (Eigen2Cusparse(Dxy_concat, SpFluenceGrad)) {
        std::cerr << "Error loading the fluence gradient operator to GPU." << std::endl;
        return 1;
    }
    MatCSR_Eigen Dxy_concatT = Dxy_concat.transpose();
    if (Eigen2Cusparse(Dxy_concatT, SpFluenceGradT)) {
        std::cerr << "Error loading the transpose fluence gradient "
            "operator to GPU." << std::endl;
        return 1;
    }
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Fluence gradient operator and its transpose construction time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif
    
    return 0;
}


bool IMRT::Eigen2Cusparse(const MatCSR_Eigen& source, MatCSR64& dest) {
    // here we assume the destination matrix is an empty one.
    if (dest.matA != nullptr || dest.d_csr_offsets != nullptr ||
        dest.d_csr_columns != nullptr || dest.d_csr_values != nullptr ||
        dest.d_buffer_spmv != nullptr) {
        std::cerr << "The destination matrix is not empty." << std::endl;
        return 1;
    }

    checkCudaErrors(cudaMalloc((void**)&dest.d_csr_offsets, (source.getRows() + 1) * sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&dest.d_csr_columns, source.getNnz() * sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&dest.d_csr_values, source.getNnz() * sizeof(float)));

    checkCudaErrors(cudaMemcpy(dest.d_csr_offsets, source.getOffset(),
        (source.getRows() + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dest.d_csr_columns, source.getIndices(),
        source.getNnz() * sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dest.d_csr_values, source.getValues(),
        source.getNnz() * sizeof(float), cudaMemcpyHostToDevice));
    
    checkCusparse(cusparseCreateCsr(
        &dest.matA, source.getRows(), source.getCols(), source.getNnz(),
        dest.d_csr_offsets, dest.d_csr_columns, dest.d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    dest.numRows = source.getRows();
    dest.numCols = source.getCols();
    dest.nnz = source.getNnz();
    return 0;
}