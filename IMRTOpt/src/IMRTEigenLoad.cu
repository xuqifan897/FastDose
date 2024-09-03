#include <iomanip>
#include <chrono>
#include <Eigen/Dense>
#include "IMRTDoseMatEigen.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTInit.cuh"
#include "IMRTOptimize.cuh"
#include "boost/filesystem.hpp"
namespace fs = boost::filesystem;

bool IMRT::OARFiltering(
    const std::string& resultFolder,
    const std::vector<StructInfo>& structs,
    MatCSR64& SpVOIMat, MatCSR64& SpVOIMatT,
    Weights_h& weights, Weights_d& weights_d
) {
    MatCSR_Eigen filter, filterT;
    if (getStructFilter(filter, filterT, structs, weights)) {
        std::cerr << "OAR filter and its transpose construction error." << std::endl;
        return 1;
    }
    weights_d.fromHost(weights);

    size_t totalNumMatrices = 0;
    fs::path NonZeroElementsFile = fs::path(resultFolder) / "NonZeroElements.bin";
    std::ifstream f(NonZeroElementsFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << NonZeroElementsFile << std::endl;
        return 1;
    }
    f.seekg(0, std::ios::end);
    totalNumMatrices = f.tellg() / sizeof(size_t);
    f.close();

    std::vector<MatCSR_Eigen> MatricesT_full(totalNumMatrices);
    std::vector<MatCSR_Eigen> VOIMatrices(totalNumMatrices);
    std::vector<MatCSR_Eigen> VOIMatricesT(totalNumMatrices);

    std::vector<MatCSR_Eigen*> MatricesT_full_t(totalNumMatrices, nullptr);
    std::vector<MatCSR_Eigen*> VOIMatrices_t(totalNumMatrices, nullptr);
    std::vector<MatCSR_Eigen*> VOIMatricesT_t(totalNumMatrices, nullptr);
    for (int i=0; i<totalNumMatrices; i++) {
        MatricesT_full_t[i] = MatricesT_full.data() + i;
        VOIMatrices_t[i] = VOIMatrices.data() + i;
        VOIMatricesT_t[i] = VOIMatricesT.data() + i;
    }
    if (parallelSpGEMM(resultFolder, filter, filterT,
        MatricesT_full_t, VOIMatrices_t, VOIMatricesT_t)) {
        std::cerr << "CPU VOI dose loading matrices and their transpose "
            "construction error." << std::endl;
        return 1;
    }

    std::vector<const MatCSR_Eigen*> VOIMatrices_ptr(VOIMatrices.size(), nullptr);
    std::vector<const MatCSR_Eigen*> VOIMatricesT_ptr(VOIMatricesT.size(), nullptr);
    for (size_t i=0; i<VOIMatrices.size(); i++) {
        VOIMatrices_ptr[i] = & VOIMatrices[i];
        VOIMatricesT_ptr[i] = & VOIMatricesT[i];
    }
    MatCSR_Eigen VOIMat, VOIMatT;
    if (parallelMatCoalesce(VOIMat, VOIMatT, VOIMatrices_ptr, VOIMatricesT_ptr)) {
        std::cerr << "GPU VOI dose loading matrix and its transpose "
            "construction error." << std::endl;
        return 1;
    }

    #if false
        if(test_VOIMat_VOIMatT(VOIMat, VOIMatT)) {
            std::cerr << "VOIMat and VOIMatT test error." << std::endl;
            return 1;
        } else {
            std::cout << "VOIMat and VOIMatT match each other, test passed!" << std::endl;
        }
    #endif

    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    if (Eigen2Cusparse(VOIMat, SpVOIMat) || Eigen2Cusparse(VOIMatT, SpVOIMatT)) {
        std::cerr << "Error loading VOIMat and VOIMatT from CPU to GPU." << std::endl;
    }
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Loading VOIMat and VOIMatT to device time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    #if false
        if(test_SpMatOAR(SpVOIMat, SpVOIMatT, filter, VOIMatrices)) {
            std::cerr << "SpVOIMat and SpVOIMatT validation error." << std::endl;
            return 1;
        }
    #endif

    return 0;
}


bool IMRT::getStructFilter (
    MatCSR_Eigen& filter, MatCSR_Eigen& filterT,
    const std::vector<StructInfo>& structs, Weights_h& weights,
    const std::vector<float>* referenceDose
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif

    std::vector<size_t> nonZeroVoxels;
    size_t totalCount = 0;
    size_t nVoxels = 0;
    size_t PTV_voxels = 0;
    size_t OAR_voxels = 0;
                        // structure, isPTV, nVoxels
    std::vector<std::tuple<StructInfo, bool, size_t>> structs_valid;
    for (int i=0; i<structs.size(); i++) {
        const StructInfo& currentStruct = structs[i];
        if (currentStruct.maxWeights < eps_fastdose &&
            currentStruct.minDoseTargetWeights < eps_fastdose &&
            currentStruct.OARWeights < eps_fastdose) {
            std::cout << "Structure: " << currentStruct.name
                << " is irrelevant in the optimization, skip." << std::endl;
            continue;
        }
        structs_valid.push_back(std::tuple<StructInfo, bool, size_t>(currentStruct, false, 0));
        auto & lastEntry = structs_valid.back();
        std::get<1>(lastEntry) = containsCaseInsensitive(
            currentStruct.name, std::string("PTV"));

        size_t localCount = 0;
        if (nVoxels == 0)
            nVoxels = currentStruct.size.x
                * currentStruct.size.y * currentStruct.size.z;
        else if (nVoxels != currentStruct.size.x
            * currentStruct.size.y * currentStruct.size.z) {
            std::cerr << "Number of voxels inconsistent among structures. idx = "
                << i << ", name = " << currentStruct.name << std::endl;
            return 1;
        }
        for (size_t j=0; j<nVoxels; j++)
            localCount += (currentStruct.mask[j] > 0);
        nonZeroVoxels.push_back(localCount);
        totalCount += localCount;

        std::get<2>(lastEntry) = localCount;
        if (std::get<1>(lastEntry))
            PTV_voxels += localCount;
        else
            OAR_voxels += localCount;
    }
    // sort VOIs to make PTVs before OARs
    std::sort(structs_valid.begin(), structs_valid.end(), structComp);
    for (int i=0; i<structs_valid.size(); i++) {
        const auto& currentTuple = structs_valid[i];
        std::cout << "Structure: " << std::get<0>(currentTuple).name << ", is PTV? "
            << std::get<1>(currentTuple) << ", non-zero voxels: "
            << std::get<2>(currentTuple) << std::endl;
    }
    // update structs
    std::vector<StructInfo> structs_sorted;
    for (int i=0; i<structs_valid.size(); i++)
        structs_sorted.push_back(std::get<0>(structs_valid[i]));
    std::cout << "Total number of non-zero voxels: " << totalCount << std::endl << std::endl;

    EigenIdxType* h_filterOffsets = (EigenIdxType*)malloc((totalCount+1)*sizeof(EigenIdxType));
    EigenIdxType* h_filterColumns = new EigenIdxType[totalCount];
    float* h_filterValues = new float[totalCount];
    for (size_t i=0; i<totalCount; i++)
        h_filterValues[i] = 1.0f;

    size_t idx = 0;
    for (int i=0; i<structs_sorted.size(); i++) {
        const StructInfo& currentStruct = structs_sorted[i];
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


    // weights initialization
    weights.voxels_PTV = PTV_voxels;
    weights.voxels_OAR = OAR_voxels;
    weights.maxDose.resize(totalCount);  // PTV and OAR
    weights.maxWeightsLong.resize(totalCount);  // PTV and OAR
    weights.minDoseTarget.resize(PTV_voxels);  // PTV only
    weights.minDoseTargetWeights.resize(PTV_voxels);  // PTV only
    weights.OARWeightsLong.resize(OAR_voxels);  // OAR only
    size_t offset0 = 0;  // account for the former 4
    size_t offset1 = 0;  // account for OARWeightsLong
    for (int i=0; i<structs_valid.size(); i++) {
        const auto & currentEntry = structs_valid[i];
        size_t current_num_voxels = std::get<2>(currentEntry);
        const StructInfo currentStruct = std::get<0>(currentEntry);
        float maxDose = currentStruct.maxDose;
        float maxWeights = currentStruct.maxWeights;
        float minDoseTarget = currentStruct.minDoseTarget;
        float minDoseTargetWeights = currentStruct.minDoseTargetWeights;
        float OARWeights = currentStruct.OARWeights;
        if (std::get<1>(currentEntry)) {
            // its a PTV
            if (referenceDose == nullptr) {
                // without SIB, using normal piece-wise constant parameters
                for (size_t ii=0; ii<current_num_voxels; ii++) {
                    weights.maxDose[offset0 + ii] = maxDose;
                    weights.maxWeightsLong[offset0 + ii] = maxWeights;
                    weights.minDoseTarget[offset0 + ii] = minDoseTarget;
                    weights.minDoseTargetWeights[offset0 + ii] = minDoseTargetWeights;
                }
            } else {
                // with SIB, use the reference dose
                size_t ii_local = 0;
                for (size_t jj=0; jj<nVoxels; jj++) {
                    if (currentStruct.mask[jj] > 0) {
                        weights.maxDose[offset0 + ii_local] = (*referenceDose)[jj];
                        weights.maxWeightsLong[offset0 + ii_local] = maxWeights;
                        weights.minDoseTarget[offset0 + ii_local] = (*referenceDose)[jj];
                        weights.minDoseTargetWeights[offset0 + ii_local] = minDoseTargetWeights;
                        ii_local ++;
                    }
                }
                if (ii_local != current_num_voxels) {
                    std::cerr << "Error, inconsistency in the current number of voxels." << std::endl;
                    return 1;
                }
            }
            offset0 += current_num_voxels;
            continue;
        }
        else {
            for (size_t ii=0; ii<current_num_voxels; ii++) {
                weights.maxDose[offset0 + ii] = maxDose;
                weights.maxWeightsLong[offset0 + ii] = maxWeights;
                weights.OARWeightsLong[offset1 + ii] = OARWeights;
            }
            offset0 += current_num_voxels;
            offset1 += current_num_voxels;
            continue;
        }
    }

    return 0;
}

bool IMRT::fluenceGradInit(
    MatCSR64& SpFluenceGrad, MatCSR64& SpFluenceGradT,
    std::vector<uint8_t>& fluenceArray, const std::string& fluenceMapPath,
    int fluenceDim
) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
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

    // allocate buffer
    size_t bufferSize;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    array_1d<float> input, output;
    arrayInit(input, dest.numCols);
    arrayInit(output, dest.numRows);
    cusparseHandle_t handle_cusparse;
    checkCusparse(cusparseCreate(&handle_cusparse));
    checkCusparse(cusparseSpMV_bufferSize(
        handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, dest.matA, input.vec, &beta, output.vec,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc((void**)&dest.d_buffer_spmv, bufferSize));

    return 0;
}