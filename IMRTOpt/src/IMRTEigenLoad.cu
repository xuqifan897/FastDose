#include <chrono>
#include "IMRTDoseMatEigen.cuh"
#include "IMRTInit.cuh"

bool IMRT::OARFiltering(
    const std::string& resultFolder,
    const std::vector<StructInfo>& structs
) {
    MatCSR_Eigen filter, filterT;
    getStructFilter(filter, filterT, structs);
    std::vector<MatCSR_Eigen> OARMatrices;
    std::vector<MatCSR_Eigen> OARMatricesT;
    parallelSpGEMM(resultFolder, filter, filterT, OARMatrices,
        OARMatricesT);

    MatCSR_Eigen OARMat, OARMatT;
    parallelMatCoalease(OARMat, OARMatT, OARMatrices,
        OARMatricesT);

    #if false
        if(test_OARMat_OARMatT(OARMat, OARMatT)) {
            std::cerr << "OARMat and OARMatT test error." << std::endl;
            return 1;
        } else {
            std::cout << "OARMat and OARMatT match each other, test passed!" << std::endl;
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
            << duration.count() * 0.001f << " [s]. Transpose starts." << std::endl;
    #endif
    
    filter = filterT.transpose();
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<
            std::chrono::milliseconds>(time2 - time1);
        std::cout << "OAR filter transpose time: " << duration.count() * 0.001f
            << " [s]." << std::endl;
    #endif

    return 0;
}