#include <random>
#include <omp.h>
#include <mutex>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include "IMRTDoseMatEigen.cuh"
#include "IMRTArgs.h"

bool IMRT::test_parallelSpGEMM(
    const std::vector<MatCSR_Eigen>& OARMatrices,
    const std::vector<MatCSR_Eigen>& OARMatricesT,
    const std::vector<MatCSR_Eigen>& matricesT,
    const MatCSR_Eigen& filter
) {
    // firstly, test the correctness of the matricesT
    int numMatrices = matricesT.size();

    #if true
        fs::path resultFolder(getarg<std::string>("outputFolder"));
        resultFolder /= std::string("BeamDoseMatEigen");
        if (! fs::is_directory(resultFolder))
            fs::create_directory(resultFolder);

        for (int i=0; i<numMatrices; i++) {
            const MatCSR_Eigen& matT = matricesT[i];
            auto numRows = matT.getRows();
            MatCSR_Eigen weight;
            EigenIdxType* weightOffset = new EigenIdxType[2];
            weightOffset[0] = 0;
            weightOffset[1] = numRows;
            EigenIdxType* weightColumns = new EigenIdxType[numRows];
            float* weightValues = new float[numRows];
            for (EigenIdxType j=0; j<numRows; j++) {
                weightColumns[j] = j;
                weightValues[j] = 1.0f;
            }
            weight.customInit(1, numRows, numRows, weightOffset,
                weightColumns, weightValues);
            MatCSR_Eigen result = weight * matT;
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> result_dense = result.toDense();

            fs::path file = resultFolder / (std::string("beam")
                + std::to_string(i) + std::string(".bin"));
            std::ofstream ofs(file.string());
            if (! ofs.is_open()) {
                std::cerr << "Cannot open file: " << file << std::endl;
            }
            ofs.write((char*)result_dense.data(), result.getCols()*sizeof(float));
            ofs.close();
            std::cout << file << std::endl;
        }
        return 0;
    #endif

    bool pass = true;
    std::mutex passMutex;
    #pragma omp parallel for
    for (int i=0; i<numMatrices; i++) {
        const MatCSR_Eigen& matT = matricesT[i];
        auto numRows = matT.getRows();
        MatCSR_Eigen randWeight;
        EigenIdxType* offset = new EigenIdxType[2];
        offset[0] = 0;
        offset[1] = numRows;
        EigenIdxType* columns = new EigenIdxType[numRows];
        float* values = new float[numRows];
        for (int j=0; j<numRows; j++) {
            columns[j] = j;
            values[j] = (float)std::rand() / RAND_MAX;
        }
        randWeight.customInit(1, numRows, numRows, offset, columns, values);

        auto result1 = randWeight * matT * filter;
        auto result2 = randWeight * OARMatricesT[i];
        auto result3 = randWeight * OARMatrices[i].transpose();

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
            result1_dense = result1.toDense();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
            result2_dense = result2.toDense();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
            result3_dense = result3.toDense();

        bool localPass = true;
        for (size_t j=0; j<filter.getCols(); j++) {
            localPass = localPass
                && std::abs(result1_dense.data()[j] - result2_dense.data()[j]) < 1e-4
                && std::abs(result1_dense.data()[j] - result3_dense.data()[j]) < 1e-4;
            if (! localPass)
                break;
        }
        std::lock_guard<std::mutex> lock(passMutex);
        pass = pass && localPass;
    }

    if (pass) {
        std::cout << "The matrices, matricesT, OARMatricesT, and OARMatrices, "
            "passed the test!" << std::endl;
    } else {
        std::cout << "The matrices, matricesT, OARMatricesT, and OARMatrices, "
            "didn't pass the test." << std::endl;
    }
    return 0;
}


bool IMRT::test_OARMat_OARMatT(const MatCSR_Eigen& OARMat, const MatCSR_Eigen& OARMatT) {
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
        std::cout << "Transposing OARMat to get the reference OARMatT ..." << std::endl;
    #endif
    MatCSR_Eigen OARMatT_reference = OARMat.transpose();
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<
            std::chrono::milliseconds>(time1-time0);
        std::cout << "Transpose time elapsed: " << duration.count() * 0.001f
            << " [s]" << std::endl;
        std::cout << "Element-wise comparison ..." << std::endl;
    #endif

    size_t nnz = OARMatT_reference.getNnz();
    size_t numRows = OARMatT_reference.getRows();
    if (nnz != OARMatT.getNnz() || numRows != OARMatT.getRows()) {
        std::cerr << "Either the number of non-zero elements or the number "
            "of rows are inconsistent between OARMatT and the reference." << std::endl;
        return 1;
    }

    const EigenIdxType* offsets = OARMatT.getOffset();
    const EigenIdxType* offsets_reference = *OARMatT_reference.getOffset();
    const EigenIdxType* columns = OARMatT.getIndices();
    const EigenIdxType* columns_reference = OARMatT_reference.getIndices();
    const float* values = OARMatT.getValues();
    const float* values_reference = OARMatT_reference.getValues();

    for (size_t i=0; i<numRows+1; i++) {
        if (offsets[i] != offsets_reference[i]) {
            std::cerr << "The offsets values are inconsistent at row " << i << std::endl;
            return 1;
        }
    }

    for (size_t i=0; i<nnz; i++) {
        if (columns[i] != columns_reference[i]) {
            std::cerr << "The column indices at element " << i << " are inconsistent."
                " Column: " << columns[i] << ", reference column: " << columns_reference[i]
                << std::endl;
            return 1;
        } else if (std::abs(values[i] - values_reference[i] > 1e-4)) {
            std::cerr << "The values at element " << i << " are inconsistent."
                " Value: " << values[i] << ", reference value: " << values_reference[i]
                << std::endl;
            return 1;
        }
    }
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<
            std::chrono::milliseconds>(time2-time1);
        std::cout << "Element-wise comparison finished. Time elapsed: "
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif
    return 0;
}