#include <iomanip>
#include <boost/filesystem.hpp>
#include "cuda_runtime.h"
#include "fastdose.cuh"
#include "IMRTArgs.h"
#include "IMRTInit.cuh"
#include "IMRTBeamBundle.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTDoseMatEns.cuh"
#include "IMRTDoseMatEigen.cuh"
#include "IMRTOptimize.cuh"
#include "IMRTOptimize_var.cuh"
#include "IMRTDebug.cuh"

namespace fs = boost::filesystem;
namespace fd = fastdose;

int main(int argc, char** argv) {
    if (IMRT::argparse(argc, argv))
        return 0;

    int mode = IMRT::getarg<int>("mode");
    int deviceIdx = IMRT::getarg<int>("deviceIdx");
    cudaSetDevice(deviceIdx);

    if (fd::showDeviceProperties(deviceIdx)) {
        std::cerr << "Cannot show device properties." << std::endl;
        return 1;
    }

    std::vector<IMRT::StructInfo> structs;
    IMRT::MatCSREnsemble* matEns = nullptr;
    if (mode == 0) {
        // for mode 0, do dose calculation and store the result.
        // for mode 1, do beam orientation optimization
        if (IMRT::StructsInit_dosecalc(structs)) {
            std::cerr << "Structure initialization error In dose calculation."
                << std::endl;
            return 1;
        }

        fd::DENSITY_h density_h;
        fd::DENSITY_d density_d;
        if (IMRT::densityInit(density_h, density_d, structs)) {
            std::cerr << "Density initialization error." << std::endl;
            return 1;
        }

        fd::SPECTRUM_h spectrum_h;
        if (IMRT::specInit(spectrum_h)) {
            std::cerr << "Spectrum initialization error." << std::endl;
            return 1;
        }

        fd::KERNEL_h kernel_h;
        if (IMRT::kernelInit(kernel_h)) {
            std::cerr << "Kernel initialization error." << std::endl;
            return 1;
        }

        std::vector<IMRT::BeamBundle> beam_bundles;
        if (IMRT::BeamBundleInit(beam_bundles, density_h, structs)) {
            std::cerr << "Beam bundles initialization error." << std::endl;
            return 1;
        }

        if (IMRT::DoseMatConstruction(beam_bundles, density_d, spectrum_h, kernel_h, &matEns)) {
            std::cerr << "Dose matrix construction error." << std::endl;
            return 1;
        }

        if (IMRT::getarg<std::vector<std::string>>("outputFolder").size() != 1) {
            std::cerr << "Only one entry is expected in the argument \"outputFolder\"" << std::endl;
            return 1;
        }
        fs::path doseMatFolder(IMRT::getarg<std::vector<std::string>>("outputFolder")[0]);
        doseMatFolder /= std::string("doseMatFolder");
        matEns->tofile(doseMatFolder.string());
        fs::path fluenceMapPath = doseMatFolder / std::string("fluenceMap.bin");
        IMRT::beamletFlagSave(beam_bundles, fluenceMapPath.string());
        fs::path doseDataPath = doseMatFolder / std::string("dose_data.h5");
        IMRT::doseDataSave(beam_bundles, doseDataPath.string());
        return 0;
    }

    if (IMRT::StructsInit(structs)) {
        std::cerr << "Structure initialization error in beam "
            "orientation optimization." << std::endl;
        return 1;
    }

    IMRT::Params params;
    IMRT::Weights_h weights_h;
    std::vector<IMRT::MatCSR_Eigen> MatricesT_full, VOIMatrices, VOIMatricesT;
    std::vector<IMRT::MatCSR_Eigen> SpFluenceGrad, SpFluenceGradT;
    std::vector<uint8_t> fluenceArray;

    if (IMRT::ParamsInit(params)) {
        std::cerr << "Paramsters initialization error." << std::endl;
        return 1;
    }

    const std::vector<std::string>& outputFolder = 
        IMRT::getarg<std::vector<std::string>>("outputFolder");
    std::vector<std::string> doseMatFolders;
    doseMatFolders.reserve(outputFolder.size());
    for (int i=0; i<outputFolder.size(); i++) {
        fs::path item = fs::path(outputFolder[i]) / std::string("doseMatFolder");
        doseMatFolders.push_back(item.string());
    }
    const std::vector<int>& phantomDim = IMRT::getarg<std::vector<int>>("phantomDim");
    int SpMatT_ColsPerMat = phantomDim[0] * phantomDim[1] * phantomDim[2];
    
    if (IMRT::OARFiltering(doseMatFolders, structs,
        MatricesT_full, VOIMatrices, VOIMatricesT, weights_h)) {
        std::cerr << "VOI matrices and their transpose initialization error." << std::endl;
        return 1;
    }

    if (IMRT::fluenceGradInitGroup(SpFluenceGrad, SpFluenceGradT,
        fluenceArray, doseMatFolders)) {
        std::cerr << "Fluence gradient matrices and their transpose "
            "initialization error." << std::endl;
        return 1;
    }

    std::vector<float> costs;
    std::vector<int> activeBeams;
    std::vector<float> activeNorms;
    if (IMRT::BeamOrientationOptimization(
        VOIMatrices, VOIMatricesT, SpFluenceGrad, SpFluenceGradT,
        weights_h, params, fluenceArray, costs, activeBeams,
        activeNorms)) {
        std::cerr << "Beam orientation optimization error." << std::endl;
        return 1;
    }

    params.maxIter = 500;
    std::vector<float> costs_polish;
    Eigen::VectorXf xFull;
    if (IMRT::FluencePolish(
        VOIMatrices, VOIMatricesT, SpFluenceGrad, SpFluenceGradT,
        weights_h, params, fluenceArray, activeBeams, costs_polish, xFull)) {
        std::cerr << "Fluence map polishing error." << std::endl;
        return 1;
    }

    Eigen::VectorXf finalDose;
    if (IMRT::resultDoseCalc(
        MatricesT_full, activeBeams, xFull, finalDose)) {
        std::cerr << "Final dose calculation error." << std::endl;
        return 1;
    }

    if (IMRT::writeResults(activeBeams, MatricesT_full, xFull, finalDose)) {
        std::cerr << "Saving results error." << std::endl;
        return 1;
    }

    return 0;
}