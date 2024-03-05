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
    if (IMRT::StructsInit(structs)) {
        std::cerr << "Structure initialization error." << std::endl;
        return 1;
    }

    IMRT::MatCSREnsemble* matEns = nullptr;
    if (mode == 0 || mode == 1) {
        // for mode 0, do dose calculation and store the result.
        // for mode 1, do dose calculation and perform beam orientation optimization
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

        if (mode == 0) {
            fs::path doseMatFolder(IMRT::getarg<std::string>("outputFolder"));
            doseMatFolder /= std::string("doseMatFolder");
            matEns->tofile(doseMatFolder.string());
            fs::path fluenceMapPath = doseMatFolder / std::string("fluenceMap.bin");
            IMRT::beamletFlagSave(beam_bundles, fluenceMapPath.string());
            return 0;
        }
        return 0; // ignore mode 1 at this time.
    }

    IMRT::Params params;
    IMRT::Weights_h weights_h;
    std::vector<IMRT::MatCSR_Eigen> VOIMatrices, VOIMatricesT;
    std::vector<IMRT::MatCSR_Eigen> SpFluenceGrad, SpFluenceGradT;
    std::vector<uint8_t> fluenceArray;

    if (IMRT::ParamsInit(params)) {
        std::cerr << "Paramsters initialization error." << std::endl;
        return 1;
    }

    fs::path doseMatFolder(IMRT::getarg<std::string>("outputFolder"));
    doseMatFolder /= std::string("doseMatFolder");
    const std::vector<int>& phantomDim = IMRT::getarg<std::vector<int>>("phantomDim");
    int SpMatT_ColsPerMat = phantomDim[0] * phantomDim[1] * phantomDim[2];
    
    if (IMRT::OARFiltering(doseMatFolder.string(), structs,
        VOIMatrices, VOIMatricesT, weights_h)) {
        std::cerr << "VOI matrices and their transpose initialization error." << std::endl;
        return 1;
    }

    fs::path fluenceMapPath = doseMatFolder / std::string("fluenceMap.bin");
    if (IMRT::fluenceGradInit(SpFluenceGrad, SpFluenceGradT,
        fluenceArray, fluenceMapPath.string())) {
        std::cerr << "Fluence gradient matrices and their transpose "
            "initialiation error." << std::endl;
        return 1;
    }

    std::vector<float> xFull;
    std::vector<float> costs;
    std::vector<int> activeBeams;
    std::vector<float> activeNorms;
    std::vector<std::pair<int, std::vector<int>>> topN;
    if (IMRT::BeamOrientationOptimization(
        VOIMatrices, VOIMatricesT, SpFluenceGrad, SpFluenceGradT,
        weights_h, params, fluenceArray, xFull, costs, activeBeams,
        activeNorms, topN)) {
        std::cerr << "Beam orientation optimization error." << std::endl;
        return 1;
    }
}