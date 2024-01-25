#include <boost/filesystem.hpp>
#include "cuda_runtime.h"
#include "fastdose.cuh"
#include "IMRTArgs.h"
#include "IMRTInit.cuh"
#include "IMRTBeamBundle.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTDoseMatEns.cuh"
#include "IMRTDoseMatEigen.cuh"
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
    }

    IMRT::MatCSR_Eigen matCsrT_eigen;
    fs::path doseMatFolder(IMRT::getarg<std::string>("outputFolder"));
    doseMatFolder /= std::string("doseMatFolder");
    if (mode == 0) {
        matEns->tofile(doseMatFolder.string());
        delete matEns;
        return;
    } else if (mode == 1) {
        matCsrT_eigen.fromEnsemble(*matEns);
        delete matEns;
    } else if (mode == 2) {
        const std::vector<int>& phantomDim = IMRT::getarg<std::vector<int>>("phantomDim");
        size_t numCols = phantomDim[0] * phantomDim[1] * phantomDim[2];
        matCsrT_eigen.fromfile(doseMatFolder.string(), numCols);
    }
    #if false
        IMRT::test_MatCSR_load(matCsrT_eigen, doseMatFolder.string());
    #endif

    IMRT::MatCSR_Eigen A_Eigen, AT_Eigen;
    IMRT::MatOARSlicing(matCsrT_eigen, A_Eigen, AT_Eigen, structs);

    #if false
        fs::path doseMatFolder(IMRT::getarg<std::string>("outputFolder"));
        doseMatFolder /= std::string("doseMatFolder");
        if (mode == 0) {
            matEns->tofile(doseMatFolder.string());
            return;
        } else if (mode == 2) {
            const std::vector<int>& phantomDim = IMRT::getarg<std::vector<int>>("phantomDim");
            size_t numColsPerMat = phantomDim[0] * phantomDim[1] * phantomDim[2];
            matEns = new IMRT::MatCSREnsemble(numColsPerMat);
            matEns->fromfile(doseMatFolder.string());
        }

        #if false
            IMRT::sparseValidation(matEns);
        #endif

        IMRT::MatCSR<size_t> *matrixT;
        matrixT = new IMRT::MatCSR<size_t>;
        matrixT->fuseEnsemble(*matEns);
        delete matEns;

        #if false
            IMRT::conversionValidation(matrix, *matEns);
        #endif

        // A, AT are the sub-blocks of matrix above, containing only the OAR voxels
        IMRT::MatCSR<size_t> *A, *AT;
        A = new IMRT::MatCSR<size_t>;
        AT = new IMRT::MatCSR<size_t>;

        IMRT::MatOARSlicing(*matrixT, *A, *AT, structs);
        delete matrixT;
    #endif
}