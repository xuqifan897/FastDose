#include "cuda_runtime.h"
#include "fastdose.cuh"
#include "IMRTArgs.h"
#include "IMRTInit.cuh"
#include "IMRTBeamBundle.cuh"
#include "IMRTDoseMat.cuh"

namespace fd = fastdose;

int main(int argc, char** argv) {
    if (IMRT::argparse(argc, argv))
        return 0;

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

    if (IMRT::DoseMatConstruction(beam_bundles, density_d, spectrum_h, kernel_h)) {
        std::cerr << "Dose matrix construction error." << std::endl;
        return 1;
    }
}