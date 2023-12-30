#include "fastdose.cuh"
#include "PlanOptmArgs.cuh"
#include "PlanOptmInit.cuh"
#include "PlanOptmBeamBundle.cuh"
#include "PlanOptmTestCase.cuh"

#include "cuda_runtime.h"

namespace fd = fastdose;

int main(int argc, char** argv) {
    if (PlanOptm::argparse(argc, argv))
        return 1;
    
    int deviceIdx = PlanOptm::getarg<int>("deviceIdx");
    cudaSetDevice(deviceIdx);
    PlanOptm::showDeviceProperties(deviceIdx);

    fd::DENSITY_h density_h;
    fd::DENSITY_d density_d;
    if (PlanOptm::densityInit(density_h, density_d)) {
        std::cerr << "density initialization failed." << std::endl;
        return 1;
    }

    fd::SPECTRUM_h spectrum_h;
    if (PlanOptm::specInit(spectrum_h)) {
        std::cerr << "spectrum initialization fialed." << std::endl;
        return 1;
    }

    fd::KERNEL_h kernel_h;
    if (PlanOptm::kernelInit(kernel_h)) {
        std::cerr << "kernel initialization fialed." << std::endl;
        return 1;
    }

    #if false
        if (PlanOptm::testCase(density_h, density_d, spectrum_h, kernel_h)) {
            std::cerr << "test case failed." << std::endl;
            return 1;
        }
    #endif

    std::vector<PlanOptm::BeamBundle> beam_bundles;
    if (PlanOptm::BeamBundleInit(beam_bundles)) {
        std::cerr << "beam bundle initialization failed." << std::endl;
        return 1;
    }
}