#include "cuda_runtime.h"
#include "fastdose.cuh"
#include "IMRTArgs.h"
#include "IMRTInit.cuh"

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
}