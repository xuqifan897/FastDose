#include "fastdose.cuh"
#include "argparse.h"
#include "init.h"

namespace fd = fastdose;
using namespace example;

int main(int argc, char** argv) {
    if(argparse(argc, argv))
        return 0;
    
    int deviceIdx = getarg<int>("deviceIdx");
    cudaSetDevice(deviceIdx);

    fd::DENSITY_h density_h;
    fd::DENSITY_d density_d;
    if(densityInit(density_h, density_d)) {
        std::cerr << "density initialization failure." << std::endl;
        return 1;
    }
#if false
    densityTest(density_h, density_d);
#endif

    std::vector<fd::BEAM_h> beams_h;
    std::vector<fd::BEAM_d> beams_d;
    if (beamsInit(beams_h, beams_d, density_h)) {
        std::cerr << "beam initialization failure." << std::endl;
        return 1;
    }

    fd::SPECTRUM_h spectrum_h;
    if (specInit(spectrum_h)) {
        std::cerr << "spectrum initialization failure." << std::endl;
        return 1;
    }

#if false
    std::string outputFolder = getarg<std::string>("outputFolder");
    if(fd::test_TermaCompute(beams_d[0], density_d, spectrum_h, outputFolder))
        return 1;
#endif

#if true
    std::string outputFolder = getarg<std::string>("outputFolder");
//     if(fd::profile_TermaCompute(beams_d, density_d, spectrum_h, outputFolder))
//         return 1;
    if (fd::TermaComputeCollective(beams_d, density_d, spectrum_h))
        return 1;
#endif
}