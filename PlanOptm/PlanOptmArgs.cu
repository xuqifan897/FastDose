#include <string>
#include <vector>
#include <iostream>
#include "boost/program_options.hpp"
#include "cuda_runtime.h"

#include "PlanOptmArgs.cuh"
#include "utils.cuh"

namespace po = boost::program_options;
po::variables_map PlanOptm::vm;

bool PlanOptm::argparse(int argc, char** argv) {
    po::options_description desc("The argument list for treatment optimization");
    desc.add_options()
        ("help", "Produce help messages.")
    ("phantom", po::value<std::string>()->required(),
        "The path to the raw phantom file")
    ("phantomDim", po::value<std::vector<int>>()->multitoken()->required(),
        "The phantom dimension")
    ("voxelSize", po::value<std::vector<float>>()->multitoken()->required(),
        "Phantom voxel size in cm")
    ("isocenter", po::value<std::vector<float>>()->multitoken()->required(),
        "Phantom isocenter in cm")
    ("SAD", po::value<float>()->required(),
        "Source-to-axis distance in cm")
    ("boundingBoxStart", po::value<std::vector<int>>()->multitoken(),
        "Dose bounding box start index")
    ("boundingBoxDimensions", po::value<std::vector<int>>()->multitoken()->required(),
        "Dose bounding box dimensions")
    ("beamlist", po::value<std::string>()->required(),
        "The path to the beamlsit")
    
    // dose calculation
    ("deviceIdx", po::value<int>()->default_value(0),
        "GPU idx")
    ("spectrum", po::value<std::string>()->required(),
        "The path to the spectrum file")
    ("kernel", po::value<std::string>()->required(),
        "The path to the exponential CCCS kernel")
    ("nPhi", po::value<int>()->default_value(8),
        "The number of phi angles in convolution.")
    ("fluenceDim", po::value<int>()->default_value(16),
        "Fluence map dimension")
    ("beamletSize", po::value<float>()->default_value(0.4))
    ("subFluenceDim", po::value<int>()->default_value(16),
        "The dimension of subdivided fluence for dose calculation accuracy")
    ("subFluenceOn", po::value<int>()->default_value(4),
        "The number of fluence pixels that are on in the subdivided fluence map, "
        "which corresponds to the beamlet size")
    ("longSpacing", po::value<float>()->default_value(0.25),
        "Longitudinal voxel size in the dose calculation")

    // io
    ("outputFolder", po::value<std::string>()->required(),
        "Output folder")
    
    // others
    ("nBeamsReserve", po::value<int>()->default_value(1200),
        "reserve space for beam allocation");

    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    int width = 60;
    std::cout << "Parameters:" << std::endl;
    for (const auto& pair: vm) {
        std::stringstream second;
        const auto& value  = pair.second.value();
        if (auto ptr = boost::any_cast<int>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<float>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<std::vector<float>>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<std::vector<int>>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<std::string>(&value))
            second << *ptr;
        else
            second << "(unknown type)";
        
        std::string second_string = second.str();
        int remaining = width - pair.first.size() - second_string.size();
        remaining = std::max(5, remaining);

        std::stringstream output;
        output << pair.first << std::string(remaining, '.') << second_string;
        std::cout << output.str() << std::endl;
    }
    std::cout << std::endl;

    return 0;
}


bool PlanOptm::showDeviceProperties(int deviceIdx) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    std::cout << "Device " << deviceIdx << ": " << deviceProp.name << std::endl;
    std::cout << "    Global Memory Size: " << 
        deviceProp.totalGlobalMem << " bytes" << std::endl;
    std::cout << "    Shared Memory Size per Block: " << 
        deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "    Number of Registers per Block: " <<
        deviceProp.regsPerBlock << " bytes" << std::endl;
    std::cout << "    Max Blocks per Multiprocessor: " <<
        deviceProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "    Maximum resident threads per Multiprocessor: " <<
        deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "    Shared memory available per Multiprocessor: " << 
        deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "    Number of Threads per Warp: " <<
        deviceProp.warpSize << std::endl;
    std::cout << "    Number of SMs in total: " << 
        deviceProp.multiProcessorCount << std::endl;
    std::cout << std::endl;
    return 0;
}