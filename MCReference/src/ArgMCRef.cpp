#include "ArgMCRef.h"
#include <boost/program_options.hpp>
#include <vector>
#include <string>
#include <iostream>
namespace po = boost::program_options;

po::variables_map *MCRef::vm;

bool MCRef::argsInit(int argc, char** argv) {
    po::options_description desc(
        "An application to generate the baseline dose "
        "distribution of a phantom using a Cartesian grid");
    desc.add_options()
        ("help", "This application is to calculate the dose distribution"
        "of a diverging beam in a slab phantom.")
        ("SpectrumFile", po::value<std::string>()->required(),
            "The spectrum file")
        ("nParticles", po::value<int>()->default_value(1),
            "The number of particles to simulate")
        
        // Geometry
        ("voxelSize", po::value<float>()->default_value(0.125f),
            "Voxel size in half [cm].")
        ("phantomDim", po::value<std::vector<int>>()->multitoken(),
            "Phantom dimension.")
        ("phantomPath", po::value<std::string>()->required(),
            "The path to the binary phantom file.")
        ("SAD", po::value<float>()->default_value(100.0f),
            "Source-to-axis distance [cm]")
        ("FmapOn", po::value<float>()->default_value(0.25),
            "The half size of the fluence map. [cm]")
        ("scoringStartIdx", po::value<int>()->default_value(0),
            "The starting index of the scoring")
        ("scoringSliceSize", po::value<int>()->default_value(10),
            "The thickness of the scoring slab")
        ("superSampling", po::value<int>()->default_value(1),
            "Super sampling in the Z direction")
        
        // IO
        ("outputFolder", po::value<std::string>()->required(),
            "The folder for output")
        ("logFrequency", po::value<int>()->default_value(100000),
            "Log frequency");
    
    vm = new po::variables_map();
    po::store(po::parse_command_line(argc, argv, desc), *vm);
    po::notify(*vm);

    if ((*vm).count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    const std::vector<int>& phantomDim = (*vm)["phantomDim"].as<std::vector<int>>();

    std::cout << "Spectrum file: " << (*vm)["SpectrumFile"].as<std::string>() << std::endl
        << "Number of particles: " << (*vm)["nParticles"].as<int>() << std::endl
        << "Voxel size (half): " << (*vm)["voxelSize"].as<float>() << "cm" << std::endl
        << "Phantom dimension: (" << phantomDim[0] << ", " << phantomDim[1] << ", " << phantomDim[2] << ")" << std::endl
        << "Phantom path: " << (*vm)["phantomPath"].as<std::string>() << std::endl
        << "SAD: " << (*vm)["SAD"].as<float>() << "cm" << std::endl
        << "Fluence map size (half): " << (*vm)["FmapOn"].as<float>() << "cm" << std::endl;
        
    return 0;
}