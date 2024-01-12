#include <string>
#include "utils.cuh"
#include "IMRTArgs.h"

namespace po = boost::program_options;
po::variables_map IMRT::vm;

bool IMRT::argparse(int argc, char** argv) {
    po::options_description desc("The argument list for IMRT treatment optimization");
    desc.add_options()
        ("help", "Produce help messages")
    ("phantomDim", po::value<std::vector<int>>()->multitoken()->required(),
        "The phantom dimension")
    ("voxelSize", po::value<std::vector<float>>()->multitoken()->required(),
        "The isotropic resolution [cm]")
    ("SAD", po::value<float>()->required(),
        "Source-to-axis distance [cm]. The isocenter by default "
        "is the center of mass of the PTV volume")
    ("density", po::value<std::string>()->required(),
        "The path to the density raw file")
    ("masks", po::value<std::string>()->required(),
        "The path to the masks file")
    ("primaryROI", po::value<std::string>()->required(),
        "The ROI to cover in dose calculation, which is typically the PTV area")
    ("bboxROI", po::value<std::string>()->required(),
        "The region within which to calculate the dose")
    ("structureInfo", po::value<std::string>()->required(),
        "The path to the file containing mask information")
    ("params", po::value<std::string>()->required(),
        "The path to the optimization parameters")
    ("beamlist", po::value<std::string>()->required(),
        "The path to the beamlist")
    
    // dose calculation
    ("deviceIdx", po::value<int>()->default_value(2),
        "The device index")
    ("spectrum", po::value<std::string>()->required(),
        "The path to the spectrum")
    ("kernel", po::value<std::string>()->required(),
        "The path to the exponential CCCS kernel")
    ("nPhi", po::value<int>()->default_value(8),
        "The number of phi angles in convolution")
    ("fluenceDim", po::value<int>()->default_value(20),
        "Fluence map dimension")
    ("beamletSize", po::value<float>()->default_value(0.5),
        "Beamlet size in cm")
    ("subFluenceDim", po::value<int>()->default_value(16),
        "The dimension of subdivided fluence for dose calculation accuracy")
    ("subFluenceOn", po::value<int>()->default_value(4),
        "The number of fluence pixels that are on in the subdivided fluence map, "
        "which corresponds to the beamlet size")
    ("longSpacing", po::value<float>()->default_value(0.25),
        "Longitudinal voxel size in the dose calculation")
    ("concurrency", po::value<int>()->default_value(8),
        "How many beams whose beamlet dose matrices are computed concurrently")
    ("extent", po::value<float>()->default_value(2.0f),
        "Used in dose interpolation, i.e., the dose from BEV to PVCS. For voxels "
        "farther than the distance, the dose is 0 [cm]")
        
    // io
    ("outputFolder", po::value<std::string>()->required(),
        "Output folder")
    
    // others
    ("nBeamsReserve", po::value<int>()->default_value(500),
        "reserve space for beam allocation");

    // to see if "--help" is in the argument
    if (argc == 1) {
        std::cout << desc << std::endl;
        return 1;
    } else {
        std::string firstArg(argv[1]);
        if (firstArg == std::string("--help")) {
            std::cout << desc << std::endl;
            return 1;
        }
    }

    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

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