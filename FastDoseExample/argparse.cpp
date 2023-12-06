#include "argparse.h"
#include "utils.cuh"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::variables_map example::vm;

bool example::argparse(int argc, char** argv) {
    po::options_description desc("The argument list for the example of fast dose calculation.");
    desc.add_options()
        ("help", "Produce help messages. This program requires the following files in the inputFolder:\n"
            "1) spec_6mv.spec, which contains the poly-energetic spectrum."
            "2) convolution_phi_angles.raw, which contains the convolution phi angles for each convolution direction.\n"
            "3) convolution_theta_angles.raw, which contains the convolution theta angles for each convolution direction.\n"
            "4) cumulative_kernels.h5, which contains the kernel used in the dose calculation.\n"
            "5) density.raw, which contains the raw density of the phantom.\n"
            "6) beam_lists.txt, which contains the specifications of the beam information.\n"
            "All the parameters marked as (required) should be specified.\n\n"
        )
        ("inputFolder", po::value<std::string>(), "The folder that contains the input data (required)")
        ("outputFolder", po::value<std::string>(), "The folder to output results (required).")
        ("deviceIdx", po::value<int>()->default_value(0), "The GPU to use")
        
        // geometries
        ("dicomVolumeDimension", po::value<std::vector<int>>()->multitoken(),
            "Dicom volume dimension, 3 elements (required).")
        ("voxelSize", po::value<std::vector<float>>()->multitoken(),
            "Dicom voxel size. [cm] (required)")
        ("doseBoundingBoxStartIndices", po::value<std::vector<int>>()->multitoken(),
            "Dose bounding box start indices (required).")
        ("doseBoundingBoxDimensions", po::value<std::vector<int>>()->multitoken(),
            "Dose bounding box dimensions (required).")
        
        // dose calculation
        ("kernelType", po::value<std::string>()->default_value("tabulated"),
            "Choose between \"tabulated\" and \"exponential\".")
        ("kernelExtent", po::value<float>()->default_value(4.0),
            "Dose kernel radius truncate distance, used in tabulated kernel. [cm]")
        ("nphi", po::value<int>()->default_value(8),
            "number of phi angles in convolution")
        ("ntheta", po::value<int>()->default_value(8),
            "number of theta values in convolution")
        ("nradii", po::value<int>()->default_value(24),
            "number of radii values in convolution");
        
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