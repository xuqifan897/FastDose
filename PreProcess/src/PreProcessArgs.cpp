#include "PreProcessArgs.h"
#include <vector>
#include <string>

namespace po = boost::program_options;
po::variables_map PreProcess::vm;

bool PreProcess::argparse(int argc, char** argv) {
    po::options_description desc("The argument list for dicom preprocessing");
    desc.add_options()
        ("help", "Produce help messages")
    ("dicomFolder", po::value<std::string>()->required(),
        "The path to the dicom files, including the RTStructure file.")
    ("structuresFile", po::value<std::string>()->required(),
        "The path to the structure json file.")
    ("ptv_name", po::value<std::string>()->default_value("P_"),
        "The PTV name.")
    ("bbox_name", po::value<std::string>()->default_value("Body"),
        "The name of the bounding box ROI.")
    ("target_exact_match", po::value<bool>()->default_value(false),
        "match by substring by default")
    ("verbose", po::value<bool>()->default_value(true),
        "To log more or less.")
    ("nolut", po::value<bool>()->default_value(false), 
        "To turn of the conversion from HU to mass density")
    ("ctlutFile", po::value<std::string>()->default_value(""),
        "The custom table to convert HU to mass density")
    ("voxelSize", po::value<float>()->default_value(0.25),
        "Isotropic voxel size in [cm]")
    ("inputFolder", po::value<std::string>()->required(),
        "The output folder of this program, the input folder of the optimization program, "
        "where the Dose_Coefficients.mask is stored.");

    
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
        else if (auto ptr = boost::any_cast<bool>(&value))
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