#include "spectrum.cuh"
#include "argparse.h"

#include <fstream>
#include <boost/filesystem.hpp>

namespace fd = fastdose;
namespace fs = boost::filesystem;

bool fd::SPECTRUM_h::read_spectrum_file(const std::string& spectrum_file) {
    std::ifstream f(spectrum_file);
    if (! f) {
        std::cout << "Cannot open spectrum file " << spectrum_file << std::endl;
        return true;
    }

    int nkernels = 0;
    float sum_fluence = 0.;
    
}