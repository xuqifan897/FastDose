#ifndef __ARGSCUBESCORE_H__
#define __ARGSCUBESCORE_H__

#include <boost/program_options.hpp>
#include <vector>
#include <iostream>

namespace po = boost::program_options;

namespace cube_score {
    extern po::variables_map vm;
    extern std::vector<std::pair<float, double>> Spectrum;
    extern std::vector<std::pair<std::string, int>> SlabPhantom;
    extern std::map<std::string, float> MaterialDensity;

    // Store the current state of the standard IO stream
    extern std::ios::fmtflags original_flags;
    extern std::streamsize original_precision;
    extern std::streambuf* original_buffer;

    bool argsInit(int argc, char** argv);

    bool ReadCSV(const std::string& file, std::vector<std::vector<std::string>>& output);

    template<class T> const T& getarg(std::string key) {
        try {
            return vm.at(key).as<T>();
        } catch (const std::out_of_range&) {
            std::cerr << "The key " << key << " doesn't exist in the argument list!" << std::endl;
            exit(1);
        }
    }
}

#endif