#ifndef __PLANOPTMARGS_CUH__
#define __PLANOPTMARGS_CUH__

#include "boost/program_options.hpp"
#include <iostream>

namespace PlanOptm {
    bool argparse(int argc, char** argv);
    extern boost::program_options::variables_map vm;

    template<class T>
    const T& getarg(const std::string& key) {
        try {
            return vm.at(key).as<T>();
        } catch (const std::out_of_range&) {
            std::cerr << "The key " << key << " doesn't exist in the argument list!" << std::endl;
            exit(1);
        }
    }

    bool showDeviceProperties(int deviceIdx);
}

#endif