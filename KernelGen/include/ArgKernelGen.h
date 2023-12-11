#ifndef __ARGKERNELGEN_H__
#define __ARGKERNELGEN_H__

#include "globals.hh"
#include <boost/program_options.hpp>
#include <vector>

namespace fastdose {
    bool ArgsInitKernelGen(int argc, char** argv);
    extern boost::program_options::variables_map vm;

    template <class T>
    const T& getArgKG(const std::string& key) {
        if (vm.find(key) == vm.end()) {
            G4cerr << "No such key: " << key << " in the variables map" << G4endl;
            exit(1);
        }
        return vm[key].as<T>();
    }

    bool ReadSpectrum();
    //                          energy, fluence
    extern std::vector<std::pair<float, float>> spectrum;
}

#endif