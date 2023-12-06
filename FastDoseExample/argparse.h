#ifndef __ARGPARSE_H__
#define __ARGPARSE_H__
#include <boost/program_options.hpp>
#include <array>
#include <iostream>

namespace example {
    bool argparse(int argc, char** argv);

    extern boost::program_options::variables_map vm;

    template<class T>
    const T& getarg(const std::string& key) {
        if (vm.count(key))
            return vm[key].as<T>();
        std::cerr << "The key " << key << " doesn't exist in the argument list!" << std::endl;
        exit(1);
    }
};

#endif