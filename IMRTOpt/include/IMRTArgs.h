#ifndef __IMRTArgs_h__
#define __IMRTArgs_h__
#include <boost/program_options.hpp>
#include <iostream>

namespace IMRT {
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

// Convert cuda vector types to c-style arrays
#define VECT2ARR(a, v) a[0] = v.x; a[1] = v.y;
#define VECT3ARR(a, v) a[0] = v.x; a[1] = v.y; a[2] = v.z;
// Convert c-style array to cuda vector types
#define ARR2VECT(v, a) v.x = a[0]; v.y = a[1];
#define ARR3VECT(v, a) v.x = a[0]; v.y = a[1]; v.z = a[2];

// std::cout formatting
#define FORMAT_3VEC(v) "("<<v.x<<", "<<v.y<<", "<<v.z<<")"
#define FORMAT_2VEC(v) "("<<v.x<<", "<<v.y<<")"

#endif