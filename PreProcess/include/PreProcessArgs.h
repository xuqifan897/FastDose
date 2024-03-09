#ifndef __PREPROCESSARGS_H__
#define __PREPROCESSARGS_H__
#include <boost/program_options.hpp>
#include <iostream>

namespace PreProcess {
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
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& arr)
{
    os << "[";
    int N = arr.size();
    for (int i=0; i<N; i++)
    {
        os << arr[i];
        if (i < N-1)
            os << ", ";
    }
    os << "]";
    return os;
}

#endif