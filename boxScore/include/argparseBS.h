#ifndef argparseBS_h
#define argparseBS_h 1

#include <boost/program_options.hpp>
namespace po = boost::program_options;

namespace bs
{
    int argsInit(int argc, char** argv);
    extern po::variables_map* vm;
}

#endif