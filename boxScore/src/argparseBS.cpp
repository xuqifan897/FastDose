#include <boost/program_options.hpp>
#include "globals.hh"
#include <string>
#include <iostream>

#include "argparseBS.h"

po::variables_map* bs::vm = nullptr;

int bs::argsInit(int argc, char** argv)
{
    po::options_description desc(
        "An application to calculate the dose distribution of a "
            "phantom using a Cartesian grid");
    desc.add_options()
        ("help", "This application is to calculate the dose distribution "
            "of a diverging beam in a slab phantom. The user can specify "
            "the geometry in ./src/PhantomDef.cpp. Resolution and sizes "
            "are in half, in accordance with the settings of Geant4")
        ("Energy", po::value<float>()->default_value(6.),
            "the value of primary photon energy [MeV].")
        ("nParticles", po::value<int>()->default_value(1),
            "The number of particles to simulate")
        
        // Geometry
        ("voxelSize", po::value<float>()->default_value(0.05), "Sensitive detector "
            "voxel size. Here the sensitive detector elements are defined in "
            "the parallel world. sizes are in half values [cm]")
        ("dimXY", po::value<int>()->default_value(256), "the X and Y dimension of "
            "the sensitive detectors")
        ("SegZ", po::value<int>()->default_value(16), "The number of Z detector "
            "elements in the parallel sensitive detector")
        ("SAD", po::value<float>()->default_value(100.), "Source-to-Axis distance [cm]")
        ("beamlet-size", po::value<float>()->default_value(.25), "beamlet size, in half value [cm]")
        
        // The following block is for logging
        ("resultFolder", po::value<std::string>()->required(),
            "The folder to which we write results")
        ("logFreq", po::value<int>()->default_value(100000), "the log frequency.")
        ("iteration", po::value<int>()->default_value(0), 
            "We score the dose within one block in one iteration. "
            "This variable indicates the iteration index of the current execution.");
    
    vm = new po::variables_map();
    po::store(po::parse_command_line(argc, argv, desc), *vm);
    po::notify(*vm);

    if ((*vm).count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    G4cout << "Parameters:" << G4endl;
    for (const auto& pair : *vm)
    {
        G4cout << pair.first << " = ";
        const auto& value = pair.second.value();
        if (auto ptr = boost::any_cast<bool>(&value))
            G4cout << *ptr << G4endl;
        else if (auto ptr = boost::any_cast<int>(&value))
            G4cout << *ptr << G4endl;
        else if (auto ptr = boost::any_cast<float>(&value))
            G4cout << *ptr << G4endl;
        else if (auto ptr = boost::any_cast<std::string>(&value))
            G4cout << *ptr << G4endl;
        else
            G4cout << "(unknown type)" << G4endl;
    }
    G4cout << G4endl;

    return 0;
}