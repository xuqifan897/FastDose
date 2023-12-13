#include "ArgKernelGen.h"
#include "globals.hh"
#include "G4SystemOfUnits.hh"
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/program_options.hpp>

namespace fd = fastdose;
namespace po = boost::program_options;

std::vector<std::pair<float, float>> fd::spectrum;

po::variables_map fd::vm;

bool fd::ArgsInitKernelGen(int argc, char** argv) {
    po::options_description desc(
        "The options for kernel generation"
    );
    desc.add_options()
    ("help", "This module calculates the poly-energetic kernel. "
        "The scoring volumes are concentric rings.")
    ("outputFolder", po::value<std::string>(),
        "The folder for outputs (required).")
    ("spectrumFile", po::value<std::string>(),
        "The path to the spectrum file (required).")
    ("nParticles", po::value<int>()->default_value(10000000),
        "The number of photons to simulate.")
    ("radiusRes", po::value<float>()->default_value(0.05f),
        "The resolution of the concentric rings radius [cm].")
    ("heightRes", po::value<float>()->default_value(0.05f),
        "The resolution of the concentric rings (z direction) [cm].")
    ("radiusDim", po::value<int>()->default_value(100),
        "The number of rings in the radius dimension.")
    ("heightDim", po::value<int>()->default_value(400),
        "The number of rings in the depth dimension.")
    ("marginTail", po::value<int>()->default_value(20),
        "The number of rings in the backward direction to account for backs-scatter.")
    ("marginHead", po::value<int>()->default_value(100),
        "The number of rings in the forward direction.")
    ("logFreq", po::value<int>()->default_value(100000),
        "The frequency to print logging information");
    
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        G4cout << desc << G4endl;
        return 1;
    }
    G4cout << "Parameters: " << G4endl;
    for (const auto& pair : vm) {
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
    if (ReadSpectrum())
        return 1;
    return 0;
}

bool fd::ReadSpectrum() {
    const auto & SpectrumFile = getArgKG<std::string>("spectrumFile");
    std::ifstream f(SpectrumFile);
    if (! f) {
        G4cerr << "Could not open the spectrum file: " << SpectrumFile << G4endl;
        return 1;
    }
    spectrum.clear();
    std::string tableRow;
    std::string buff;
    float energy, fluence, mu, mu_en;
    while (std::getline(f, tableRow)) {
        if (tableRow == std::string("\n"))
            break;
        std::istringstream iss(tableRow);
        iss >> energy >> fluence >> mu >> mu_en >> buff;
        spectrum.push_back(std::make_pair(energy * MeV, fluence));
    }

    // normalize fluence
    float fluence_total = 0.;
    for (int i=0; i<spectrum.size(); i++)
        fluence_total += spectrum[i].second;
    for (int i=0; i<spectrum.size(); i++)
        spectrum[i].second /= fluence_total;

    // log
    const int width = 24;
    G4cout << std::left << std::setw(width) << "Energy (MeV)" << 
        std::left << std::setw(width) << "Fluence" << G4endl;
    for (int i=0; i<spectrum.size(); i++)
        G4cout << std::scientific << std::setprecision(4) <<
            std::left << std::setw(width) << spectrum[i].first / MeV << 
            std::left << std::setw(width) << spectrum[i].second << G4endl;
    return 0;
}