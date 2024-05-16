#include "ArgsCubeScore.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "globals.hh"
#include "G4SystemOfUnits.hh"

po::variables_map cube_score::vm;
std::vector<std::pair<float, double>> cube_score::Spectrum;
std::vector<std::pair<std::string, int>> cube_score::SlabPhantom;
std::map<std::string, float> cube_score::MaterialDensity;

std::ios::fmtflags cube_score::original_flags = std::cout.flags();
std::streamsize cube_score::original_precision = std::cout.precision();
std::streambuf* cube_score::original_buffer = std::cout.rdbuf();

bool cube_score::argsInit(int argc, char** argv) {
    po::options_description desc("An application to calculate the " \
        "dose of a user-defined grid phantom");
    desc.add_options()
        ("help", "To produce the help messages")
        ("SpectrumFile", po::value<std::string>()->required(), "A CSV file specifying the " \
            "spectrum of the X-ray beam")
        ("SlabPhantomFile", po::value<std::string>()->required(), "A CSV file specifying the " \
            "slab phantom configuration")
        ("MaterialFile", po::value<std::string>()->required(), "A CSV file specifying the material density.")
        ("VoxelSize", po::value<float>()->default_value(0.05), "The half voxel size [cm]")
        ("DimXY", po::value<int>()->default_value(99),
            "The dimension in x and y directions (z direction specified in SlabPhantomFile)")
        ("SAD", po::value<float>()->default_value(100), "The source-to-axis distance [cm] " \
            "(The isocenter is set to the center of the phantom)")
        ("FluenceSize", po::value<float>()->default_value(0.5),
            "The half fluence map dimension at the isocenter plane. [cm]")
        ("nParticles", po::value<int>()->default_value(100),
            "The number of particles in the simulation")
        ("logFreq", po::value<int>()->default_value(1), "Logging frequency")
        ("OutputFile", po::value<std::string>()->required(), "The file to store the binary " \
            "energy deposition matrix, stored in the order (x, y, z), row major, i.e., " \
            "storage contiguous in z dimension");
    
    std::string pattern("--help");
    for (int i=0; i<argc; i++) {
        char* _entry = argv[i];
        std::string entry(_entry);
        if (entry.find(pattern) != std::string::npos) {
            // print help messages
            std::cout << desc << std::endl;
            return 1;
        }
    }

    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    G4cout << "Parameters:" << G4endl;
    for (const auto& pair : vm)
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

    // Initialize spectrum
    const std::string& SpectrumFile = getarg<std::string>("SpectrumFile");
    std::vector<std::vector<std::string>> SpectrumData;
    ReadCSV(SpectrumFile, SpectrumData);
    G4double totalFluence = 0.0f;
    for (const auto& line : SpectrumData) {
        if (line.size() != 2) {
            std::cerr << "More or less than 2 entries in a line" << std::endl;
            return 1;
        }
        G4float energy = std::stof(line[0]) * MeV;
        G4double fluence = std::stof(line[1]);
        Spectrum.push_back(std::make_pair(energy, fluence));
        totalFluence += fluence;
    }
    totalFluence = getarg<int>("nParticles") / totalFluence;
    // normalize
    for (auto& line: Spectrum) {
        line.second *= totalFluence;
        line.second = int(std::round(line.second));
    }
    #if true
        std::cout << "Particles simulated:" << "\n";
        int totalNumber = 0;
        for (const auto& line: Spectrum) {
            std::cout << "Energy: " << std::fixed << std::setprecision(3) << std::setfill('0')
                << line.first / MeV << " MeV, count: " << std::defaultfloat << line.second << "\n";
            totalNumber += line.second;
        }
        std::cout << "The real number of particles simulated: " << totalNumber << "\n\n";
    #endif


    // Slab phantom initialization
    const std::string& SlabPhantomFile = getarg<std::string>("SlabPhantomFile");
    std::vector<std::vector<std::string>> SlabPhantomData;
    ReadCSV(SlabPhantomFile, SlabPhantomData);
    for (const auto& line: SlabPhantomData) {
        int thickness_dim = std::stof(line[1]);
        SlabPhantom.push_back(std::make_pair(line[0], thickness_dim));
    }

    const std::string& MaterialFile = getarg<std::string>("MaterialFile");
    std::vector<std::vector<std::string>> MaterialData;
    ReadCSV(MaterialFile, MaterialData);
    // Sanity check, to ensure no double definition
    for (const auto& line: MaterialData) {
        float density = std::stof(line[1]);
        const std::string& material_name = line[0];
        if (MaterialDensity.count(material_name) > 0) {
            std::cerr << "Error: Material double definition!" << std::endl;
            return 1;
        }
        MaterialDensity[material_name] = density * g/cm3;
    }
    #if true
    int totalDimZ = 0;
    for (int i=0; i<SlabPhantom.size(); i++) {
        const std::string& material = SlabPhantom[i].first;
        float slice_density = MaterialDensity[material];
        std::cout << "Slice " << i + 1 << ", material: " << SlabPhantom[i].first
            << ", z dimension: " << SlabPhantom[i].second << ", density: "
            << slice_density * cm3/g << "g/cm3\n";
        totalDimZ += SlabPhantom[i].second;
    }
    std::cout << "Total z dimension: " << totalDimZ << "\n\n";
    #endif

    return 0;
}

bool cube_score::ReadCSV(const std::string& file,
    std::vector<std::vector<std::string>>& output) {
    std::ifstream f(file);
    if (! f.is_open()) {
        std::cerr << "Cannot open file " << file << std::endl;
        return 1;
    }
    output.clear();
    const std::string skip("#");
    std::string tableRow, entry;
    while (std::getline(f, tableRow)) {
        if (tableRow.find(skip) == 0)
            continue;  // A comment line
        if (tableRow.size() == 0)
            break;  // An empty line
        output.push_back(std::vector<std::string>());
        std::vector<std::string>& last_line = output.back();
        std::istringstream tableRowSS(tableRow);
        while (std::getline(tableRowSS, entry, ',')) {
            last_line.push_back(entry);
        }
    }
    return 0;
}