#include "G4RunManagerFactory.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "ArgMCRef.h"
#include "DetectorConstructionMCRef.h"
#include "PhysicsListMCRef.h"
#include "ActionInitializationMCRef.h"

int main(int argc, char** argv) {
    if (MCRef::argsInit(argc, argv))
        return 1;

    auto* runManager = G4RunManagerFactory::CreateRunManager(
        G4RunManagerType::Default);
    
    G4Random::setTheSeed(std::time(nullptr));

    const std::vector<int>& phantomDim = MCRef::getarg<std::vector<int>>("phantomDim");
    int superSampling = MCRef::getarg<int>("superSampling");

    int scoringStartIdx = MCRef::getarg<int>("scoringStartIdx");
    int scoringSliceSize = MCRef::getarg<int>("scoringSliceSize");
    int scoringEndIdx = scoringStartIdx + scoringSliceSize;
    scoringEndIdx = std::min(scoringEndIdx, phantomDim[1]);
    scoringSliceSize = scoringEndIdx - scoringStartIdx;

    size_t scoringSize = phantomDim[0] * scoringSliceSize * superSampling * phantomDim[2];
    std::vector<G4double> EnergyMap(scoringSize);

    runManager->SetUserInitialization(
        new MCRef::DetectorConstruction(scoringStartIdx, scoringEndIdx));
    runManager->SetUserInitialization(
        new MCRef::PhysicsList);
    runManager->SetUserInitialization(
        new MCRef::ActionInitialization(
            scoringStartIdx, scoringEndIdx, EnergyMap));
    runManager->Initialize();

    int nParticles = MCRef::getarg<int>("nParticles");
    runManager->BeamOn(nParticles);

    // output
    fs::path outputFolder(MCRef::getarg<std::string>("outputFolder"));
    outputFolder = outputFolder / (std::string("MCRefFmap") + std::to_string(MCRef::getarg<float>("FmapOn")));
    if (! fs::is_directory(outputFolder))
        fs::create_directories(outputFolder);
    fs::path resultFile = outputFolder / (std::string("slice_") + std::to_string(scoringStartIdx) +
        std::string("_") + std::to_string(scoringEndIdx) + std::string(".bin"));
    std::ofstream f(resultFile.string());
    if (! f.is_open()) {
        std::cerr << "Could not open file: " << resultFile.string() << std::endl;
        return 1;
    }
    f.write((char*)(EnergyMap.data()), scoringSize*sizeof(G4double));
    f.close();

    delete(runManager);
}