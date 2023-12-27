#include "boost/filesystem.hpp"
namespace fs = boost::filesystem;
#include <string>

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParallelWorldPhysics.hh"

#include "argparseBS.h"
#include "PhantomDefBS.h"
#include "DetectorConstructionBS.h"
#include "PhysicsListBS.h"
#include "ActionInitializationBS.h"
#include "RunBS.h"

int main(int argc, char** argv)
{
    if (bs::argsInit(argc, argv))
        return 0;

    auto* runManager = G4RunManagerFactory::CreateRunManager(
        G4RunManagerType::Default);
    
    bs::GD = new bs::GeomDef();
    bs::GD->display();

    fs::path folder((*bs::vm)["resultFolder"].as<std::string>());
    if (! fs::exists(folder))
        fs::create_directories(folder);

    // prepare for the gloabl and local score matrices
    float voxelSize = (*bs::vm)["voxelSize"].as<float>() * cm;  // half voxel size
    int iteration = (*bs::vm)["iteration"].as<int>();  // the iteration of this execution
    float sizeZ = 0.;  // half size
    for (int i=0; i<bs::GD->layers.size(); i++)
        sizeZ += std::get<1>(bs::GD->layers[i]);
    int dimZ = std::round(sizeZ / voxelSize);
    std::cout << "Z dimension: " << dimZ;
    int dimXY = (*bs::vm)["dimXY"].as<int>();
    int SegZ = (*bs::vm)["SegZ"].as<int>();
    if (dimZ % SegZ != 0)
    {
        std::cerr << "dimZ is not a multiple of SegZ, error!" << std::endl;
        return 1;
    }
    if (iteration >= dimZ / SegZ)
    {
        std::cerr << "iteration: " << iteration << ", dimZ / SegZ: " << dimZ / SegZ << std::endl;
        return 1;
    }
    std::vector<double> localScore(dimXY * dimXY * SegZ);

    // write metadata
    std::stringstream metadataName;
    metadataName << "metadata_" << std::setw(3) << std::setfill('0') << iteration+1 << ".txt";
    std::ofstream metadataFile((folder / fs::path(metadataName.str())).string());
    metadataFile << "Block dimension (z, y, x): (" << SegZ << ", " << dimXY << ", " << dimXY << ")" << std::endl;
    metadataFile << "Number of blocks: " << dimZ / SegZ << std::endl;
    metadataFile << "Voxel size [cm] (half): " << voxelSize / cm << std::endl;
    metadataFile << "Data type: double" << std::endl;
    metadataFile.close();

    G4Random::setTheSeed(std::time(nullptr));
    float thickness = SegZ * voxelSize;
    float offset = iteration * thickness;

    runManager->SetUserInitialization(new bs::DetectorConstruction(offset, thickness));
    runManager->SetUserInitialization(new bs::PhysicsList());
    runManager->SetUserInitialization(new bs::ActionInitialization(&localScore));
    runManager->Initialize();

    int nParticles = (*bs::vm)["nParticles"].as<int>();
    runManager->BeamOn(nParticles);

    // log data
    std::stringstream dataNameSS;
    dataNameSS << "SD" << std::setw(3) << std::setfill('0') << iteration+1 << ".bin";
    fs::path dataName = folder / fs::path(dataNameSS.str());
    std::ofstream dataFile(dataName.string());
    if (dataFile.is_open())
    {
        dataFile.write((char*)(localScore.data()), SegZ*dimXY*dimXY*sizeof(double));
        dataFile.close();
        G4cout << "data " << iteration+1 << " written successfully!" << G4endl;
    }
    else
        G4cerr << "data " << iteration+1 << " writing unsuccessful." << G4endl;

    delete runManager;
    return 0;
}