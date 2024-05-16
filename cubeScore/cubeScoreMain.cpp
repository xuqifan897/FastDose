#include "ArgsCubeScore.h"
#include "DetectorCubeScore.h"
#include "ActionInitCubeScore.h"

#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "G4VisManager.hh"
#include "G4RunManagerFactory.hh"
#include "QGS_BIC.hh"
#include "G4UImanager.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <fstream>
#include <ctime>

int main(int argc, char** argv) {
    if (cube_score::argsInit(argc, argv))
        return 0;
    
    std::time_t currentTime = std::time(nullptr);
    G4Random::setTheSeed(static_cast<long>(currentTime));

    std::vector<double> result;
    auto* runManager = G4RunManagerFactory::CreateRunManager();
    runManager->SetUserInitialization(new cube_score::DetectorConstruction());
    runManager->SetUserInitialization(new QGS_BIC());
    runManager->SetUserInitialization(new cube_score::ActionInitialization(&result));
    runManager->Initialize();

    int nParticles = cube_score::getarg<int>("nParticles");
    runManager->BeamOn(nParticles);
    delete runManager;

    const std::string& resultPath = cube_score::getarg<std::string>("OutputFile");
    std::ofstream file(resultPath);
    if (! file.is_open()) {
        std::cerr << "Cannot open file " << resultPath << std::endl;;
        return 1;
    }
    file.write((char*)result.data(), result.size() * sizeof(double));
    file.close();
    std::cout << "Data written to " << resultPath << std::endl;
}