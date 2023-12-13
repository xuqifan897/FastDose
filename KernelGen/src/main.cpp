#include "G4RunManagerFactory.hh"
#include "PhysicsList.h"

#include "ArgKernelGen.h"
#include "DetectorConstruction.h"
#include "PhysicsList.h"
#include "ActionInitialization.h"

namespace fd = fastdose;

int main(int argc, char** argv) {
    if(fd::ArgsInitKernelGen(argc, argv))
        return 0;

    auto* runManager = G4RunManagerFactory::CreateRunManager();
    
    G4Random::setTheSeed(std::time(nullptr));

    runManager->SetUserInitialization(new fd::DetectorConstruction);

    runManager->SetUserInitialization(new fd::PhysicsList);

    runManager->SetUserInitialization(new fd::ActionInitialization);

    runManager->Initialize();

    int nParticles = fd::getArgKG<int>("nParticles");
    runManager->BeamOn(nParticles);

    delete runManager;
}