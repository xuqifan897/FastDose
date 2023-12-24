#include "G4RunManagerFactory.hh"
#include "PhysicsList.h"

#include "ArgKernelGen.h"
#include "DetectorConstruction.h"
#include "PhysicsList.h"
#include "ActionInitialization.h"

 namespace fdkg = kernelgen;

int main(int argc, char** argv) {
    if(fdkg::ArgsInitKernelGen(argc, argv))
        return 0;

    auto* runManager = G4RunManagerFactory::CreateRunManager();
    
    G4Random::setTheSeed(std::time(nullptr));

    runManager->SetUserInitialization(new fdkg::DetectorConstruction);

    runManager->SetUserInitialization(new fdkg::PhysicsList);

    runManager->SetUserInitialization(new fdkg::ActionInitialization);

    runManager->Initialize();

    int nParticles = fdkg::getArgKG<int>("nParticles");
    runManager->BeamOn(nParticles);

    delete runManager;
}