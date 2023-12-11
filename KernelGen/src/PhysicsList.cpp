#include "G4EmStandardPhysics.hh"
#include "G4EmStandardPhysics_option1.hh"
#include "G4EmStandardPhysics_option2.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmStandardPhysicsSS.hh"
#include "G4EmStandardPhysicsWVI.hh"
#include "G4EmStandardPhysicsGS.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4EmLowEPPhysics.hh"

#include "G4LossTableManager.hh"
#include "G4UnitsTable.hh"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4EmBuilder.hh"

#include "G4Decay.hh"
#include "StepMax.h"

#include "G4SystemOfUnits.hh"

#include "PhysicsList.h"

namespace fd = fastdose;

fd::PhysicsList::PhysicsList() : G4VModularPhysicsList()
{
    
    // EM physics
    fEmName = G4String("emstandard_opt4");
    fEmPhysicsList = new G4EmStandardPhysics_option4(1);
    if (verboseLevel>-1) {
        G4cout << "PhysicsList::Constructor with default list: <" 
               << fEmName << ">" << G4endl;
    }

    G4LossTableManager::Instance();
    SetVerboseLevel(1);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

fd::PhysicsList::~PhysicsList()
{
    delete fEmPhysicsList;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void fd::PhysicsList::ConstructParticle()
{
    // minimal set of particles for EM physics
    G4EmBuilder::ConstructMinimalEmSet();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void fd::PhysicsList::ConstructProcess()
{
    AddTransportation();
    fEmPhysicsList->ConstructProcess();
    AddDecay();
    AddStepMax();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void fd::PhysicsList::AddDecay()
{
    // Add Decay Process
    
    G4Decay* fDecayProcess = new G4Decay();
    
    auto particleIterator=GetParticleIterator();
    particleIterator->reset();
    while( (*particleIterator)() ){
        G4ParticleDefinition* particle = particleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        
        if (fDecayProcess->IsApplicable(*particle) && 
              !particle->IsShortLived()) {
            
            pmanager ->AddProcess(fDecayProcess);
            
            // set ordering for PostStepDoIt and AtRestDoIt
            pmanager ->SetProcessOrdering(fDecayProcess, idxPostStep);
            pmanager ->SetProcessOrdering(fDecayProcess, idxAtRest);
            
        }
    }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void fd::PhysicsList::AddStepMax()
{
    // Step limitation seen as a process
    fd::StepMax* stepMaxProcess = new fd::StepMax();
    
    auto particleIterator=GetParticleIterator();
    particleIterator->reset();
    while ((*particleIterator)()){
        G4ParticleDefinition* particle = particleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        
        if (stepMaxProcess->IsApplicable(*particle))
        {
            pmanager ->AddDiscreteProcess(stepMaxProcess);
        }
    }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void fd::PhysicsList::AddPhysicsList(const G4String& name)
{
    if (verboseLevel>-1) {
        G4cout << "fd::PhysicsList::AddPhysicsList: <" << name << ">" << G4endl;
    }
    
    if (name == fEmName) return;
    
    if (name == "emstandard_opt0") {
        
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmStandardPhysics(1);
        
    } else if (name == "emstandard_opt1") {
        
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmStandardPhysics_option1(1);
        
    } else if (name == "emstandard_opt2") {
        
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmStandardPhysics_option2(1);
        
    } else if (name == "emstandard_opt3") {
        
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmStandardPhysics_option3(1);

    } else if (name == "emstandard_opt4") {
        
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmStandardPhysics_option4(1);

    } else if (name == "emlowenergy") {
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmLowEPPhysics(1);

    } else if (name == "emstandardSS") {
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmStandardPhysicsSS(1);

    } else if (name == "emstandardWVI") {
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmStandardPhysicsWVI(1);

    } else if (name == "emstandardGS") {
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmStandardPhysicsGS(1);

    } else if (name == "emlivermore") {
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmLivermorePhysics(1);

    } else if (name == "empenelope") {
        fEmName = name;
        delete fEmPhysicsList;
        fEmPhysicsList = new G4EmPenelopePhysics(1);

    } else {
    
        G4ExceptionDescription description;
        description
          << "      "
          << "PhysicsList::AddPhysicsList: <" << name << "> is not defined";
        G4Exception("PhysicsList::AddPhysicsList",
                "electronScattering2_F001", FatalException, description);
    }
}