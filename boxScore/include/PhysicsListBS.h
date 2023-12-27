#ifndef PhysicsListBS_h
#define PhysicsListBS_h 1

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class G4VPhysicsConstructor;
class PhysicsListMessenger;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

namespace bs
{
    class PhysicsList: public G4VModularPhysicsList
    {
    public:
    PhysicsList();
    ~PhysicsList() override;

    void ConstructParticle() override;
    void ConstructProcess() override;
            
    void AddPhysicsList(const G4String& name);
        
    void AddDecay();
    void AddStepMax();       
        
    private:
    
    G4VPhysicsConstructor*  fEmPhysicsList = nullptr;

    G4String fEmName = "";
    };
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif