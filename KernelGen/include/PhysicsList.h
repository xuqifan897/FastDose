#ifndef __PHYSICSLIST_H__
#define __PHYSICSLIST_H__

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class G4VPhysicsConstructor;
class PhysicsListMessenger;

namespace kernelgen {
    class PhysicsList : public G4VModularPhysicsList {
    public:
        PhysicsList();
        ~PhysicsList() override;

        void ConstructParticle() override;
        void ConstructProcess() override;

        void AddPhysicsList(const G4String& name);

        void AddDecay();
        void AddStepMax();

    private:
        G4VPhysicsConstructor* fEmPhysicsList = nullptr;
        G4String fEmName = "";
    };
}

#endif