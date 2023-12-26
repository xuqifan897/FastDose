#ifndef __PRIMARYGENERATORACTION_H__
#define __PRIMARYGENERATORACTION_H__

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include <atomic>

namespace MCRef {
    class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
    public:
        PrimaryGeneratorAction();
        ~PrimaryGeneratorAction();

        void GeneratePrimaries(G4Event*) override;
    
    private:
        G4ParticleGun* fParticleGun;
        static std::atomic<int> totalCount;
        std::vector<int> countTable;
        std::vector<float> energyTable;

        float FmapOn;
        float SAD;
        int logFrequency;
    };
}

#endif