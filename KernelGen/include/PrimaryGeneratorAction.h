#ifndef __PRIMARYGENERATORACTION_H__
#define __PRIMARYGENERATORACTION_H__

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include <atomic>

namespace kernelgen {
    class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
    public:
        PrimaryGeneratorAction();
        ~PrimaryGeneratorAction() override;

        void GeneratePrimaries(G4Event*) override;
    
    private:
        G4ParticleGun* fParticleGun;
        std::vector<long> EnergyCountTable;
        int logFreq;
        long nParticles;
    };

    extern std::atomic<long> particleCount;
    extern std::atomic<bool> logFlag;
}

#endif