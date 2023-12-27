#ifndef PrimaryGeneratorActionBS_h
#define PrimaryGeneratorActionBS_h 1

#include <atomic>

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"

namespace bs
{
    class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
    {
    public:
        PrimaryGeneratorAction();
        virtual ~PrimaryGeneratorAction() override;

        void GeneratePrimaries(G4Event*) override;

        int tellInterval(size_t phoCnt, bool& change);
    
    private:
        G4ParticleGun* fParticleGun;
        float beamletSize;  // half value
        float SAD;

        static std::atomic<size_t> particleCount;
    };
}

#endif