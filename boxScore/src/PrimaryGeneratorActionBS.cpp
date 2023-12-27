#include <atomic>

#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "Randomize.hh"

#include "PrimaryGeneratorActionBS.h"
#include "argparseBS.h"
#include "PhantomDefBS.h"

std::atomic<size_t> bs::PrimaryGeneratorAction::particleCount(0);

bs::PrimaryGeneratorAction::PrimaryGeneratorAction()
{
    this->fParticleGun = new G4ParticleGun(1);

    float energy = (*vm)["Energy"].as<float>() * MeV;
    this->SAD = (*vm)["SAD"].as<float>() * cm;
    this->beamletSize = (*vm)["beamlet-size"].as<float>() * cm;

    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    std::string particleName;
    G4ParticleDefinition* particle 
        = particleTable->FindParticle(particleName="gamma");
    this->fParticleGun->SetParticleDefinition(particle);
    this->fParticleGun->SetParticleEnergy(energy);
    this->fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., - this->SAD));
}

bs::PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete this->fParticleGun;
}

void bs::PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    // Energy setting
    size_t phoCnt = particleCount.fetch_add(1);
    bool change;
    int whichEnergy = this->tellInterval(phoCnt, change);
    float energy = std::get<0>(bs::Spec[whichEnergy]) * MeV;
    this->fParticleGun->SetParticleEnergy(energy);
    // momentum sampling
    float isoplaneX = this->beamletSize * (G4UniformRand() - 0.5) * 2;
    float isoplaneY = this->beamletSize * (G4UniformRand() - 0.5) * 2;
    this->fParticleGun->SetParticleMomentumDirection(G4ThreeVector(isoplaneX, isoplaneY, this->SAD));
    this->fParticleGun->GeneratePrimaryVertex(anEvent);
    
    if (change)
        G4cout << "Photon count: " << phoCnt << ", Energy: " 
            << energy / MeV << " MeV." << G4endl;
}

int bs::PrimaryGeneratorAction::tellInterval(size_t phoCnt, bool& change)
{
    for (int i=0; i<bs::Spec.size(); i++)
        if (std::get<3>(Spec[i]) > phoCnt)
        {
            change = std::get<3>(Spec[i]) == phoCnt+1;
            return i;
        }
    G4cerr << "Photon index: " << phoCnt << ", last milestone: " << 
        std::get<3>(bs::Spec.back()) << ". Photon index out of the range!" << G4endl;
    return -1;
}