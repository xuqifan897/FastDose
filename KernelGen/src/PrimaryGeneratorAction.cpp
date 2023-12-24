#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "EventInfo.h"

#include "globals.hh"
#include "PrimaryGeneratorAction.h"
#include "ArgKernelGen.h"
#include <atomic>
#include <iomanip>

 namespace fdkg = kernelgen;

std::atomic<long> fdkg::particleCount(0);
std::atomic<bool> fdkg::logFlag(false);

fdkg::PrimaryGeneratorAction::PrimaryGeneratorAction() {
    G4int n_particle = 1;
    this->fParticleGun = new G4ParticleGun(n_particle);

    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4String particleName;
    G4ParticleDefinition* particle 
        = particleTable->FindParticle(particleName="gamma");
    this->fParticleGun->SetParticleDefinition(particle);
    this->fParticleGun->SetParticleMomentum(G4ThreeVector(0.f, 0.f, 1.f));

    // set a dummy energy
    this->fParticleGun->SetParticleEnergy(6.f * MeV);

    int heightDim = getArgKG<int>("heightDim");
    float heightRes = getArgKG<float>("heightRes") * cm;
    auto sourceOffset = G4ThreeVector(0., 0., -heightDim*heightRes);
    this->fParticleGun->SetParticlePosition(sourceOffset);

    this->EnergyCountTable.resize(spectrum.size());
    float cummu = 0.;
    this->nParticles = getArgKG<int>("nParticles");
    for (int i=0; i<this->EnergyCountTable.size(); i++) {
        cummu += spectrum[i].second;
        this->EnergyCountTable[i] = static_cast<long>(cummu * this->nParticles);
    }
    this->EnergyCountTable.back() = this->nParticles;

    bool expected = false;
    bool new_value = true;
    const int width = 24;
    if (logFlag.compare_exchange_strong(expected, new_value)){
        G4cout << "Energy particle table" << G4endl;
        G4cout << std::setw(width) << std::left << "Energy [MeV]" << 
            std::setw(width) << std::left << "Count" << G4endl;
        for (int i=0; i<this->EnergyCountTable.size(); i++)
            G4cout << std::setw(width) << std::left << spectrum[i].first / MeV
                << std::setw(width) << std::left << this->EnergyCountTable[i] << G4endl;
    }

    this->logFreq = getArgKG<int>("logFreq");
}

fdkg::PrimaryGeneratorAction::~PrimaryGeneratorAction() {
    delete this->fParticleGun;
}

void fdkg::PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent) {
    long particleIdx = particleCount.fetch_add(1);
    // determine the particle energy
    float energy = 0.;
    for (int i=0; i<this->EnergyCountTable.size(); i++) {
        if (particleIdx <= this->EnergyCountTable[i]) {
            energy = spectrum[i].first;
            break;
        }
    }
    this->fParticleGun->SetParticleEnergy(energy);
    this->fParticleGun->GeneratePrimaryVertex(anEvent);
    anEvent->SetUserInformation(new EventInfo());

    if ((particleIdx+1) % this->logFreq == 0)
        G4cout << "Progress: " << particleIdx+1 << " / " << this->nParticles << G4endl;
}