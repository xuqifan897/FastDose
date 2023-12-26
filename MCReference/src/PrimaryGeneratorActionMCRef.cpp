#include "PrimaryGeneratorActionMCRef.h"
#include "ArgMCRef.h"

#include "globals.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

#include <fstream>
#include <random>

std::atomic<int> MCRef::PrimaryGeneratorAction::totalCount(0);

MCRef::PrimaryGeneratorAction::PrimaryGeneratorAction() {
    this->fParticleGun = new G4ParticleGun(1);
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4String particleName;
    G4ParticleDefinition* particle =
        particleTable->FindParticle(particleName="gamma");
    this->fParticleGun->SetParticleDefinition(particle);
    
    // set up position
    this->SAD = getarg<float>("SAD") * cm;
    std::array<float, 3> isocenter;
    const std::vector<int> phantomDim = getarg<std::vector<int>>("phantomDim");
    const float& voxelSize = getarg<float>("voxelSize") * cm;  // half voxel size
    isocenter[0] = voxelSize * phantomDim[0];
    isocenter[1] = voxelSize * phantomDim[1];
    isocenter[2] = voxelSize * phantomDim[2];
    G4ThreeVector sourceCoords{0., -this->SAD, 0.};
    this->fParticleGun->SetParticlePosition(sourceCoords);

    this->FmapOn = getarg<float>("FmapOn") * cm;  // half size
    this->logFrequency = getarg<int>("logFrequency");

    // build energy table
    int nParticles = getarg<int>("nParticles");
    const std::string& SpectrumFile = getarg<std::string>("SpectrumFile");
    std::ifstream f(SpectrumFile);
    if (! f.is_open()) {
        std::cerr << "Cannot open the file " << SpectrumFile << std::endl;
        return;
    }
    int nkernels = 0;
    float sum_fluence = 0.;
    std::string tableRow;
    std::string buff;
    std::vector<float> fluence;
    this->energyTable.clear();
    std::vector<float> mu_en;
    std::vector<float> mu;

    while (std::getline(f, tableRow)) {
        if (tableRow == std::string("\n"))
            break;
        std::istringstream iss(tableRow);
        fluence.push_back(0.);
        this->energyTable.push_back(0.);
        mu_en.push_back(0.);
        mu.push_back(0.);

        iss >> this->energyTable.back() >> fluence.back() 
            >> mu.back() >> mu_en.back() >> buff;
        
        sum_fluence += fluence.back();
        nkernels ++;
    }

    // normalize
    float prev = 0.f;
    for (int i=0; i<nkernels; i++) {
        fluence[i] = prev + fluence[i] / sum_fluence;
        prev = fluence[i];
        this->energyTable[i] *= MeV;
    }

    this->countTable.resize(nkernels);
    for (int i=0; i<nkernels; i++) {
        this->countTable[i] = static_cast<int>(nParticles * fluence[i]);
    }

    #if false
        std::cout << std::endl;
        std::cout << "Position: " << sourceCoords / cm << "cm" << std::endl;
        for (int i=0; i<nkernels; i++) {
            std::cout << "Energy: " << this->energyTable[i] / MeV << 
                ", cumulative fraction: " << fluence[i] << ", count: " 
                << this->countTable[i] << std::endl;
        }
        std::cout << std::endl;
    #endif
}


MCRef::PrimaryGeneratorAction::~PrimaryGeneratorAction() {
    delete this->fParticleGun;
}


void MCRef::PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent) {
    int localIdx = totalCount.fetch_add(1);
    int spectrumIdx = 0;
    for (spectrumIdx=0; spectrumIdx<this->countTable.size(); spectrumIdx++) {
        if (this->countTable[spectrumIdx] > localIdx)
            break;
    }
    float localEnergy = this->energyTable[spectrumIdx];
    this->fParticleGun->SetParticleEnergy(localEnergy);

    // logging
    if ((localIdx + 1) % this->logFrequency == 0) {
        G4cout << "Event: " << localIdx + 1 << G4endl;
    }

    // set up direction
    G4ThreeVector directionVector;
    directionVector[0] = 2 * static_cast<float>(std::rand()) / RAND_MAX - 1;
    directionVector[1] = this->SAD;
    directionVector[2] = 2 * static_cast<float>(std::rand()) / RAND_MAX - 1;

    directionVector[0] *= this->FmapOn;
    directionVector[2] *= this->FmapOn;
    directionVector = directionVector.unit();

    this->fParticleGun->SetParticleMomentumDirection(directionVector);
    this->fParticleGun->GeneratePrimaryVertex(anEvent);

    #if False
        const auto& position = this->fParticleGun->GetParticlePosition() / cm;
        const auto& direction = this->fParticleGun->GetParticleMomentumDirection();
        const auto& energy = this->fParticleGun->GetParticleEnergy() / MeV;
        G4cout << "Position: " << position << "cm, direction: " <<
            direction << ", energy" << energy << "MeV" << G4endl;
    #endif
}