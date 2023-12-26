#include "G4Run.hh"
#include "G4RunManager.hh"

#include "RunActionMCRef.h"
#include "RunMCRef.h"
#include "ArgMCRef.h"

MCRef::RunAction::RunAction(int ssi, int sei, std::vector<G4double>& globalEM):
    scoringStartIdx(ssi), scoringEndIdx(sei), globalEnergyMap(globalEM)
{
    this->phantomDim = getarg<std::vector<int>>("phantomDim");
    int scoringDimY = this->scoringEndIdx - this->scoringStartIdx;
    size_t scoringVolume = phantomDim[0] * scoringDimY * phantomDim[2];
}

G4Run* MCRef::RunAction::GenerateRun() {
    return new Run(this->scoringStartIdx, this->scoringEndIdx);
}

void MCRef::RunAction::BeginOfRunAction(const G4Run* aRun) {
    if (this->isMaster)
        G4cout << "### Run: " << aRun->GetRunID() << "starts." << G4endl;
    G4RunManager::GetRunManager()->SetRandomNumberStore(false);
}

void MCRef::RunAction::EndOfRunAction(const G4Run* aRun) {
    if (this->isMaster) {
        const Run* masterRun = static_cast<const Run *>(aRun);
        const std::vector<G4double>& masterEnergyMap = masterRun->getEnergyMap();
        
        for (int i=0; i<this->globalEnergyMap.size(); i++) {
            this->globalEnergyMap[i] = masterEnergyMap[i];
        }
    }
}