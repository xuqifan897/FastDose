#include "G4SDManager.hh"
#include "G4Event.hh"
#include "G4THitsMap.hh"

#include "RunMCRef.h"
#include "ArgMCRef.h"

MCRef::Run::Run(int ssi, int sei): G4Run(), scoringStartIdx(ssi), scoringEndIdx(sei) {
    const std::vector<int> phantomDim_ = getarg<std::vector<int>>("phantomDim");
    this->phantomDim[0] = phantomDim_[0];
    this->phantomDim[1] = phantomDim_[1];
    this->phantomDim[2] = phantomDim_[2];
    int scoringDimY = (this->scoringEndIdx - this->scoringStartIdx) * getarg<int>("superSampling");
    size_t scoringVolume = this->phantomDim[0] * scoringDimY * this->phantomDim[2];
    this->EnergyMap.resize(scoringVolume);

    size_t nBars = scoringDimY * this->phantomDim[0];
    this->HCIDTable.resize(nBars);
    auto* pSDman = G4SDManager::GetSDMpointer();
    for (int i=0; i<nBars; i++) {
        G4String name = G4String("det") + std::to_string(i) + G4String("/Edep");
        this->HCIDTable[i] = pSDman->GetCollectionID(name);
    }

    // #if MCDebug
    //     for (int i=0; i<nBars; i++) {
    //         G4cout << this->HCIDTable[i] << " ";
    //     }
    //     G4cout << G4endl;
    // #endif
}

void MCRef::Run::RecordEvent(const G4Event* anEvent) {
    #if MCDebug
        G4TrajectoryContainer* traj = anEvent->GetTrajectoryContainer();
        int n_trajectories = 0;
        if (traj)
            n_trajectories = traj->entries();
        
        for (int i=0; i<n_trajectories; i++) {
            (*traj)[i]->ShowTrajectory();
        }
    #endif
    for (int i=0; i<this->HCIDTable.size(); i++) {
        int HCID = this->HCIDTable[i];
        const auto & hitsCollection = static_cast<G4THitsMap<G4double>*>(
            anEvent->GetHCofThisEvent()->GetHC(HCID));
        size_t displacement = i * this->phantomDim[2];

        for (const auto& it : *(hitsCollection->GetMap())) {
            this->EnergyMap[displacement + it.first] += *(it.second);
        }
    }
}

void MCRef::Run::Merge(const G4Run* aRun) {
    const Run* bRun = static_cast<const Run*>(aRun);
    for (int i=0; i<this->EnergyMap.size(); i++) {
        this->EnergyMap[i] += bRun->EnergyMap[i];
    }
}