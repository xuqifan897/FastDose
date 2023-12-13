#include "Run.h"
#include "ArgKernelGen.h"
#include "EventInfo.h"

#include "G4SDManager.hh"
#include "G4Event.hh"
#include <atomic>

namespace fd = fastdose;

fd::Run::Run() {
    // Initialize HitsCollectionTable
    int heightDim = getArgKG<int>("heightDim");
    this->radiusDim = getArgKG<int>("radiusDim");
    this->HitsCollectionTable.resize(heightDim);
    G4SDManager* pSDman = G4SDManager::GetSDMpointer();
    for (int i=0; i<heightDim; i++) {
        std::string HCname = std::string("det") + std::to_string(i) + std::string("/Edep");
        int HCID = pSDman->GetCollectionID(HCname);
        this->HitsCollectionTable[i].first = HCname;
        this->HitsCollectionTable[i].second = HCID;
    }

    this->marginTail = getArgKG<int>("marginTail");
    this->marginHead = getArgKG<int>("marginHead");

    this->kernel = std::vector<std::vector<double>>(
        this->marginTail + this->marginHead,
        std::vector<double>(radiusDim, 0.));

    this->heightRes = getArgKG<float>("heightRes") * cm;
    this->halfPhantomSize = heightDim * heightRes;
    this->ZTail = (-heightDim + 2 * this->marginTail) * this->heightRes;
    this->ZHead = (heightDim - 2 * this->marginHead) * this->heightRes;
    this->validCount = 0;
}

void fd::Run::RecordEvent(const G4Event* anEvent) {
    G4VUserEventInformation* VUserInfo = anEvent->GetUserInformation();
    EventInfo* UserInfo = static_cast<EventInfo*>(VUserInfo);
    bool IfInteraction = UserInfo->GetIfInteraction();
    double InteractionZ = UserInfo->GetInitInteractionZ();

    if (! IfInteraction || InteractionZ < this->ZTail || InteractionZ > this->ZHead)
        return;
    
    this->validCount++;
    int InteractionDim = (int)std::floor(InteractionZ + this->halfPhantomSize)
        / (2 * this->heightRes);
    int sliceBegin = InteractionDim - this->marginTail;
    for (int i=0; i<this->marginTail + this->marginHead; i++) {
        int depthIdx = sliceBegin + i;
        int HCID = this->HitsCollectionTable[depthIdx].second;
        auto hitsCollection = static_cast<G4THitsMap<G4double>*>(
            anEvent->GetHCofThisEvent()->GetHC(HCID));
        const auto & map = *(hitsCollection->GetMap());
        for (const auto& it : map) {
            this->kernel[i][it.first] += *(it.second);
        }
    }
}


void fd::Run::Merge(const G4Run* aRun) {
    const Run* bRun = static_cast<const Run*>(aRun);
    for (int i=0; i<this->kernel.size(); i++) {
        for (int j=0; j<this->radiusDim; j++) {
            this->kernel[i][j] += bRun->kernel[i][j];
        }
    }
    this->validCount += bRun->validCount;
}