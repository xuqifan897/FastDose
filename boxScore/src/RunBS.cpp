#include <string>
#include <iostream>
#include <atomic>

#include "G4SDManager.hh"
#include "G4Threading.hh"
#include "G4THitsMap.hh"
#include "G4Event.hh"

#include "RunBS.h"
#include "PhantomDefBS.h"
#include "argparseBS.h"

std::atomic<size_t> bs::Run::eventCounts(0);

bs::Run::Run()
{
    int dimXY = (*bs::vm)["dimXY"].as<int>();
    int SegZ = (*bs::vm)["SegZ"].as<int>();
    this->logFreq = (*bs::vm)["logFreq"].as<int>();
    this->HitsMaps.clear();

    this->HitsMaps.reserve(dimXY*SegZ);
    for (int i=0; i<dimXY*SegZ; i++)
    {
        std::string name = "SD" + std::to_string(i+1) + "/Edep";
        int id = G4SDManager::GetSDMpointer()->GetCollectionID(name);
        this->HitsMaps.push_back(std::make_tuple(name, id, new G4THitsMap<G4double>()));
    }
}

bs::Run::~Run()
{
    for (int i=0; i<this->HitsMaps.size(); i++)
        delete std::get<2>(this->HitsMaps[i]);
}

void bs::Run::RecordEvent(const G4Event* anEvent)
{
    for (int i=0; i<this->HitsMaps.size(); i++)
    {
        int HCID = std::get<1>(this->HitsMaps[i]);
        auto hitsCollection = static_cast<G4THitsMap<G4double>*>(
            anEvent->GetHCofThisEvent()->GetHC(HCID));
        *std::get<2>(this->HitsMaps[i]) += *hitsCollection;

        // // for debug purposes
        // const auto & hitsmap = *(*hitsCollection).GetMap();
        // for (auto& it : hitsmap)
        // {
        //     if (*(it.second) != 0.0)
        //         std::cout << "i = " << i << ", idx = " << it.first 
        //             << ", value = " << *(it.second) << std::endl;
        // }
    }
    size_t currentIdx = eventCounts.fetch_add(1);
    if (currentIdx % this->logFreq == 0)
        G4cout << "Event number: " << currentIdx << G4endl;
}

void bs::Run::Merge(const G4Run* aRun)
{
    const Run* bRun = static_cast<const Run*>(aRun);
    for (int i=0; i<this->HitsMaps.size(); i++)
        *std::get<2>(this->HitsMaps[i]) += *std::get<2>(bRun->HitsMaps[i]);
}