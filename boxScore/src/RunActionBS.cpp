#include "G4Run.hh"
#include "G4RunManager.hh"

#include "RunActionBS.h"
#include "RunBS.h"
#include "argparseBS.h"
#include "PhantomDefBS.h"

G4Run* bs::RunAction::GenerateRun()
    {return new Run();}

void bs::RunAction::BeginOfRunAction(const G4Run* aRun)
{
    if (this->isMaster)
        G4cout << "### Run: " << aRun->GetRunID() << "starts." << G4endl;
    G4RunManager::GetRunManager()->SetRandomNumberStore(false);
}

void bs::RunAction::EndOfRunAction(const G4Run* aRun)
{
    if (this->isMaster)
    {
        const Run* masterRun = static_cast<const Run *>(aRun);
        auto& HitsMaps = masterRun->getHitsMaps();
        int barSize = (*bs::vm)["dimXY"].as<int>();
        size_t offset = 0;
        for (int i=0; i<HitsMaps.size(); i++)
        {
            const auto & hitsmap = *std::get<2>(HitsMaps[i]);
            for (auto& it : *(hitsmap.GetMap()))
                (*this->LocalScore)[offset + it.first] = *(it.second);
            offset += barSize;
        }
    }
}