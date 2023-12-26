#include "ActionInitializationMCRef.h"
#include "PrimaryGeneratorActionMCRef.h"
#include "RunActionMCRef.h"
#include "TrackingActionMCRef.h"

void MCRef::ActionInitialization::BuildForMaster() const
{
    SetUserAction(new RunAction(this->scoringStartIdx,
        this->scoringEndIdx, this->globalEnergyMap));
}

void MCRef::ActionInitialization::Build() const
{
    SetUserAction(new PrimaryGeneratorAction);
    SetUserAction(new RunAction(this->scoringStartIdx,
        this->scoringEndIdx, this->globalEnergyMap));
    // SetUserAction(new TrackingAction);
}