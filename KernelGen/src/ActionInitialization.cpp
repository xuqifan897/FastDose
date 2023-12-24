#include "ActionInitialization.h"
#include "PrimaryGeneratorAction.h"
#include "RunAction.h"
#include "EventAction.h"
#include "TrackingAction.h"

 namespace fdkg = kernelgen;

void fdkg::ActionInitialization::BuildForMaster() const {
    SetUserAction(new RunAction);
}

void fdkg::ActionInitialization::Build() const {
    SetUserAction(new RunAction);
    SetUserAction(new PrimaryGeneratorAction);
    SetUserAction(new EventAction);
    SetUserAction(new TrackingAction);
}