#include "ActionInitialization.h"
#include "PrimaryGeneratorAction.h"
#include "RunAction.h"
#include "EventAction.h"
#include "TrackingAction.h"

namespace fd = fastdose;

void fd::ActionInitialization::BuildForMaster() const {
    SetUserAction(new RunAction);
}

void fd::ActionInitialization::Build() const {
    SetUserAction(new RunAction);
    SetUserAction(new PrimaryGeneratorAction);
    SetUserAction(new EventAction);
    SetUserAction(new TrackingAction);
}