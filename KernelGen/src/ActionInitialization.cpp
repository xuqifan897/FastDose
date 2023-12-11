#include "ActionInitialization.h"
#include "PrimaryGeneratorAction.h"

namespace fd = fastdose;

void fd::ActionInitialization::BuildForMaster() const {
    // SetUserAction(new RunAction);
}

void fd::ActionInitialization::Build() const {
    SetUserAction(new PrimaryGeneratorAction);
}