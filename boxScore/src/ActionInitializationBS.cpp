#include "ActionInitializationBS.h"
#include "PrimaryGeneratorActionBS.h"
#include "RunActionBS.h"

void bs::ActionInitialization::Build() const
{
    SetUserAction(new PrimaryGeneratorAction);
    SetUserAction(new RunAction(this->LocalScore));
}

void bs::ActionInitialization::BuildForMaster() const
{
    SetUserAction(new RunAction(this->LocalScore));
}