#include "StepMax.h"

#include "StepMax.h"

 namespace fdkg = kernelgen;
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

fdkg::StepMax::StepMax(const G4String& processName)
: G4VDiscreteProcess(processName), fMaxChargedStep(DBL_MAX)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

fdkg::StepMax::~StepMax() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool fdkg::StepMax::IsApplicable(const G4ParticleDefinition& particle)
{
    return (particle.GetPDGCharge() != 0.);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void fdkg::StepMax::SetMaxStep(G4double step) {fMaxChargedStep = step;}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double fdkg::StepMax::PostStepGetPhysicalInteractionLength(const G4Track&,
                                                       G4double,
                                                       G4ForceCondition* condition )
{
    // condition is set to "Not Forced"
    *condition = NotForced;
    
    G4double ProposedStep = DBL_MAX;
    
    if(fMaxChargedStep > 0.) ProposedStep = fMaxChargedStep;
    
    return ProposedStep;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VParticleChange* fdkg::StepMax::PostStepDoIt(const G4Track& aTrack, const G4Step&)
{
    // do nothing
    aParticleChange.Initialize(aTrack);
    return &aParticleChange;
}