#ifndef __STEPMAX_H__
#define __STEPMAX_H__

#include "globals.hh"
#include "G4VDiscreteProcess.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"

class StepMaxMessenger;

namespace fastdose {
    class StepMax : public G4VDiscreteProcess
    {
    public:
        StepMax(const G4String& processName ="UserStepMax");
        virtual ~StepMax();
        
        virtual G4bool IsApplicable(const G4ParticleDefinition&);
        void     SetMaxStep(G4double);
        G4double GetMaxStep() {return fMaxChargedStep;};
        
        virtual G4double PostStepGetPhysicalInteractionLength( const G4Track& track,
                                                            G4double   previousStepSize,
                                                            G4ForceCondition* condition);
        
        virtual G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&);
        
        virtual G4double GetMeanFreePath(const G4Track&, G4double, G4ForceCondition*)
        {return 0.;};     // it is not needed here !
        
    private:
        G4double fMaxChargedStep;
    };
}

#endif