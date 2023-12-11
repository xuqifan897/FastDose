#ifndef __RUNACTION_H__
#define __RUNACTION_H__

#include "G4UserRunAction.hh"
#include "globals.hh"

class G4Run;

namespace fastdose {
    class RunAction : public G4UserRunAction {
    public:
        RunAction() = default;
        ~RunAction() override = default;

        virtual G4Run* GenerateRun();
        void BeginOfRunAction(const G4Run*) override;
        void EndOfRunAction(const G4Run*) override;
    };
}

#endif