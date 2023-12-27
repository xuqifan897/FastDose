#ifndef RunActionBS_h
#define RunActionBS_h 1

#include <vector>

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"

namespace bs
{
    class RunAction : public G4UserRunAction
    {
    public:
        RunAction(std::vector<double>* localScore):
            LocalScore(localScore) {}
        virtual ~RunAction() override = default;

        virtual G4Run* GenerateRun();
        void BeginOfRunAction(const G4Run*) override;
        void EndOfRunAction(const G4Run*) override;
    
    private:
        // maintained externally, do not have 
        // to deallocate in the destructor
        std::vector<double>* LocalScore;
    };
}

#endif