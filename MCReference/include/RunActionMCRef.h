#ifndef __RUNACTIONMCREF_H__
#define __RUNACTIONMCREF_H__

#include <vector>
#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "globals.hh"

namespace MCRef {
    class RunAction : public G4UserRunAction {
    public:
        RunAction(int ssi, int sei, std::vector<G4double>& globalEM);
        virtual ~RunAction() override = default;

        virtual G4Run* GenerateRun();
        void BeginOfRunAction(const G4Run*) override;
        void EndOfRunAction(const G4Run*) override;
    private:
        int scoringStartIdx;
        int scoringEndIdx;
        std::vector<int> phantomDim;
        std::vector<G4double>& globalEnergyMap;
    };
}

#endif