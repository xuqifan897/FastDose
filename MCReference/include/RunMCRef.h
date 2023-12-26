#ifndef __RUNMCREF_H__
#define __RUNMCREF_H__

#include "G4Run.hh"
#include "G4THitsMap.hh"

namespace MCRef {
    class Run : public G4Run {
    public:
        Run(int ssi, int sei);
        ~Run() = default;

        virtual void RecordEvent(const G4Event*);
        virtual void Merge(const G4Run*);

        const std::vector<G4double>& getEnergyMap() const {
            return this->EnergyMap;
        }

    private:
        std::vector<G4double> EnergyMap;
        std::vector<int> HCIDTable;
        std::array<int, 3> phantomDim;
        int scoringStartIdx;
        int scoringEndIdx;
    };
}

#endif