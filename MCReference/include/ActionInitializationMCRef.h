#ifndef __ACTIONINITIALIZATIONMCREF_H__
#define __ACTIONINITIALIZATIONMCREF_H__

#include "G4VUserActionInitialization.hh"
#include "globals.hh"
#include <vector>

namespace MCRef {
    class ActionInitialization : public G4VUserActionInitialization
    {
    public:
        ActionInitialization(int ssi, int sei, std::vector<G4double>& globalEM):
            scoringStartIdx(ssi), scoringEndIdx(sei), globalEnergyMap(globalEM) {}
        ~ActionInitialization() override = default;

        void BuildForMaster() const override;
        void Build() const override;
    
    private:
        int scoringStartIdx;
        int scoringEndIdx;
        std::vector<G4double>& globalEnergyMap;
    };
}

#endif