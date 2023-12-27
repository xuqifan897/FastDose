#ifndef ActionInitializationBS_h
#define ActionInitializationBS_h 1

#include "globals.hh"
#include "G4VUserActionInitialization.hh"

#include <vector>

namespace bs
{
    class ActionInitialization : public G4VUserActionInitialization
    {
    public:
        ActionInitialization(std::vector<double>* localScore): 
            LocalScore(localScore) {}
        virtual ~ActionInitialization() override = default;

        virtual void Build() const;
        virtual void BuildForMaster() const;
    
    private:
        std::vector<double>* LocalScore;
    };
}

#endif