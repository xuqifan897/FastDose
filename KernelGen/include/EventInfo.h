#ifndef __EVENTINFO_H__
#define __EVENTINFO_H__

#include "G4VUserEventInformation.hh"
#include "globals.hh"
#include "G4SystemOfUnits.hh"

namespace fastdose
{
    class EventInfo : public G4VUserEventInformation
    {
    public:
        EventInfo(): InitInteractionZ(0.), IfInteraction(false) {}
        ~EventInfo() = default;
    
        double& GetInitInteractionZ() {return this->InitInteractionZ;}
        void SetInitInteractionZ(double iiz) {this->InitInteractionZ = iiz;}
        bool& GetIfInteraction() {return this->IfInteraction;}
        void SetIfInteraction(bool ii) {this->IfInteraction = ii;}
        virtual void Print() const override
        {
            G4cout << (this->IfInteraction ? 
                G4String("The initial interaction happened, at z coordinate ") + 
                std::to_string(this->InitInteractionZ / cm) + G4String(" cm"): 
                "The initial interaction did not happen") << G4endl;
        }

    private:
        double InitInteractionZ;
        bool IfInteraction;
    };
}

#endif