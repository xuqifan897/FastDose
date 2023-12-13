#ifndef __TRAJECTORYPOINT_H__   
#define __TRAJECTORYPOINT_H__

#include "G4VTrajectoryPoint.hh"

namespace fastdose
{
    class TrajectoryPoint : public G4VTrajectoryPoint
    {
    public:
        TrajectoryPoint(G4ThreeVector pos, G4ThreeVector mom, bool flag=false):
            fPosition(pos), fMomentum(mom), fFlag(flag) {}
        
        inline const G4ThreeVector GetPosition() const
            {return this->fPosition;}
        inline const G4ThreeVector GetMomentum() const 
            {return this->fMomentum;}
        inline const bool GetFlag() const
            {return this->fFlag;}
    private:
        G4ThreeVector fPosition;
        G4ThreeVector fMomentum;
        bool fFlag;
    };
}

#endif