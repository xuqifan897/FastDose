#ifndef __TRACKINGACTIONMCREF_H__
#define __TRACKINGACTIONMCREF_H__

#include "G4UserTrackingAction.hh"

namespace MCRef
{

    class TrackingAction : public G4UserTrackingAction
    {
    public:
        TrackingAction();
        virtual ~TrackingAction() = default;

        virtual void PreUserTrackingAction(const G4Track*);
    };
}

#endif