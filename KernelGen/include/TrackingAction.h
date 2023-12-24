#ifndef __TRACKINGACTION_H__
#define __TRACKINGACTION_H__

#include "G4UserTrackingAction.hh"

namespace kernelgen {
    class TrackingAction : public G4UserTrackingAction {
    public:
        TrackingAction();
        virtual ~TrackingAction() = default;

        virtual void PreUserTrackingAction(const G4Track*);
    };
}


#endif