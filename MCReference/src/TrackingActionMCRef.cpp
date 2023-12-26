#include "G4TrackingManager.hh"
#include "G4Track.hh"

#include "TrackingActionMCRef.h"
#include "TrajectoryMCRef.h"

MCRef::TrackingAction::TrackingAction()
:G4UserTrackingAction()
{;}

void MCRef::TrackingAction::PreUserTrackingAction(const G4Track* aTrack)
{
    this->fpTrackingManager->SetStoreTrajectory(true);
    this->fpTrackingManager->SetTrajectory(new Trajectory(aTrack));
}