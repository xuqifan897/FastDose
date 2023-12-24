#include "G4TrackingManager.hh"
#include "G4Track.hh"

#include "TrackingAction.h"
#include "Trajectory.h"

 namespace fdkg = kernelgen;

fdkg::TrackingAction::TrackingAction()
:G4UserTrackingAction()
{;}

void fdkg::TrackingAction::PreUserTrackingAction(const G4Track* aTrack)
{
    this->fpTrackingManager->SetStoreTrajectory(true);
    this->fpTrackingManager->SetTrajectory(new Trajectory(aTrack));
}