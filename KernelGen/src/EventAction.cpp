#include "EventAction.h"
#include "ArgKernelGen.h"
#include "Trajectory.h"
#include "TrajectoryPoint.h"

#include "G4TrajectoryContainer.hh"
#include "G4Event.hh"
#include "EventInfo.h"

namespace fd = fastdose;

fd::EventAction::EventAction() {
    int heightDim = getArgKG<int>("heightDim");
    float heightRes = getArgKG<float>("heightRes") * cm;

    int marginTail = getArgKG<int>("marginTail");
    int marginHead = getArgKG<int>("marginHead");
    this->ZTail = (-heightDim + 2 * marginTail) * heightRes;
    this->ZHead = (heightDim - 2 * marginHead) * heightRes;
}

void fd::EventAction::EndOfEventAction(const G4Event* evt) {
    G4TrajectoryContainer* trajectoryContainer = evt->GetTrajectoryContainer();
    int n_trajectories = 0;
    if (trajectoryContainer)
        n_trajectories = trajectoryContainer->entries();

#if false
    for (int i=0; i<trajectoryContainer->size(); i++) {
        auto* VTrj = (*trajectoryContainer)[i];
        Trajectory* trj = static_cast<Trajectory*>(VTrj);
        trj->ShowTrajectory();
        G4cout << G4endl << G4endl;
    }
#endif
    
    // Assume the first trajectory corresponds to the primary particle
    auto* VTrj = (*trajectoryContainer)[0];
    Trajectory* trj = static_cast<Trajectory*>(VTrj);

    if (trj->GetTrackID() != 1) {
        G4cerr << "The first trajectory does not correspond to the primary particle." << G4endl;
        exit(1);
    }

    int interactionIdx = 0;
    bool interaction_ = false;

    if(trj->GetFlag()) {
        interactionIdx = trj->GetInterIdx();
        interaction_ = true;
    }

    //  If no interaction happened, or all trajectory 
    //  points are on the original direction, probably in the last interaction, 
    //  all energy is released to secondary particles.
    if (! interaction_) {
        const auto VLastPoint = trj->GetPoint(trj->GetPointEntries()-1);
        TrajectoryPoint* point = static_cast<TrajectoryPoint*>(VLastPoint);
        const G4ThreeVector& position = point->GetPosition();
        if (position[2] < this->ZHead) {
            interaction_ = true;
            interactionIdx = trj->GetPointEntries()-1;
        }
    }

    if (interaction_) {
        G4VTrajectoryPoint* VInterPoint = trj->GetPoint(interactionIdx);
        TrajectoryPoint* InterPoint = static_cast<TrajectoryPoint*>(VInterPoint);
        double interZ = InterPoint->GetPosition()[2];

        G4VUserEventInformation* VUserInfo = evt->GetUserInformation();
        EventInfo* UserInfo = static_cast<EventInfo*>(VUserInfo);
        UserInfo->SetIfInteraction(true);
        UserInfo->SetInitInteractionZ(interZ);
    }
}