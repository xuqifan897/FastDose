#ifndef __EVENTACTION_H__
#define __EVENTACTION_H__

#include "G4UserEventAction.hh"
#include <atomic>

namespace fastdose
{
    class EventAction : public G4UserEventAction
    {
    public:
        EventAction();
        ~EventAction() = default;

        virtual void EndOfEventAction(const G4Event*);
    
    private:
        float ZTail;
        float ZHead;
    };
}

#endif