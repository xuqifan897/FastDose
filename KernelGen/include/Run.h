#ifndef __RUN_H__
#define __RUN_H__

#include <vector>
#include <atomic>

#include "G4Run.hh"
#include "G4Event.hh"
#include "G4THitsMap.hh"
#include "RunAction.h"

namespace fastdose {
    class Run : public G4Run
    {
    public:
        Run();
        ~Run() = default;

        virtual void RecordEvent(const G4Event*);
        virtual void Merge(const G4Run*);

        const std::vector<std::vector<double>>&
            GetKernel() const
            {return this->kernel;}

        friend RunAction;
    
    private:
        std::vector<std::pair<std::string, int>> HitsCollectionTable;
        std::vector<std::vector<double>> kernel;
        float ZTail;
        float ZHead;
        int marginTail;
        int marginHead;
        int radiusDim;
        float halfPhantomSize;
        float heightRes;
        long validCount;
    };
}

#endif