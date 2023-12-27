#ifndef RunBS_h
#define RunBS_h

#include <vector>
#include <atomic>

#include "G4Run.hh"
#include "G4THitsMap.hh"

namespace bs
{
    class Run : public G4Run
    {
    public:
        Run();
        ~Run();

        virtual void RecordEvent(const G4Event*);
        virtual void Merge(const G4Run*);
        const std::vector<std::tuple<std::string, int, G4THitsMap<G4double>*>>&
            getHitsMaps() const
            {return this->HitsMaps;}
            
        static std::atomic<size_t> eventCounts;
    private:
        // hit collection id
        std::vector<std::tuple<std::string, int, G4THitsMap<double>*>> HitsMaps;
        int logFreq;
    };
}

#endif