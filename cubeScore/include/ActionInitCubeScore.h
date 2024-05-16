#ifndef __ACTIONINITCUBESCORE_H__
#define __ACTIONINITCUBESCORE_H__

#include "globals.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "G4THitsMap.hh"
#include "G4VUserEventInformation.hh"
#include "G4UserEventAction.hh"
#include "G4UserEventAction.hh"

#include <atomic>
#include <vector>

namespace cube_score {
    extern std::atomic<int> particleCount;

    class ActionInitialization: public G4VUserActionInitialization {
    public:
        ActionInitialization(std::vector<double>* result):
            G4VUserActionInitialization(), Result(result) {}

        virtual void Build() const override;
        virtual void BuildForMaster() const override;
    private:
        std::vector<double>* Result;
    };


    class PrimaryGeneratorAction: public G4VUserPrimaryGeneratorAction {
    public:
        PrimaryGeneratorAction();
        ~PrimaryGeneratorAction();
        virtual void GeneratePrimaries(G4Event*) override;

    private:
        int tellInterval(int phoCnt, bool& change);
        G4ParticleGun* fParticleGun;
        float FluenceSize;  // half fluence size
        float SAD;
        int logFreq;
    };


    class RunAction: public G4UserRunAction {
    public:
        RunAction(std::vector<double>* result);
        virtual G4Run* GenerateRun() override;
        virtual void BeginOfRunAction(const G4Run*) override;
        virtual void EndOfRunAction(const G4Run*) override;
    private:
        std::vector<G4String> fSDName;
        std::vector<double>* Result;
    };


    class Run: public G4Run {
    public:
        Run(const std::vector<G4String> mfdName);
        virtual ~Run() override;
        virtual void RecordEvent(const G4Event*) override;
        virtual void Merge(const G4Run*) override;
        G4int GetNumberOfHitsMap() const {return this->fRunMap.size();}
        G4THitsMap<G4double>* GetHitsMap(G4int i){return fRunMap[i];}
        G4THitsMap<G4double>* GetHitsMap(const G4String& detName, 
                                        const G4String& colName);
        G4THitsMap<G4double>* GetHitsMap(const G4String& fullName);

    private:
        std::vector<G4String> fCollName;
        std::vector<G4int> fCollID;
        std::vector<G4THitsMap<G4double>*> fRunMap;
    };
}

#endif