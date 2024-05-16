#include "ActionInitCubeScore.h"
#include "ArgsCubeScore.h"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"
#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4Event.hh"
#include "G4Threading.hh"

// ================================================================
// Action Initialization
// ================================================================
std::atomic<int> cube_score::particleCount(0);

void cube_score::ActionInitialization::Build() const {
    SetUserAction(new PrimaryGeneratorAction());
    SetUserAction(new RunAction(this->Result));
}

void cube_score::ActionInitialization::BuildForMaster() const {
    SetUserAction(new RunAction(this->Result));
}


// ================================================================
// Primary Generator Action
// ================================================================
cube_score::PrimaryGeneratorAction::PrimaryGeneratorAction():
    G4VUserPrimaryGeneratorAction(),
    fParticleGun(0) {
    G4int nParticle = 1;
    this->fParticleGun = new G4ParticleGun(nParticle);

    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particle = particleTable->FindParticle("gamma");
    this->fParticleGun->SetParticleDefinition(particle);

    this->SAD = getarg<float>("SAD") * cm;
    this->fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., -this->SAD));

    this->FluenceSize = getarg<float>("FluenceSize") * cm;

    this->logFreq = getarg<int>("logFreq");
}

cube_score::PrimaryGeneratorAction::~PrimaryGeneratorAction() {
    delete this->fParticleGun;
}

void cube_score::PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent) {
    int phoCnt = particleCount.fetch_add(1);
    bool change;
    int whichEnergy = this->tellInterval(phoCnt, change);
    G4double energy = Spectrum[whichEnergy].first;
    // momentum sampling
    float isoplaneX = this->FluenceSize * (G4UniformRand() - 0.5f) * 2;
    float isoplaneY = this->FluenceSize * (G4UniformRand() - 0.5f) * 2;
    G4ThreeVector direction(isoplaneX, isoplaneY, this->SAD);
    this->fParticleGun->SetParticleEnergy(energy);
    this->fParticleGun->SetParticleMomentumDirection(direction);
    this->fParticleGun->GeneratePrimaryVertex(anEvent);

    if (change)
        G4cout << "Photon count: " << phoCnt << ", Energy: "
            << energy / MeV << " MeV." << G4endl;
    if ((phoCnt + 1) % this->logFreq == 0) {
        G4cout << "Photon count: " << phoCnt + 1 << G4endl;
    }
}

int cube_score::PrimaryGeneratorAction::tellInterval(int phoCnt, bool& change) {
    int cumu = 0;
    for (int i=0; i<Spectrum.size(); i++) {
        int cumu_next = cumu + Spectrum[i].second;
        if (cumu_next > phoCnt) {
            change = std::abs(phoCnt + 1 - cumu_next) < 1e-4f;
            return i;
        }
        cumu = cumu_next;
    }
    G4cerr << "Photon index: " << phoCnt << " out of range!" << G4endl;
    return -1;
}



// ================================================================
// Run Action
// ================================================================
cube_score::RunAction::RunAction(std::vector<double>* result) {
    this->Result = result;
    this->fSDName.push_back(G4String("PhantomSD"));
}

G4Run* cube_score::RunAction::GenerateRun() {
    return new Run(this->fSDName);
}

void cube_score::RunAction::BeginOfRunAction(const G4Run* aRun) {
    if (this->IsMaster())
        G4cout << "### Run starts." << std::endl;
}


void cube_score::RunAction::EndOfRunAction(const G4Run* aRun) {
    if (! IsMaster()) return;

    Run* MasterRun = (Run*)aRun;
    G4THitsMap<G4double>* totEdep = MasterRun->GetHitsMap("PhantomSD/totalEDep");

    size_t dimension_x = getarg<int>("DimXY");
    size_t dimension_y = dimension_x;
    size_t dimension_z = 0;
    for (int i=0; i<SlabPhantom.size(); i++)
        dimension_z += SlabPhantom[i].second;
    
    size_t totalNumVoxels = dimension_x * dimension_y * dimension_z;
    this->Result->resize(totalNumVoxels);
    for (size_t i=0; i<dimension_x; i++) {
        for (size_t j=0; j<dimension_y; j++) {
            for(size_t k=0; k<dimension_z; k++) {
                size_t idx = k + dimension_z * (j + dimension_y * i);
                G4double* totED = (*totEdep)[idx];
                G4double value = 0.0;
                if (totED)
                    value = *totED;
                (*this->Result)[idx] = value;
            }
        }
    }
}


// ================================================================
// Run
// ================================================================
cube_score::Run::Run(const std::vector<G4String> mfdName): G4Run() {
    G4SDManager* pSDman = G4SDManager::GetSDMpointer();
    G4int nMfd = mfdName.size();
    for (G4int idet=0; idet<nMfd; idet++) {
        G4String detName = mfdName[idet];
        G4MultiFunctionalDetector* mfd = (G4MultiFunctionalDetector*)
            (pSDman->FindSensitiveDetector(detName));
        if (mfd) {
            for (G4int icol=0; icol<mfd->GetNumberOfPrimitives(); icol++) {
                G4VPrimitiveScorer* scorer=mfd->GetPrimitive(icol);
                G4String collectionName = scorer->GetName();
                G4String fullCollectionName = detName+"/"+collectionName;
                G4int    collectionID = pSDman->GetCollectionID(fullCollectionName);

                if (collectionID >= 0) {
                    this->fCollName.push_back(fullCollectionName);
                    this->fCollID.push_back(collectionID);
                    this->fRunMap.push_back(new G4THitsMap<double>(detName, collectionName));
                    if (G4Threading::G4GetThreadId() == 0)
                        G4cout << "++ "<<fullCollectionName<< " id " << collectionID << G4endl;
                } else {
                    G4cerr << "Error: collection " << fullCollectionName << " not found." << G4endl;
                }
            }
        }
    }
}

cube_score::Run::~Run() {
    for (int i=0; i<this->fRunMap.size(); i++)
        delete this->fRunMap[i];
}

void cube_score::Run::RecordEvent(const G4Event* anEvent) {
    this->numberOfEvent ++;

    G4HCofThisEvent* pHCE = anEvent->GetHCofThisEvent();
    if (! pHCE) return;
    G4int nCol = this->fCollID.size();
    for (G4int i=0; i<nCol; i++) {
        G4THitsMap<G4double>* evtMap = nullptr;
        if (this->fCollID[i] >= 0)
            evtMap = (G4THitsMap<G4double>*)(pHCE->GetHC(this->fCollID[i]));
        else
            G4cerr << "Error: evtMap not found " << i << G4endl;
        if (evtMap)
            *this->fRunMap[i] += *evtMap;
    }
}

void cube_score::Run::Merge(const G4Run* aRun) {
    const Run* localRun = static_cast<const Run*>(aRun);

    G4int nCol = this->fCollID.size();
    for (G4int i=0; i<nCol; i++)
        *this->fRunMap[i] += *localRun->fRunMap[i];

    G4Run::Merge(aRun);
}

G4THitsMap<G4double>* cube_score::Run::GetHitsMap(const G4String& detName,
                                         const G4String& colName){
    G4String fullName = detName+"/"+colName;
    return GetHitsMap(fullName);
}

G4THitsMap<G4double>* cube_score::Run::GetHitsMap(const G4String& fullName){
    G4int nCol = fCollName.size();
    for ( G4int i = 0; i < nCol; i++)
        if ( this->fCollName[i] == fullName )
            return this->fRunMap[i];
    std::cerr << "Cannot find the collection of name " << fullName << std::endl;
    return nullptr;
}