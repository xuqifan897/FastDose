#include "RunAction.h"
#include "Run.h"
#include "ArgKernelGen.h"

#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
 namespace fdkg = kernelgen;

G4Run* fdkg::RunAction::GenerateRun() {
    return new Run;
}

void fdkg::RunAction::BeginOfRunAction(const G4Run* aRun) {
    if (this->isMaster)
        G4cout << "### Run: " << aRun->GetRunID() << "starts." << G4endl;
    G4RunManager::GetRunManager()->SetRandomNumberStore(false);
}

void fdkg::RunAction::EndOfRunAction(const G4Run* aRun) {
    if (this->isMaster) {
        const Run* masterRun = static_cast<const Run*>(aRun);
        std::string outputFolder = getArgKG<std::string>("outputFolder");
        if (! fs::exists(outputFolder)) {
            fs::create_directory(outputFolder);
        }

        std::stringstream metadataSS;
        metadataSS << "Number of events: " << getArgKG<int>("nParticles") << std::endl;
        metadataSS << "Number of valid events (events whose initial "
            "interactions are within the valid range): " << masterRun->validCount << std::endl;
        metadataSS << "Phantom dimension (height, radius) = (" << getArgKG<int>("heightDim") 
            << ", " << getArgKG<int>("radiusDim") << ")" << std::endl;
        metadataSS << "Phantom resolution (height, radius) = (" << getArgKG<float>("heightRes") 
            << ", " << getArgKG<float>("radiusRes") << ")" << std::endl;
        metadataSS << "Kernel back-scatter region dimension: " << getArgKG<int>("marginTail") << std::endl;
        metadataSS << "Kernel forward-scatter region dimension: " << getArgKG<int>("marginHead") << std::endl;

        std::string metadata = metadataSS.str();
        fs::path metadataFile = fs::path(outputFolder) / std::string("metadata.txt");
        std::ofstream f(metadataFile.string());
        if (! f.is_open()) {
            G4cerr << "Could not open file: " << metadataFile.string() << G4endl;
            exit(1);
        }
        f << metadata;
        f.close();

        const auto& kernel = masterRun->GetKernel();
        fs::path kernelFile = fs::path(outputFolder) / std::string("kernel.bin");
        f.open(kernelFile.string());
        if (! f.is_open()) {
            G4cerr << "Could not open file " << kernelFile.string() << G4endl;
            exit(1);
        }
        for (int i=0; i<kernel.size(); i++) {
            f.write((char*)kernel[i].data(), kernel[i].size()*sizeof(double));
        }
        f.close();
    }
}