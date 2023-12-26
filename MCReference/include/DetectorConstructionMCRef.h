#ifndef __DETECTORCONSTRUCTIONMCREF_H__
#define __DETECTORCONSTRUCTIONMCREF_H__

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"
#include <vector>
#include <array>

class G4VPhysicalVolume;
class G4LogicalVolume;

namespace MCRef {
    class DetectorConstruction : public G4VUserDetectorConstruction {
    public:
        DetectorConstruction(int scoringStartIdx, int scoringEndIdx);
        
        G4VPhysicalVolume* Construct() override;
        void ConstructSDandField() override;   
    
    private:
        std::vector<G4LogicalVolume*> logicals;
        std::vector<float> unique_density_values;

        std::vector<int> layerDensity;
        std::vector<float> layerOffset;
        std::vector<bool> scoringFlag;
        
        std::array<int, 3> dimension;

        int scoringStartIdx;
        int scoringEndIdx;
        int superSampling;
    };
}

#endif