#ifndef __DETECTORCONSTRUCTION_H__
#define __DETECTORCONSTRUCTION_H__

#include "G4VUserDetectorConstruction.hh"

namespace fastdose {
    class DetectorConstruction : public G4VUserDetectorConstruction {
    public:
        DetectorConstruction() = default;
        ~DetectorConstruction() = default;

        G4VPhysicalVolume* Construct() override;
        void ConstructSDandField() override;
    
    private:
        std::vector<G4LogicalVolume*> SenseDetList;
    };
}

#endif