#ifndef __DETECTORCUBESCORE_H__
#define __DETECTORCUBESCORE_H__

#include "globals.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4MultiFunctionalDetector.hh"

namespace cube_score {
    class DetectorConstruction: public G4VUserDetectorConstruction {
    public:
        DetectorConstruction();
        virtual G4VPhysicalVolume* Construct() override; 
        virtual void ConstructSDandField() override;
    private:
        std::map<std::string, G4Material*> MaterialMap;
        G4LogicalVolume* fLVPhantomSens;
        int dimension_x;
        int dimension_y;
        int dimension_z;
    };
}

#endif