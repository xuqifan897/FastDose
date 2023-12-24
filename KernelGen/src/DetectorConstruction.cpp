#include "DetectorConstruction.h"
#include "ArgKernelGen.h"

#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4MaterialTable.hh"
#include "G4Element.hh"
#include "G4ElementTable.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "G4PVParameterised.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4SDManager.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserLimits.hh"
#include "G4Threading.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4VUserDetectorConstruction.hh"

 namespace fdkg = kernelgen;

G4VPhysicalVolume* fdkg::DetectorConstruction::Construct() {
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* water = nist->FindOrBuildMaterial("G4_WATER");
    G4cout << water << G4endl;
    bool checkOverlaps = false;

    int radiusDim = getArgKG<int>("radiusDim");
    int heightDim = getArgKG<int>("heightDim");
    float radiusRes = getArgKG<float>("radiusRes") * cm;
    float heightRes = getArgKG<float>("heightRes") * cm;

    G4VPhysicalVolume* worldPV = nullptr;
    auto worldSV = new G4Tubs("World", 0., 
        radiusDim*radiusRes, heightDim*heightRes, 0.*deg, 360.*deg);
    auto worldLV = new G4LogicalVolume(worldSV, water, "World");
    worldPV = new G4PVPlacement(
        nullptr,            // no rotation
        G4ThreeVector(),    // no translation
        worldLV,            // its logical volume
        "World",            // its name
        nullptr,            // its mother volume
        false,              // no boolean operator
        0,                  // copy number
        checkOverlaps
    );

    float halfHeightSize = heightDim * heightRes;
    float radiusMax = radiusDim * radiusRes;
    for (int i=0; i<heightDim; i++) {
        G4String sliceName = G4String("Slice") + std::to_string(i);
        auto* sliceSV = new G4Tubs(sliceName, 0.f,
            radiusMax, heightRes, 0.f*deg, 360.f*deg);
        auto* sliceLV = new G4LogicalVolume(sliceSV, water, sliceName);
        G4ThreeVector offset(0.f, 0.f, -halfHeightSize + (2*i+1)*heightRes);
        new G4PVPlacement(
            nullptr,
            offset,
            sliceLV,
            sliceName,
            worldLV,
            false,
            0,
            checkOverlaps
        );

        G4String sliceRepName = G4String("SliceRep") + std::to_string(i);
        auto sliceRepSV = new G4Tubs(sliceRepName, 0.f,
            radiusMax, heightRes, 0.f*deg, 360.f*deg);
        auto sliceRepLV = new G4LogicalVolume(sliceRepSV, water, sliceRepName);
        new G4PVReplica(
            sliceRepName,
            sliceRepLV,
            sliceLV,
            kRho,
            radiusDim,
            radiusRes
        );

        this->SenseDetList.push_back(sliceRepLV);
    }
    return worldPV;
}


void fdkg::DetectorConstruction::ConstructSDandField() {
    G4SDManager* pSDman = G4SDManager::GetSDMpointer();
    for (int i=0; i<this->SenseDetList.size(); i++) {
        G4String SDname = std::string("det") + std::to_string(i);
        G4MultiFunctionalDetector* det = new G4MultiFunctionalDetector(SDname);
        pSDman->AddNewDetector(det);

        auto* primitive = new G4PSEnergyDeposit("Edep");
        det->RegisterPrimitive(primitive);
        SetSensitiveDetector(this->SenseDetList[i], det);
    }
}