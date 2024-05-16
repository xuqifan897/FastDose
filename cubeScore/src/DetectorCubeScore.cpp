#include "DetectorCubeScore.h"
#include "ArgsCubeScore.h"
#include "ParamsCubeScore.h"

#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Material.hh"
#include "G4ThreeVector.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4PVParameterised.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4PSEnergyDeposit3D.hh"

#include <iomanip>

cube_score::DetectorConstruction::DetectorConstruction():
    G4VUserDetectorConstruction() {
    // material construction
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* air = nist->FindOrBuildMaterial("G4_AIR");
    this->MaterialMap[std::string("Air")] = air;

    G4Element* elH = new G4Element(
        std::string("Hydrogen"), std::string("H"), 1., 1.01*g/mole);
    G4Element* elO = new G4Element(
        std::string("Oxygen"), std::string("O"), 8., 16.00*g/mole);
    for (const std::pair<std::string, float>& entry : MaterialDensity) {
        const std::string& material_name = entry.first;
        float density = entry.second;
        G4Material* material = new G4Material(material_name, density, 2);
        material->AddElement(elH, 2);
        material->AddElement(elO, 1);
        this->MaterialMap[material_name] = material;
    }
    #if true
    std::cout << "Material information:\n" << std::setfill(' ');
    for (const auto& pair : this->MaterialMap)
        std::cout << pair.second << std::endl;
    #endif
    return;
}


G4VPhysicalVolume* cube_score::DetectorConstruction::Construct() {
    // calculate z dimension
    this->dimension_x = getarg<int>("DimXY");
    this->dimension_y = this->dimension_x;
    this->dimension_z = 0;
    for (const auto& entry : SlabPhantom)
        this->dimension_z += entry.second;
    
    float VoxelSize = getarg<float>("VoxelSize") * cm;
    float size_x = this->dimension_x * VoxelSize;
    float size_y = this->dimension_y * VoxelSize;
    float size_z = this->dimension_z * VoxelSize;

    G4String envelopeName{"envelope"};
    G4ThreeVector envelopeSize = G4ThreeVector(200*cm, 200*cm, 200*cm);
    G4Box* solidEnvelope = new G4Box(envelopeName, envelopeSize.x()/2, envelopeSize.y()/2, envelopeSize.z()/2);
    G4LogicalVolume* logicalEnvelope = new G4LogicalVolume(solidEnvelope,
        this->MaterialMap[std::string("Air")], envelopeName, 0, 0, 0);
    G4VPhysicalVolume* physEnvelope = new G4PVPlacement(0, G4ThreeVector(),
        logicalEnvelope, envelopeName, 0, false, 0);

    auto worldS = new G4Box("World", size_x, size_y, size_z);
    auto worldLV = new G4LogicalVolume(worldS, this->MaterialMap[std::string("Air")], "World");
    auto worldPV = new G4PVPlacement(
        nullptr,  //no rotation
        G4ThreeVector(),  // at (0, 0, 0)
        worldLV,  // its logical volume,
        "World",  // its name
        logicalEnvelope,  // no mother volume
        false,  // no boolean operator
        0,  // copy number
        false  // checking overlaps
    );

    // Replication of phantom volume
    G4String xRepName("RepX");
    G4VSolid* solXRep = new G4Box(xRepName, VoxelSize, size_y, size_z);
    G4LogicalVolume* logXRep = new G4LogicalVolume(solXRep, this->MaterialMap["Air"], xRepName);
    G4PVReplica* xReplica = new G4PVReplica(xRepName, logXRep,
        worldLV, kXAxis, this->dimension_x, VoxelSize*2);
    
    G4String yRepName("RepY");
    G4VSolid* solYRep = new G4Box(yRepName, VoxelSize, VoxelSize, size_z);
    G4LogicalVolume* logYRep = new G4LogicalVolume(solYRep, this->MaterialMap["Air"], yRepName);
    G4PVReplica* yReplica = new G4PVReplica(yRepName, logYRep,
        logXRep, kYAxis, this->dimension_y, VoxelSize*2);

    G4String zVoxName("PhantomSens");
    G4VSolid* solVoxel = new G4Box(zVoxName, VoxelSize, VoxelSize, VoxelSize);
    this->fLVPhantomSens = new G4LogicalVolume(solVoxel, this->MaterialMap["Air"], zVoxName);
    ParamsCubeScore* params = new ParamsCubeScore(this->MaterialMap);
    G4VPhysicalVolume* physiPhantomSens = 
        new G4PVParameterised(
            zVoxName,
            this->fLVPhantomSens,
            logYRep,
            kUndefined,
            this->dimension_z,
            params
        );

    // Visualization attributes
    G4VisAttributes* PhantomVisAtt = new G4VisAttributes(G4Colour(1.0, 1.0, 0.0));
    this->fLVPhantomSens->SetVisAttributes(PhantomVisAtt);
    return physEnvelope;
}


void cube_score::DetectorConstruction::ConstructSDandField() {
    G4SDManager* pSDman = G4SDManager::GetSDMpointer();
    G4String phantomSDName{"PhantomSD"};

    G4MultiFunctionalDetector* mFDet = new G4MultiFunctionalDetector(phantomSDName);
    pSDman->AddNewDetector(mFDet);
    this->fLVPhantomSens->SetSensitiveDetector(mFDet);

    G4String psName{"totalEDep"};
    G4PSEnergyDeposit3D* scorer = new G4PSEnergyDeposit3D(psName,
        this->dimension_x, this->dimension_y, this->dimension_z);
    mFDet->RegisterPrimitive(scorer);
}