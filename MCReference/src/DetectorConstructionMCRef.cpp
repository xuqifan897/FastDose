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

#include <array>
#include <vector>
#include <string>
#include <fstream>

#include "DetectorConstructionMCRef.h"
#include "ArgMCRef.h"

MCRef::DetectorConstruction::DetectorConstruction(
    int scoringStartIdx, int scoringEndIdx):G4VUserDetectorConstruction(),
    scoringStartIdx(scoringStartIdx), scoringEndIdx(scoringEndIdx)
{
    const std::vector<int>& phantomDim = getarg<std::vector<int>>("phantomDim");
    this->dimension = std::array<int, 3>{phantomDim[0], phantomDim[1], phantomDim[2]};
    const std::string& phantomPath = getarg<std::string>("phantomPath");
    size_t phantomSize = phantomDim[0] * phantomDim[1] * phantomDim[2];
    std::vector<float> phantom(phantomSize, 0.);
    std::ifstream f(phantomPath);
    if (! f.is_open()) {
        std::cerr << "Cannot open the file: " << phantomPath << std::endl;
        return;
    }
    f.read((char*)(phantom.data()), phantomSize*sizeof(float));
    f.close();

    // check for different density values
    this->unique_density_values.clear();
    for (size_t i=0; i<phantomSize; i++) {
        float value = phantom[i];
        bool found = false;
        for (int j=0; j<this->unique_density_values.size(); j++) {
            if (std::abs(value - this->unique_density_values[j]) < eps_MCRef) {
                found = true;
                break;
            }
        }
        if (! found) {
            this->unique_density_values.push_back(value);
        }
    }
    std::cout << "Unique density values: " << std::endl;
    for (int i=0; i<this->unique_density_values.size(); i++) {
        std::cout << this->unique_density_values[i] << " ";
    }
    std::cout << std::endl;

    std::vector<int> layerDensity_(phantomDim[1]);
    for (int idx_y=0; idx_y<phantomDim[1]; idx_y++) {
        size_t idx = phantomDim[0] * idx_y;
        float first_voxel_value = phantom[idx];
        // for sanity check
        for (int idx_x=0; idx_x<phantomDim[0]; idx_x++) {
            for (int idx_z=0; idx_z<phantomDim[2]; idx_z++) {
                idx = idx_x + phantomDim[0] * (idx_y + phantomDim[1] * idx_z);
                float local_voxel_value = phantom[idx];
                if (std::abs(first_voxel_value - local_voxel_value) > eps_MCRef) {
                    std::cout << "slice idx: " << idx_y << ", the density values of this slice is not equal" << std::endl;
                    return;
                }
            }
        }
        auto valueIdx = std::find(this->unique_density_values.begin(),
            this->unique_density_values.end(), first_voxel_value);
        layerDensity_[idx_y] = std::distance(this->unique_density_values.begin(), valueIdx);
        #if false
            std::cout << "Slice " << idx_y << ", density index: "
                << this->layerDensity[idx_y] << ", density value: "
                << first_voxel_value << std::endl;
        #endif
    }

    // apply super-sampling
    float voxelSize = getarg<float>("voxelSize") * cm;  // half voxel size
    int layerIdx = 0;
    float layerOffsetValue = - this->dimension[1] * voxelSize;
    this->superSampling = getarg<int>("superSampling");
    int nTotalSlices = scoringStartIdx + (scoringEndIdx - scoringStartIdx) 
        * superSampling + (phantomDim[1] - scoringEndIdx);
    this->layerDensity.resize(nTotalSlices);
    this->layerOffset.resize(nTotalSlices);
    this->scoringFlag.resize(nTotalSlices);
    for (int i=0; i<scoringStartIdx; i++) {
        this->layerDensity[layerIdx] = layerDensity_[i];
        this->layerOffset[layerIdx] = layerOffsetValue + voxelSize;
        this->scoringFlag[layerIdx] = false;
        layerOffsetValue += 2 * voxelSize;
        layerIdx ++;
    }

    for (int i=scoringStartIdx; i<scoringEndIdx; i++) {
        for (int j=0; j<this->superSampling; j++) {
            this->layerDensity[layerIdx] = layerDensity_[i];
            this->layerOffset[layerIdx] = layerOffsetValue + voxelSize / this->superSampling;
            this->scoringFlag[layerIdx] = true;
            layerOffsetValue += 2 * voxelSize / this->superSampling;
            layerIdx ++;
        }
    }

    for (int i=scoringEndIdx; i<this->dimension[1]; i++) {
        this->layerDensity[layerIdx] = layerDensity_[i];
        this->layerOffset[layerIdx] = layerOffsetValue + voxelSize;
        this->scoringFlag[layerIdx] = false;
        layerOffsetValue += 2 * voxelSize;
        layerIdx ++;
    }
}

G4VPhysicalVolume* MCRef::DetectorConstruction::Construct() {
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* air = nist->FindOrBuildMaterial("G4_AIR");
    G4Element* elH = new G4Element(
        std::string("Hydrogen"), std::string("H"), 1., 1.01*g/mole);
    G4Element* elO = new G4Element(
        std::string("Oxygen"), std::string("O"), 8., 16.00*g/mole);

    // for each density, create a new material
    std::vector<G4Material*> materials(this->unique_density_values.size());
    for (int i=0; i<this->unique_density_values.size(); i++) {
        G4double density = this->unique_density_values[i] * g/cm3;
        materials[i] = new G4Material(
            std::string("material") + std::to_string(i),
            density, 2);
        materials[i]->AddElement(elH, 2);
        materials[i]->AddElement(elO, 1);
    }


    float voxelSize = getarg<float>("voxelSize") * cm;  // half voxel size
    float sizeX = voxelSize * this->dimension[0];  // half phantom size X
    float sizeY = voxelSize * this->dimension[1];  // half phantom size Y
    float sizeZ = voxelSize * this->dimension[2];  // half phantom size Z

    // world initialization
    auto worldS = new G4Box("World", sizeX, sizeY, sizeZ);
    auto worldLV = new G4LogicalVolume(worldS, air, "World");
    auto worldPV = new G4PVPlacement(
        nullptr,  // no rotation
        G4ThreeVector(),  // no translation
        worldLV,  // its logical volume
        "World",  // its name
        nullptr,  // no mother volume
        false,    // no boolean operator
        0,        // no copy number
        false     // checking overlaps
    );

    for (int j=0; j<this->layerDensity.size(); j++) {
        float localOffset = this->layerOffset[j];
        int localMaterialIdx = this->layerDensity[j];
        float localScoringFlag = this->scoringFlag[j];
        G4Material* localMaterial = materials[localMaterialIdx];

        if (localScoringFlag) {
            for (int i=0; i<this->dimension[0]; i++) {
                std::string name = std::string("bar_") + std::to_string(j)
                    + std::string("_") + std::to_string(i);
                int displacementX = - sizeX + (2 * i + 1) * voxelSize;
                int displacementY = localOffset;
                auto barS = new G4Box(name, voxelSize, voxelSize/this->superSampling, sizeZ);
                auto barLV = new G4LogicalVolume(barS, localMaterial, name);
                new G4PVPlacement(
                    nullptr,   // no rotation
                    G4ThreeVector(displacementX, displacementY, 0.),  // displacement
                    barLV,     // its logical volume
                    name,      // its name
                    worldLV,   // its mother volume
                    false,     // no boolean operator
                    0,         // copy number
                    false      // checking overlaps
                );

                name = std::string("voxel_") + std::to_string(j)
                    + std::string("_") + std::to_string(i);
                auto voxelS = new G4Box(name, voxelSize, voxelSize/superSampling, voxelSize);
                auto voxelLV = new G4LogicalVolume(voxelS, localMaterial, name);
                new G4PVReplica(name, voxelLV, barLV,
                    kZAxis, this->dimension[2], 2*voxelSize);
                this->logicals.push_back(voxelLV);
            }
        } else {
            // non-scoring layers
            std::string name = std::string("slice") + std::to_string(j);
            auto sliceS = new G4Box(name, sizeX, voxelSize, sizeZ);
            auto sliceLV = new G4LogicalVolume(sliceS, localMaterial, name);
            float displacement = localOffset;
            new G4PVPlacement(
                nullptr,   // no rotation
                G4ThreeVector(0., displacement, 0.),  // displacement
                sliceLV,   // its logical volume
                name,      // its name
                worldLV,   // its mother volume
                false,     // no boolean operator
                0,         // copy number
                false      // checking overlaps
            );
        }
    }

    return worldPV;
}

void MCRef::DetectorConstruction::ConstructSDandField() {
    // Sensitive Detector Manager
    G4SDManager* pSDman = G4SDManager::GetSDMpointer();
    for (int i=0; i<this->logicals.size(); i++) {
        G4String SDname = std::string("det") + std::to_string(i);
        auto det = new G4MultiFunctionalDetector(SDname);

        G4VPrimitiveScorer* primitive;
        primitive = new G4PSEnergyDeposit("Edep");
        det->RegisterPrimitive(primitive);
        pSDman->AddNewDetector(det);
        SetSensitiveDetector(this->logicals[i], det);
    }
}