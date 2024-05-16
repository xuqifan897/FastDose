#include "ParamsCubeScore.h"
#include "ArgsCubeScore.h"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Box.hh"
#include <iostream>

cube_score::ParamsCubeScore::ParamsCubeScore(
    const std::map<std::string, G4Material*>& material_map):
    G4VNestedParameterisation(),
    MaterialMap(material_map) {
    // calculate the z dimension
    int dimension_z = 0;
    for (const auto& entry: SlabPhantom)
        dimension_z += entry.second;
    this->VoxelSize = getarg<float>("VoxelSize") * cm;
    this->BaseOffset = - (dimension_z - 1) * this->VoxelSize;
}


G4Material* cube_score::ParamsCubeScore::ComputeMaterial(
    G4VPhysicalVolume* /*currentVol*/, const G4int copyNo,
    const G4VTouchable* parentTouch) {
    if (parentTouch == nullptr) return this->MaterialMap["Air"];
    
    G4int cumuNo = 0;
    for (const auto& entry: SlabPhantom) {
        G4int cumuNo_next = cumuNo + entry.second;
        if (cumuNo_next > copyNo)
            return this->MaterialMap[entry.first];
        cumuNo = cumuNo_next;
    }
    std::cerr << "Error: copyNo not within the list" << std::endl;
    return this->MaterialMap["Air"];
}

G4int cube_score::ParamsCubeScore::GetNumberOfMaterials() const {
    return this->MaterialMap.size();
}

G4Material* cube_score::ParamsCubeScore::GetMaterial(G4int i) const {
    for (const auto& entry : this->MaterialMap) {
        if (i != 0) {
            i--;
            continue;
        }
        return entry.second;
    }
    std::cerr << "The index " << i << " is out of range" << std::endl;
    return nullptr;
}

void cube_score::ParamsCubeScore::ComputeTransformation(
    const G4int no, G4VPhysicalVolume* physVol) const {
    G4ThreeVector position(0.0f, 0.0f, this->BaseOffset + no * 2 * this->VoxelSize);
    physVol->SetTranslation(position);
}

void cube_score::ParamsCubeScore::ComputeDimensions(
    G4Box& box, const G4int, const G4VPhysicalVolume*) const {
    box.SetXHalfLength(this->VoxelSize);
    box.SetYHalfLength(this->VoxelSize);
    box.SetZHalfLength(this->VoxelSize);
}