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

#include "DetectorConstructionBS.h"
#include "PhantomDefBS.h"
#include "argparseBS.h"

G4VPhysicalVolume* bs::DetectorConstruction::Construct()
{
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* air = nist->FindOrBuildMaterial("G4_AIR");

    G4Element* elH = new G4Element(
        std::string("Hydrogen"), std::string("H"), 1., 1.01*g/mole);
    G4Element* elO = new G4Element(
        std::string("Oxygen"), std::string("O"), 8., 16.00*g/mole); 
    
    G4double density = 1.000*g/cm3;
    G4Material* water = new G4Material(std::string("water"), density, 2);
    water->AddElement(elH, 2);
    water->AddElement(elO, 1);

    density = 0.92*g/cm3;
    // density = 0.95610404*g/cm3;
    G4Material* adipose = new G4Material(std::string("adipose"), density, 2);
    adipose->AddElement(elH, 2);
    adipose->AddElement(elO, 1);

    density = 1.04*g/cm3;
    // density = 1.0652606*g/cm3;
    G4Material* muscle = new G4Material(std::string("muscle"), density, 2);
    muscle->AddElement(elH, 2);
    muscle->AddElement(elO, 1);

    density = 1.85*g/cm3;
    // density = 1.6093124*g/cm3;
    G4Material* bone = new G4Material(std::string("bone"), density, 2);
    bone->AddElement(elH, 2);
    bone->AddElement(elO, 1);

    density = 0.25*g/cm3;
    // density = 0.2481199*g/cm3;
    G4Material* lung = new G4Material(std::string("lung"), density, 2);
    lung->AddElement(elH, 2);
    lung->AddElement(elO, 1);

    this->LUTmat = {
        {"air", air},
        {"water", water},
        {"adipose", adipose},
        {"muscle", muscle},
        {"bone", bone},
        {"lung", lung}
    };

    // display material information
    std::cout << std::endl;
    std::cout << "Material information: " << std::endl;
    for (auto it=LUTmat.begin(); it!=LUTmat.end(); it++)
        G4cout << it->second << G4endl;
    
    if (GD == nullptr)
    {
        G4cout << "Material geometry is not initialized, "
            "please initialize it by calling \"si::GD = new "
            "si::GeomDef();\" before the detector construction" << G4endl;
        exit(1);
    }

    // get the thickness of all slices
    float SizeZ = 0.;
    for (int i=0; i<GD->layers.size(); i++)
        SizeZ += std::get<1>(GD->layers[i]);
    
    int dimXY = (*vm)["dimXY"].as<int>();
    float voxelSize = (*vm)["voxelSize"].as<float>() * cm;
    float sizeXY = dimXY * voxelSize;

    auto worldS = new G4Box("World", sizeXY, sizeXY, SizeZ);
    auto worldLV = new G4LogicalVolume(
        worldS,
        air,
        "World");
    
    auto worldPV = new G4PVPlacement(
        nullptr,  // no rotation
        G4ThreeVector(),  // at (0, 0, 0)
        worldLV,  // its logical volume
        "World",  // its name
        nullptr,  // no mother volume
        false,  // no boolean operator
        0,  // copy number
        false);  // checking overlaps
    
    SenseAssign(worldLV);

    return worldPV;
}

void bs::DetectorConstruction::SenseAssign(G4LogicalVolume* worldLogical)
{
    float SizeZ = 0.;
    for (int i=0; i<GD->layers.size(); i++)
        SizeZ += std::get<1>(GD->layers[i]);
    
    float voxelSize = (*bs::vm)["voxelSize"].as<float>() * cm;
    int DimZ = static_cast<int>(std::round(SizeZ / voxelSize));
    int DimOffset = static_cast<int>(std::round(this->offset / voxelSize));
    int DimThickness = static_cast<int>(std::round(this->thickness / voxelSize));

    // firstly, divide the layers into sub layers, 
    // each represented by its material name
    std::vector<std::string> subLayers(DimZ);
    int currentIdx = 0;
    for (int i=0; i<GD->layers.size(); i++)
    {
        float layerThickness = std::get<1>(GD->layers[i]);
        std::string& material = std::get<0>(GD->layers[i]);
        int layerDimZ = static_cast<int>(layerThickness / voxelSize);
        for (int j=0; j<layerDimZ; j++)
            subLayers[currentIdx + j] = material;
        currentIdx += layerDimZ;
    }

    // then, we take out the part to be scored.
    std::vector<std::string> PreLayers(DimOffset);
    std::copy(subLayers.begin(), subLayers.begin()+DimOffset, PreLayers.begin());

    std::vector<std::string> ScoringLayers(DimThickness);
    std::copy(subLayers.begin() + DimOffset, subLayers.begin() + 
        DimOffset + DimThickness, ScoringLayers.begin());

    std::vector<std::string> PostLayers(DimZ - DimOffset - DimThickness);
    std::copy(subLayers.begin() + DimOffset + DimThickness, 
        subLayers.end(), PostLayers.begin());

    // then, we merge adjacent layers of the same material
    std::vector<std::tuple<std::string, int>> PreLayersMerge;
    LayerMerge(PreLayersMerge, PreLayers);

    std::vector<std::tuple<std::string, int>> PostLayersMerge;
    LayerMerge(PostLayersMerge, PostLayers);
    
    int Idx = 0;
    int DimXY = (*bs::vm)["dimXY"].as<int>();
    float SizeXY = DimXY * voxelSize;
    float currentOffset = -SizeZ;
    G4cout << "Pre-scoring layers:" << G4endl;
    for (auto& it : PreLayersMerge)
    {
        std::string& materialName = std::get<0>(it);
        int thickDim = std::get<1>(it);
        float thickPhysical = thickDim * voxelSize;
        float localOffset = currentOffset + thickPhysical;

        auto MatPhysical = this->LUTmat[materialName];
        std::string layerName = "layer_" + std::to_string(++Idx);
        auto layerS = new G4Box(layerName, SizeXY, SizeXY, thickPhysical);
        auto layerLV = new G4LogicalVolume(layerS, MatPhysical, layerName);
        new G4PVPlacement(
            nullptr,  //no rotation
            G4ThreeVector(0., 0., localOffset),  // displacement
            layerLV,  // its logical volume
            layerName,   // its name
            worldLogical,  // its mother volume
            false,  // no boolean operation
            0,  // copy number
            false);

        currentOffset += 2 * thickPhysical;

        // set visualization properties
        G4VisAttributes* layer_logVisAtt 
            = new G4VisAttributes(G4Colour(0., 0., 1., .3));
        layer_logVisAtt->SetForceWireframe(true);
        layerLV->SetVisAttributes(layer_logVisAtt);

        G4cout << "material: " << std::left << std::setw(10) << materialName << "thickness(cm): " 
            << std::left << std::setw(10) << thickPhysical / cm  << 
            "offset(cm)" << std::left << std::setw(10) << localOffset / cm << G4endl;
    }

    G4cout << G4endl << "Scoring layers:" << G4endl;
    this->logicals.reserve(DimThickness * DimXY);
    for (auto& it : ScoringLayers)
    {
        std::string& materialName = it;
        int thickDim = 1;
        float thickPhysical = thickDim * voxelSize;
        float localOffset = currentOffset + thickPhysical;
        int layerIdx = ++Idx;

        auto MatPhysical = this->LUTmat[materialName];
        for (int ii=0; ii<DimXY; ii++)
        {
            float OffsetY = -SizeXY + (2*ii+1) * voxelSize;
            std::string layerName = "layer_" + std::to_string(layerIdx) + "_" + std::to_string(ii+1);
            auto layerS = new G4Box(layerName, SizeXY, voxelSize, thickPhysical);
            auto layerLV = new G4LogicalVolume(layerS, MatPhysical, layerName);
            new G4PVPlacement(
                nullptr,  // no rotation
                G4ThreeVector(0., OffsetY, localOffset),  // displacement
                layerLV,  // its logical volume
                layerName,  // its name
                worldLogical,  // its mother volume
                false,  // no boolean operation
                0,  // copy number
                false);
            
            std::string elementName = "element_" + std::to_string(layerIdx) + "_" + std::to_string(ii+1);
            auto elementS = new G4Box(elementName, voxelSize, voxelSize, voxelSize);
            auto elementLV = new G4LogicalVolume(elementS, MatPhysical, elementName);
            new G4PVReplica(elementName, elementLV, layerLV, kXAxis, DimXY, 2*voxelSize);

            // set visualization properties
            G4VisAttributes* layer_logVisAtt 
                = new G4VisAttributes(G4Colour(1., 0., 0., .3));
            layer_logVisAtt->SetForceWireframe(true);
            layerLV->SetVisAttributes(layer_logVisAtt);
            elementLV->SetVisAttributes(layer_logVisAtt);

            this->logicals.push_back(elementLV);
        }

        currentOffset += 2 * thickPhysical;

        G4cout << "material: " << std::left << std::setw(10) << it << "thickness(cm): " 
            << std::left << std::setw(10) << thickPhysical / cm 
            << "offset(cm):" << localOffset / cm << G4endl;
    }

    G4cout << G4endl << "Post-scoring layers:" << G4endl;
    for (auto& it : PostLayersMerge)
    {
        std::string& materialName = std::get<0>(it);
        int thickDim = std::get<1>(it);
        float thickPhysical = thickDim * voxelSize;
        float localOffset = currentOffset + thickPhysical;

        auto MatPhysical = this->LUTmat[materialName];
        std::string layerName = "layer_" + std::to_string(++Idx);
        auto layerS = new G4Box(layerName, SizeXY, SizeXY, thickPhysical);
        auto layerLV = new G4LogicalVolume(layerS, MatPhysical, layerName);
        new G4PVPlacement(
            nullptr,  //no rotation
            G4ThreeVector(0., 0., localOffset),  // displacement
            layerLV,  // its logical volume
            layerName,   // its name
            worldLogical,  // its mother volume
            false,  // no boolean operation
            0,  // copy number
            false);

        currentOffset += 2 * thickPhysical;

        // set visualization properties
        G4VisAttributes* layer_logVisAtt 
            = new G4VisAttributes(G4Colour(0., 0., 1., .3));
        layer_logVisAtt->SetForceWireframe(true);
        layerLV->SetVisAttributes(layer_logVisAtt);

        G4cout << "material: " << std::left << std::setw(10) << materialName << "thickness(cm): " 
            << std::left << std::setw(10) << thickPhysical / cm  << 
            "offset(cm)" << std::left << std::setw(10) << localOffset / cm << G4endl;
    }
}

void bs::DetectorConstruction::LayerMerge(
    std::vector<std::tuple<std::string, int>>& result, 
    std::vector<std::string>& input)
{
    result.clear();
    if (input.size() == 0)
        return;
    int lastIdx = 0;
    int currentIdx = 0;
    while (true)
    {
        std::string initMat = input[lastIdx];
        bool flag = false;
        while (true)
        {
            if (currentIdx == input.size())
            {
                int thickness = currentIdx - lastIdx;
                if (thickness != 0)
                    result.push_back(std::make_tuple(initMat, thickness));
                flag = true;
                break;
            }

            if (input[currentIdx] == initMat)
                currentIdx += 1;
            else
            {
                int thickness = currentIdx - lastIdx;
                result.push_back(std::make_tuple(initMat, thickness));
                lastIdx = currentIdx;
                break;
            }
        }
        if (flag)
            break;
    }
}

void bs::DetectorConstruction::ConstructSDandField()
{
    auto SDMpointer = G4SDManager::GetSDMpointer();
    SDMpointer->SetVerboseLevel(0);

    for (int i=0; i<this->logicals.size(); i++)
    {
        std::string SDname = std::string("SD") + std::to_string(i+1);
        auto senseDet = new G4MultiFunctionalDetector(SDname);
        
        G4VPrimitiveScorer* primitive;
        primitive = new G4PSEnergyDeposit("Edep");
        senseDet->RegisterPrimitive(primitive);

        SDMpointer->AddNewDetector(senseDet);
        SetSensitiveDetector(this->logicals[i], senseDet);
    }
}