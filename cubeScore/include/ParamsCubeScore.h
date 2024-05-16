#ifndef __PARAMSCUBESCORE_H__
#define __PARAMSCUBESCORE_H__

#include "G4VNestedParameterisation.hh"
#include "G4Material.hh"

namespace cube_score {
    class ParamsCubeScore: public G4VNestedParameterisation {
    public:
        ParamsCubeScore(const std::map<std::string, G4Material*>& material_map);
        virtual G4Material* ComputeMaterial(G4VPhysicalVolume* currentVol,
            const G4int repNo, const G4VTouchable* parentTouch=nullptr) override;

        virtual G4int GetNumberOfMaterials() const override;
        virtual G4Material* GetMaterial(G4int idx) const override;
        virtual void ComputeTransformation(const G4int no, G4VPhysicalVolume* currentPV)
            const override;
        virtual void ComputeDimensions(G4Box& , const G4int, const G4VPhysicalVolume* )
            const override;
    
    private:
        std::map<std::string, G4Material*> MaterialMap;

        float BaseOffset;
        float VoxelSize;
    };
}

#endif