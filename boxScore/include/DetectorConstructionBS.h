#ifndef DetectorConstructionBS_h
#define DetectorConstructionBS_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;

namespace bs
{
    class DetectorConstruction : public G4VUserDetectorConstruction
    {
    public:
        DetectorConstruction(float offset, float thickness)
            : offset(offset), thickness(thickness) {}
        ~DetectorConstruction() = default;

        G4VPhysicalVolume* Construct() override;
        void virtual ConstructSDandField();
        
        // This function divides the nominal layers into 
        // blocks so that multifunctional detector can be 
        // used to record the score.
        void SenseAssign(G4LogicalVolume* worldLogical);

        // This function merges layers of the same material into one layer
        void LayerMerge(std::vector<std::tuple<std::string, int>>& result, 
            std::vector<std::string>& input);

        float& getOffset() {return this->offset;}
        float& getThickness() {return this->thickness;}
    
    private:
        // To speedup experiment, we only choose a portion of 
        // the volume to use as the scoring volume. The parameter 
        // offset is the distance between the region and the boundary
        // of the world volume. In half size [length]
        float offset;

        // The thickness of the scoring volume. In half size [length]
        float thickness;

        std::map<std::string, G4Material*> LUTmat;

        // logical volumes to register sensitive detectors
        std::vector<G4LogicalVolume*> logicals;
    };
}

#endif