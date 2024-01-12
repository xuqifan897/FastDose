#ifndef __IMRTINIT_CUH__
#define __IMRTINIT_CUH__
#include <string>
#include <vector>
#include <iostream>
#include "fastdose.cuh"

namespace IMRT {
    class StructInfo {
    public:
        std::string name;
        float maxWeights;
        float maxDose;
        float minDoseTargetWeights;
        float minDoseTarget;
        float OARWeights;
        float IdealDose;

        // mask information
        uint3 size;
        uint3 crop_size;
        uint3 crop_start;
        std::vector<uint8_t> mask;

        friend std::ostream& operator<<(std::ostream& os, const StructInfo& obj);
    };
    std::ostream& operator<<(std::ostream& os, const StructInfo& obj);

    bool StructsInit(std::vector<StructInfo>& structs, bool verbose=true);
    bool readMaskFromHDF5(std::vector<StructInfo>& structs, const std::string& h5file);

    bool densityInit(fastdose::DENSITY_h& density_h,
        fastdose::DENSITY_d& density_d,
        const std::vector<StructInfo>& structs);
    bool getBBox(const std::vector<uint8_t>& bbox, const uint3 shape,
        uint3& bbox_start, uint3& bbox_size);

    bool specInit(fastdose::SPECTRUM_h& spectrum_h);
    bool kernelInit(fastdose::KERNEL_h& kernel_h);
}

#endif