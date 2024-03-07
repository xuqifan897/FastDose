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


    class Params {
    public:
        Params(): beamWeight(0), gamma(0), eta(0), numBeamsWeWant(0), stepSize(0),
            maxIter(0), showTrigger(1), changeWeightsTrigger(1) {}
        float beamWeight;
        float gamma;
        float eta;
        int numBeamsWeWant;
        float stepSize;
        int maxIter;
        int showTrigger;
        int changeWeightsTrigger;
        int pruneTrigger;
    };

    bool ParamsInit(Params& params);
}

#define DECLARE_PUBLIC_MEMBERS \
    MEMBER(float, beamWeight) \
    MEMBER(float, gamma) \
    MEMBER(float, eta) \
    MEMBER(int, numBeamsWeWant) \
    MEMBER(float, stepSize) \
    MEMBER(int, maxIter) \
    MEMBER(int, showTrigger) \
    MEMBER(int, changeWeightsTrigger) \
    MEMBER(int, pruneTrigger)

#endif