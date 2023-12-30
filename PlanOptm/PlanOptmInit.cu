#include <vector>
#include <iostream>
#include <fstream>

#include "PlanOptmArgs.cuh"
#include "PlanOptmInit.cuh"
#include "fastdose.cuh"
namespace fd = fastdose;

bool PlanOptm::densityInit(fd::DENSITY_h& density_h, fd::DENSITY_d& density_d) {
    const std::vector<float>& voxelSize = getarg<std::vector<float>>("voxelSize");
    const std::vector<int>& phantomDim = getarg<std::vector<int>>("phantomDim");
    const std::vector<int>& BBoxStart = getarg<std::vector<int>>("boundingBoxStart");
    const std::vector<int>& BBoxDim = getarg<std::vector<int>>("boundingBoxDimensions");

    density_h.VoxelSize = float3{voxelSize[0], voxelSize[1], voxelSize[2]};
    density_h.VolumeDim = uint3{(uint)phantomDim[0], (uint)phantomDim[1], (uint)phantomDim[2]};
    density_h.BBoxStart = uint3{(uint)BBoxStart[0], (uint)BBoxStart[1], (uint)BBoxStart[2]};
    density_h.BBoxDim = uint3{(uint)BBoxDim[0], (uint)BBoxDim[1], (uint)BBoxDim[2]};

    size_t volumeSize = phantomDim[0] * phantomDim[1] * phantomDim[2];
    density_h.density.resize(volumeSize);

    const std::string& densityFile = getarg<std::string>("phantom");
    std::ifstream f(densityFile);
    if (! f.is_open()) {
        std::cerr << "Could not open file: " << densityFile << std::endl;
        return 1;
    }
    f.read((char*)(density_h.density.data()), volumeSize*sizeof(float));
    f.close();

    fd::density_h2d(density_h, density_d);
    return 0;
}


bool PlanOptm::specInit(fastdose::SPECTRUM_h& spectrum_h) {
    const std::string& spectrum_file = getarg<std::string>("spectrum");
    if (spectrum_h.read_spectrum_file(spectrum_file))
        return 1;
    if (spectrum_h.bind_spectrum())
        return 1;
    #if true
        fd::test_spectrum(spectrum_h);
    #endif
    return 0;
}


bool PlanOptm::kernelInit(fastdose::KERNEL_h& kernel_h){
    const std::string& kernel_file = getarg<std::string>("kernel");
    int nPhi = getarg<int>("nPhi");
    if (kernel_h.read_kernel_file(kernel_file, nPhi))
        return 1;
    if (kernel_h.bind_kernel())
        return 1;
    #if true
        fd::test_kernel(kernel_h);
    #endif
    return 0;
}