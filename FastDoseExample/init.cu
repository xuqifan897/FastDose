#include "init.h"
#include "argparse.h"
#include "fastdose.cuh"

#include <vector>
#include <string>
#include <fstream>
#include <boost/filesystem.hpp>

namespace fd = fastdose;
namespace fs = boost::filesystem;

bool example::densityInit(fd::DENSITY_h& density_h, fd::DENSITY_d& density_d) {
    std::vector<float> VoxelSize = getarg<std::vector<float>>("voxelSize");
    std::vector<int> VolumeDim = getarg<std::vector<int>>("dicomVolumeDimension");
    std::vector<int> BBoxStart = getarg<std::vector<int>>("doseBoundingBoxStartIndices");
    std::vector<int> BBoxDim = getarg<std::vector<int>>("doseBoundingBoxDimensions");

    density_h.VoxelSize = float3{VoxelSize[0], VoxelSize[1], VoxelSize[2]};
    density_h.VolumeDim = uint3{(uint)VolumeDim[0], (uint)VolumeDim[1], (uint)VolumeDim[2]};
    density_h.BBoxStart = uint3{(uint)BBoxStart[0], (uint)BBoxStart[1], (uint)BBoxStart[2]};
    density_h.BBoxDim = uint3{(uint)BBoxDim[0], (uint)BBoxDim[1], (uint)BBoxDim[2]};
    
    size_t volumeSize = VolumeDim[0] * VolumeDim[1] * VolumeDim[2];
    density_h.density.resize(volumeSize);

    fs::path densityFile(getarg<std::string>("inputFolder"));
    densityFile = densityFile / std::string("density.raw");
    if (! fs::exists(densityFile)) {
        std::cerr << "The file " << densityFile.string() << " doesn't exist!" << std::endl;
        return 1;
    }

    std::ifstream f(densityFile.string());
    if (!f) {
        std::cerr << "Could not open file " << densityFile.string() << std::endl;
        return 1;
    }
    f.read((char*)(density_h.density.data()), volumeSize*sizeof(float));
    f.close();

    fd::density_h2d(density_h, density_d);
    return 0;
}

void example::densityTest(fd::DENSITY_h& density_h, fd::DENSITY_d& density_d) {
    // This function tests if density_h and density_d match each other
    const auto& volume = density_d.VolumeDim;
    size_t volumeSize = volume.x * volume.y * volume.z;
    std::vector<float> probe(volumeSize);
    float* probe_d = nullptr;
    checkCudaErrors(cudaMalloc((void**)&probe_d, volumeSize*sizeof(float)));

    fd::h_readTexture3D(probe_d, density_d.densityTex, volume.x, volume.y, volume.z);

    // Compare the original and the new results
    double deviation = 0.;
    double sum_h = 0.;
    double sum_d = 0.;
    for (int i=0; i<volumeSize; i++) {
        deviation += std::abs(density_h.density[i] - probe[i]);
        sum_h += density_h.density[i];
        sum_d += probe[i];
    }

    std::cout << "absolute deviation: " << deviation << std::endl;
    std::cout << "host sum: " << sum_h << std::endl;
    std::cout << "device sum: " << sum_d << std::endl;
}

bool example::beamsInit(
    std::vector<fd::BEAM_h>& beams_h,
    std::vector<fd::BEAM_d>& beams_d,
    fd::DENSITY_h& density_h
) {
    beams_h.clear();
    beams_d.clear();

    fs::path beamFile(getarg<std::string>("inputFolder"));
    beamFile = beamFile / std::string("beam_lists.txt");
    if (! fs::exists(beamFile)) {
        std::cerr << "The file " << beamFile.string() << " doesn't exist!" << std::endl;
        return 1;
    }

    std::ifstream f(beamFile.string());
    if (! f) {
        std::cerr << "Could not open the file: " << beamFile.string() << std::endl;
        return 1;
    }

    std::string tableRow;
    bool skipFirst = true;
    while (std::getline(f, tableRow)) {
        if (skipFirst) {
            skipFirst = false;
            continue;
        }
        beams_h.emplace_back(fd::BEAM_h());
        auto& last_beam = beams_h.back();
        std::istringstream iss(tableRow);
        iss >> last_beam.angles.x >> last_beam.angles.y >> last_beam.angles.z >> 
        last_beam.sad >> last_beam.isocenter.x >> last_beam.isocenter.y >> 
        last_beam.isocenter.z >> last_beam.beamlet_size.x >> last_beam.beamlet_size.y >>
        last_beam.fmap_size.x >> last_beam.fmap_size.y >> last_beam.long_spacing;
        int n_beamlets = last_beam.fmap_size.x * last_beam.fmap_size.y;
        last_beam.fluence = std::vector<float>(n_beamlets, 1.);

        last_beam.calc_range(density_h);
    }

    beams_d.resize(beams_h.size());
    for (int i=0; i<beams_d.size(); i++) {
        fd::beam_h2d(beams_h[i], beams_d[i]);
        #if false
            // for debug purposes
            test_TermaBEVPitch(last_beam_d);
            break;
        #endif
    }

    return 0;
}

bool example::specInit(fastdose::SPECTRUM_h& spectrum_h) {
    fs::path spectrum_file(getarg<std::string>("inputFolder"));
    spectrum_file = spectrum_file / std::string("spec_6mv.spec");
    if(spectrum_h.read_spectrum_file(spectrum_file.string()))
        return 1;
    if(spectrum_h.bind_spectrum())
        return 1;
#if false
    fd::test_spectrum(spectrum_h);
#endif
    return 0;
}