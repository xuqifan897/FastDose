#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include "cuda_runtime.h"
#include "helper_math.cuh"
#include "fastdose.cuh"
namespace po = boost::program_options;
namespace fd = fastdose;
po::variables_map vm;

template<class T>
const T& getarg(const std::string& key) {
    try {
        return vm.at(key).as<T>();
    } catch (const std::out_of_range&) {
        std::cerr << "The key " << key << " doesn't exist in the argument list!" << std::endl;
        exit(1);
    }
}

bool argparse(int argc, char** argv) {
    po::options_description desc("This program calculates the single-beam dose "
        "in beam-eye-view coordinate frame");
    desc.add_options()
        ("help", "Produce help messages")
    ("phantomDim", po::value<std::vector<int>>()->multitoken()->required(),
        "The phantom dimension")
    ("voxelSize", po::value<std::vector<float>>()->multitoken()->required(),
        "The isotropic resolution [cm]")
    ("SAD", po::value<float>()->required(),
        "Source-to-axis distance [cm]. The isocenter by default "
        "is the center of mass of the PTV volume")
    ("density", po::value<std::string>()->required(),
        "The path to the density raw file")
    ("deviceIdx", po::value<int>()->default_value(2),
        "The device index")
    ("spectrum", po::value<std::string>()->required(),
        "The path to the spectrum")
    ("kernel", po::value<std::string>()->required(),
        "The path to the exponential CCCS kernel")
    ("nPhi", po::value<int>()->default_value(8),
        "The number of phi angles in convolution")
    ("subFluenceRes", po::value<float>()->default_value(0.25),
        "The resolution of the beam subfluence map [cm]")
    ("subFluenceDim", po::value<int>()->default_value(16),
        "The dimension of subdivided fluence for dose calculation accuracy")
    ("subFluenceOn", po::value<int>()->default_value(4),
        "The number of fluence pixels that are on in the subdivided fluence map, "
        "which corresponds to the beamlet size")
    ("longSpacing", po::value<float>()->default_value(0.25),
        "Longitudinal voxel size in the dose calculation [cm]")
    ("outputFile", po::value<std::string>()->required(),
        "File to store the binary dose array");

    
    // to see if "--help" is in the argument
    if (argc == 1) {
        std::cout << desc << std::endl;
        return 1;
    } else {
        std::string firstArg(argv[1]);
        if (firstArg == std::string("--help")) {
            std::cout << desc << std::endl;
            return 1;
        }
    }


    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    int width = 60;
    std::cout << "Parameters:" << std::endl;
    for (const auto& pair: vm) {
        std::stringstream second;
        const auto& value  = pair.second.value();
        if (auto ptr = boost::any_cast<int>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<float>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<std::vector<float>>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<std::vector<int>>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<std::string>(&value))
            second << *ptr;
        else if (auto ptr = boost::any_cast<std::vector<std::string>>(&value))
            second << *ptr;
        else
            second << "(unknown type)";
        
        std::string second_string = second.str();
        int remaining = width - pair.first.size() - second_string.size();
        remaining = std::max(5, remaining);

        std::stringstream output;
        output << pair.first << std::string(remaining, '.') << second_string;
        std::cout << output.str() << std::endl;
    }
    std::cout << std::endl;
    return 0;
}

bool densityInit(fd::DENSITY_h& density_h, fd::DENSITY_d& density_d) {
    const std::vector<float>& voxelSize = getarg<std::vector<float>>("voxelSize");
    const std::vector<int>& phantomDim = getarg<std::vector<int>>("phantomDim");

    density_h.VoxelSize = float3{voxelSize[0], voxelSize[1], voxelSize[2]};
    density_h.VolumeDim = uint3{(uint)phantomDim[0], (uint)phantomDim[1], (uint)phantomDim[2]};
    density_h.BBoxStart = uint3{0, 0, 0};
    density_h.BBoxDim = density_h.VolumeDim;
    std::cout << "BBoxStart: " << density_h.BBoxStart << ", BBoxDim: "
        << density_h.BBoxDim << std::endl << std::endl;
    
    size_t volumeSize = phantomDim[0] * phantomDim[1] * phantomDim[2];
    density_h.density.resize(volumeSize);

    const std::string& densityFile = getarg<std::string>("density");
    std::fstream f(densityFile);
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << densityFile << std::endl;
        return 1;
    }
    f.read((char*)density_h.density.data(), volumeSize*sizeof(float));
    f.close();

    fd::density_h2d(density_h, density_d);
    #if false
        fd::test_density();
    #endif
    return 0;
}

bool beamInit(fd::BEAM_h& beam_h) {
    const std::vector<float>& voxelSize = getarg<std::vector<float>>("voxelSize");
    const std::vector<int>& phantomDim = getarg<std::vector<int>>("phantomDim");
    float3 isocenter{voxelSize[0] * phantomDim[0],
        voxelSize[1] * phantomDim[1],
        voxelSize[2] * phantomDim[2]
    };
    isocenter *= 0.5f;

    float subFluenceRes = getarg<float>("subFluenceRes");
    int subFluenceDim = getarg<int>("subFluenceDim");
    int subFluenceOn = getarg<int>("subFluenceOn");
    float SAD = getarg<float>("SAD");
    float longSpacing = getarg<float>("longSpacing");

    beam_h.isocenter = isocenter;
    beam_h.beamlet_size = float2{subFluenceRes, subFluenceRes};
    beam_h.fmap_size = uint2{(uint)subFluenceDim, (uint)subFluenceDim};
    beam_h.sad = SAD;
    beam_h.angles = float3{0.f, 0.f, 0.f};
    beam_h.long_spacing = longSpacing;

    beam_h.fluence.resize(subFluenceDim * subFluenceDim);
    std::fill(beam_h.fluence.begin(), beam_h.fluence.end(), 0.0f);
    int margin = subFluenceDim - subFluenceOn;
    if (margin % 2 != 0) {
        std::cerr << "We expect the margin, i.e., subFluenceDim - subFluenceOn, to be even, "
            "but subFluenceDim=" << subFluenceDim << ", subFluenceOn=" << subFluenceOn << std::endl;
        return 1;
    }
    margin = margin / 2;
    int idx_end = subFluenceDim - margin;
    for (int i=margin; i<idx_end; i++) {
        for (int j=margin; j<idx_end; j++) {
            int idx_global = i * subFluenceDim + j;
            beam_h.fluence[idx_global] = 1.0f;
        }
    }

    beam_h.lim_min = SAD - beam_h.isocenter.y;
    beam_h.lim_max = SAD + beam_h.isocenter.y;
    beam_h.long_dim = 2 * beam_h.isocenter.y / longSpacing;
    beam_h.source = float3{isocenter.x, isocenter.y - SAD, isocenter.z};
    #if false
        std::cout << beam_h << std::endl;
    #endif
    return 0;
}

bool specInit(fd::SPECTRUM_h& spectrum_h) {
    const std::string& spectrum_file = getarg<std::string>("spectrum");
    if (spectrum_h.read_spectrum_file(spectrum_file)) {
        return 1;
    }
    if (spectrum_h.bind_spectrum())
        return 1;
    #if true
        fd::test_spectrum(spectrum_h);
    #endif
    return 0;
}

bool kernelInit(fd::KERNEL_h& kernel_h) {
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

bool doseCalulation(
    fd::DENSITY_d& density_d,
    fd::BEAM_h& beam_h,
    fd::SPECTRUM_h& spectrum_h,
    fd::KERNEL_h& kernel_h,
    std::vector<float>& doseMat
) {
    float* d_denseDoseMat = nullptr;
    uint3 densityDim = density_d.VolumeDim;
    size_t denseDoseMatSize = densityDim.x * densityDim.y * densityDim.z;
    checkCudaErrors(cudaMalloc((void**)&d_denseDoseMat, denseDoseMatSize*sizeof(float)));

    float* d_FluenceBuffer = nullptr;
    float* d_DensityBEVBuffer = nullptr;
    float* d_TermaBEVBuffer = nullptr;
    float* d_DoseBEVBuffer = nullptr;
    size_t fluenceBufferSize = beam_h.fmap_size.x * beam_h.fmap_size.y;
    size_t BEVBufferSize = fluenceBufferSize * beam_h.long_dim;
    checkCudaErrors(cudaMalloc((void**)&d_FluenceBuffer, fluenceBufferSize*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_DensityBEVBuffer, BEVBufferSize*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_TermaBEVBuffer, BEVBufferSize*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_DoseBEVBuffer, BEVBufferSize*sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_FluenceBuffer, beam_h.fluence.data(),
        beam_h.fluence.size()*sizeof(float), cudaMemcpyHostToDevice));
    
    float** d_FluenceArray = nullptr;
    float** d_DensityArray = nullptr;
    float** d_TermaArray = nullptr;
    float** d_DoseArray = nullptr;
    fd::d_BEAM_d* d_BeamArray = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_FluenceArray, sizeof(float*)));
    checkCudaErrors(cudaMalloc((void***)&d_DensityArray, sizeof(float*)));
    checkCudaErrors(cudaMalloc((void***)&d_TermaArray, sizeof(float*)));
    checkCudaErrors(cudaMalloc((void***)&d_DoseArray, sizeof(float*)));
    checkCudaErrors(cudaMalloc((void**)&d_BeamArray, sizeof(fd::d_BEAM_d)));

    checkCudaErrors(cudaMemcpy(d_FluenceArray, &d_FluenceBuffer,
        sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_DensityArray, &d_DensityBEVBuffer,
        sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_TermaArray, &d_TermaBEVBuffer,
        sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_DoseArray, &d_DoseBEVBuffer,
        sizeof(float*), cudaMemcpyHostToDevice));
    size_t pitch = beam_h.fmap_size.x * beam_h.fmap_size.y * sizeof(float);
    fd::d_BEAM_d h_beam_h(beam_h, pitch, pitch);
    checkCudaErrors(cudaMemcpy(d_BeamArray, &h_beam_h,
        sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice));
    

    fd::TermaComputeCollective(
        beam_h.fluence.size(),
        1,
        d_BeamArray,
        d_FluenceArray,
        d_TermaArray,
        d_DensityArray,
        density_d,
        spectrum_h,
        0
    );


    fd::DoseComputeCollective(
        beam_h.fluence.size(),
        1,
        d_BeamArray,
        d_TermaArray,
        d_DensityArray,
        d_DoseArray,
        kernel_h.nTheta,
        kernel_h.nPhi,
        0
    );

    doseMat.resize(BEVBufferSize);
    checkCudaErrors(cudaMemcpy(doseMat.data(), d_DoseBEVBuffer,
        BEVBufferSize*sizeof(float), cudaMemcpyDeviceToHost));

    // clean up
    checkCudaErrors(cudaFree(d_DoseArray));
    checkCudaErrors(cudaFree(d_TermaArray));
    checkCudaErrors(cudaFree(d_DensityArray));
    checkCudaErrors(cudaFree(d_FluenceArray));

    checkCudaErrors(cudaFree(d_DoseBEVBuffer));
    checkCudaErrors(cudaFree(d_TermaBEVBuffer));
    checkCudaErrors(cudaFree(d_DensityBEVBuffer));
    checkCudaErrors(cudaFree(d_FluenceBuffer));
    checkCudaErrors(cudaFree(d_denseDoseMat));
    return 0;
}

int main(int argc, char** argv) {
    if (argparse(argc, argv))
        return 0;
    
    int deviceIdx = getarg<int>("deviceIdx");
    cudaSetDevice(deviceIdx);

    fd::DENSITY_h density_h;
    fd::DENSITY_d density_d;

    if (densityInit(density_h, density_d)) {
        std::cerr << "Density initialization error." << std::endl;
        return 1;
    }
    
    fd::BEAM_h beam_h;
    if (beamInit(beam_h)) {
        std::cerr << "Beam initialization error." << std::endl;
        return 1;
    }

    fd::SPECTRUM_h spectrum_h;
    if (specInit(spectrum_h)) {
        std::cerr << "Spectrum initialization error." << std::endl;
        return 1;
    }

    fd::KERNEL_h kernel_h;
    if (kernelInit(kernel_h)) {
        std::cerr << "Kernel initialization error." << std::endl;
        return 1;
    }

    std::vector<float> doseMat;
    if (doseCalulation(density_d, beam_h, spectrum_h, kernel_h, doseMat)) {
        std::cerr << "Dose calculation error." << std::endl;
        return 1;
    }

    #if false
        float maxValue = 0.0f;
        for (int i=0; i<doseMat.size(); i++)
            maxValue = max(maxValue, doseMat[i]);
        std::cout << "dose max value: " << maxValue << std::endl;
    #endif

    std::ofstream f(getarg<std::string>("outputFile"));
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << getarg<std::string>("outputFile") << std::endl;
        return 1;
    }
    f.write((char*)doseMat.data(), doseMat.size()*sizeof(float));
    f.close();
    std::cout << "Data written to file: " << getarg<std::string>("outputFile") << std::endl;
    return 0;
}