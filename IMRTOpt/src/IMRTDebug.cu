#include <boost/filesystem.hpp>
#include "IMRTDebug.cuh"
#include "IMRTArgs.h"

namespace fs = boost::filesystem;
namespace fd = fastdose;

bool IMRT::doseCalcDebug(
    std::vector<BeamBundle>& beam_bundles,
    fastdose::DENSITY_d& density_d,
    fastdose::SPECTRUM_h& spectrum_h,
    fastdose::KERNEL_h& kernel_h,
    cudaStream_t stream
) {
    int beamIdx = getarg<int>("beamIdxDebug");
    BeamBundle& beam_bundle = beam_bundles[beamIdx];
    int nBeamlets = beam_bundle.beams_h.size();
    std::vector<fd::BEAM_d> beamlets(nBeamlets);
    for (int i=0; i<nBeamlets; i++)
        fd::beam_h2d(beam_bundle.beams_h[i], beamlets[i]);

    #if false
        std::vector<fd::BEAM_h>& beamlets_h = beam_bundle.beams_h;
        for (int i=0; i<beamlets_h.size(); i++) {
            std::cout << "Beamlet " << i << std::endl;
            std::cout << beamlets_h[i] << std::endl << std::endl;
        }
        return 0;
    #endif

    // preparation
    std::vector<fd::d_BEAM_d> h_beams;
    h_beams.reserve(nBeamlets);
    for (int i=0; i<nBeamlets; i++)
        h_beams.emplace_back(fd::d_BEAM_d(beamlets[i]));
    fd::d_BEAM_d* d_beams = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beams, nBeamlets*sizeof(fd::d_BEAM_d)));
    checkCudaErrors(cudaMemcpy(d_beams, h_beams.data(),
        nBeamlets*sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice));

    std::vector<float*> h_fluence_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_fluence_array[i] = beamlets[i].fluence;
    float** d_fluence_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_fluence_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_fluence_array, h_fluence_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    std::vector<float*> h_TermaBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_TermaBEV_array[i] = beamlets[i].TermaBEV;
    float** d_TermaBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_TermaBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_TermaBEV_array, h_TermaBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    std::vector<float*> h_DensityBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_DensityBEV_array[i] = beamlets[i].DensityBEV;
    float** d_DensityBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_DensityBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_DensityBEV_array, h_DensityBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    std::vector<float*> h_DoseBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_DoseBEV_array[i] = beamlets[i].DoseBEV;
    float** d_DoseBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_DoseBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_DoseBEV_array, h_DoseBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    size_t fmap_npixels = beamlets[0].fmap_size.x * beamlets[0].fmap_size.y;

    // calculate Terma collectively
    fd::TermaComputeCollective(
        fmap_npixels,
        nBeamlets,
        d_beams,
        d_fluence_array,
        d_TermaBEV_array,
        d_DensityBEV_array,
        density_d,
        spectrum_h,
        stream
    );
    cudaDeviceSynchronize();
    std::cout << "Collective Terma calculation finished." << std::endl << std::endl;

    // log out data
    fs::path resultFolder(getarg<std::string>("outputFolder"));
    resultFolder /= std::string("doseCompDebug");
    if (! fs::is_directory(resultFolder))
        fs::create_directory(resultFolder);

    #if false
        // log out Terma data
        for (int i=0; i<nBeamlets; i++) {
            const fd::BEAM_d& beamlet = beamlets[i];
            size_t DoseBEVSize = beamlet.DensityBEV_pitch / sizeof(float) * beamlet.long_dim;
            std::vector<float> h_TermaBEV(DoseBEVSize, 0.0f);
            checkCudaErrors(cudaMemcpy(h_TermaBEV.data(), beamlet.TermaBEV,
                DoseBEVSize*sizeof(float), cudaMemcpyDeviceToHost));
            
            fs::path file = resultFolder / (std::string("BEVTerma")
                + std::to_string(i) + ".bin");
            std::ofstream f(file.string());
            if (! f.is_open()) {
                std::cerr << "Could not open file: " << file.string();
                return 1;
            }
            f.write((char*)(h_TermaBEV.data()), DoseBEVSize*sizeof(float));
            f.close();
        }

        // log out Density data
        for (int i=0; i<nBeamlets; i++) {
            const fd::BEAM_d& beamlet = beamlets[i];
            size_t DoseBEVSize = beamlet.DensityBEV_pitch / sizeof(float) * beamlet.long_dim;
            std::vector<float> h_DensityBEV(DoseBEVSize, 0.0f);
            checkCudaErrors(cudaMemcpy(h_DensityBEV.data(), beamlet.DensityBEV,
                DoseBEVSize*sizeof(float), cudaMemcpyDeviceToHost));
            
            fs::path file = resultFolder / (std::string("BEVDensity")
                + std::to_string(i) + ".bin");
            std::ofstream f(file.string());
            if (! f.is_open()) {
                std::cerr << "Could not open file: " << file.string();
                return 1;
            }
            f.write((char*)(h_DensityBEV.data()), DoseBEVSize*sizeof(float));
            f.close();
        }
        return 0;
    #endif

    // print the longitudinal dimensions of beamlets
    for (int i=0; i<nBeamlets; i++)
        std::cout << "Beamlet " << i << ", long_dim: "
        << beamlets[i].long_dim << std::endl;
    std::cout << std::endl;

    #if true
        // calculate Dose collectively
        fd::DoseComputeCollective(
            fmap_npixels,
            nBeamlets,
            d_beams,
            d_TermaBEV_array,
            d_DensityBEV_array,
            d_DoseBEV_array,
            kernel_h.nTheta,
            kernel_h.nPhi,
            stream
        );
        cudaDeviceSynchronize();
        std::cout << "Collective dose calculation finished." << std::endl;
    #else
        for (int i=0; i<nBeamlets; i++) {
            fd::DoseComputeCollective(
                fmap_npixels,
                1,
                d_beams + i,
                d_TermaBEV_array + i,
                d_DensityBEV_array + i,
                d_DoseBEV_array + i,
                kernel_h.nTheta,
                kernel_h.nPhi,
                stream
            );
            cudaDeviceSynchronize();
            std::cout << "Dose calculation. Beamlet: " << i
                << " / " << nBeamlets << std::endl;
        }
        
        for (int i=0; i<nBeamlets; i++) {
            const fd::BEAM_d& beamlet = beamlets[i];
            size_t DoseBEVSize = beamlet.DensityBEV_pitch / sizeof(float) * beamlet.long_dim;
            std::vector<float> h_DoseBEV(DoseBEVSize, 0.0f);
            checkCudaErrors(cudaMemcpy(h_DoseBEV.data(), beamlet.DoseBEV,
                DoseBEVSize*sizeof(float), cudaMemcpyDeviceToHost));
            
            fs::path file = resultFolder / (std::string("BEVDose")
                + std::to_string(i) + ".bin");
            std::ofstream f(file.string());
            if (! f.is_open()) {
                std::cerr << "Could not open file: " << file.string();
                return 1;
            }
            f.write((char*)(h_DoseBEV.data()), DoseBEVSize*sizeof(float));
            f.close();
        }
    #endif

    // clean-up
    checkCudaErrors(cudaFree(d_DoseBEV_array));
    checkCudaErrors(cudaFree(d_DensityBEV_array));
    checkCudaErrors(cudaFree(d_TermaBEV_array));
    checkCudaErrors(cudaFree(d_fluence_array));
    checkCudaErrors(cudaFree(d_beams));

    return 0;
}