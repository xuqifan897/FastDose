#include "PlanOptmTestCase.cuh"
#include "PlanOptmArgs.cuh"
#include "fastdose.cuh"
#include "cuda_runtime.h"

#include <fstream>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
namespace fd = fastdose;

std::string formatFloat(float number, int digits) {
    std::string formattedString = std::to_string(number);
    int decimalPointPos = formattedString.find('.');
    if (decimalPointPos != std::string::npos && decimalPointPos < formattedString.size()-1) {
        formattedString.resize(decimalPointPos + 2);
    }
    return formattedString;
}

bool PlanOptm::testCase(
    fd::DENSITY_h& density_h,
    fd::DENSITY_d& density_d,
    fd::SPECTRUM_h& spectrum_h,
    fd::KERNEL_h& kernel_h,
    cudaStream_t stream
) {
    // beam initialization
    fd::BEAM_h beam_h;
    fd::BEAM_d beam_d;

    beam_h.angles.x = 0.f;
    beam_h.angles.y = 0.f;
    beam_h.angles.z = 0.f;
    beam_h.sad = 100.f;
    beam_h.isocenter.x = 12.875f;
    beam_h.isocenter.y = 12.875f;
    beam_h.isocenter.z = 12.875f;
    beam_h.beamlet_size.x = 0.1f;
    beam_h.beamlet_size.y = 0.1f;
    beam_h.fmap_size.x = 16;
    beam_h.fmap_size.y = 16;
    beam_h.long_spacing = 0.25f;
    
    int nbeamlets = beam_h.fmap_size.x * beam_h.fmap_size.y;
    beam_h.fluence = std::vector<float>(nbeamlets);
    int FmapOn = 4;
    int FmapLeadingX = static_cast<int>((beam_h.fmap_size.x - FmapOn) / 2);
    int FmapLeadingY = static_cast<int>((beam_h.fmap_size.y - FmapOn) / 2);
    std::fill(beam_h.fluence.begin(), beam_h.fluence.end(), 0.);
    for (int j=FmapLeadingY; j<FmapLeadingY+FmapOn; j++) {
        for (int i=FmapLeadingX; i<FmapLeadingX+FmapOn; i++) {
            int beamletIdx = i + j * beam_h.fmap_size.x;
            beam_h.fluence[beamletIdx] = 1.f;
        }
    }
    beam_h.calc_range(density_h);

    if (fd::beam_h2d(beam_h, beam_d))
        return 1;


    // Terma calculation preparation
    fd::d_BEAM_d h_beam_d(beam_d);
    fd::d_BEAM_d* d_beams = nullptr;
    checkCudaErrors(cudaMalloc((void**)(&d_beams), sizeof(fd::d_BEAM_d)));
    checkCudaErrors(cudaMemcpy(d_beams, &h_beam_d,
        sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice));

    float** fluence_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&fluence_array, sizeof(float*)));
    checkCudaErrors(cudaMemcpy(fluence_array, &(beam_d.fluence),
        sizeof(float*), cudaMemcpyHostToDevice));

    float** TermaBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&TermaBEV_array, sizeof(float*)));
    checkCudaErrors(cudaMemcpy(TermaBEV_array, &(beam_d.TermaBEV),
        sizeof(float*), cudaMemcpyHostToDevice));
    
    float** DensityBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&DensityBEV_array, sizeof(float*)));
    checkCudaErrors(cudaMemcpy(DensityBEV_array, &(beam_d.DensityBEV),
        sizeof(float*), cudaMemcpyHostToDevice));

    float** DoseBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&DoseBEV_array, sizeof(float*)));
    checkCudaErrors(cudaMemcpy(DoseBEV_array, &(beam_d.DoseBEV),
        sizeof(float*), cudaMemcpyHostToDevice));

    // for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    fd::TermaComputeCollective(
        nbeamlets,
        1,
        d_beams,
        fluence_array,
        TermaBEV_array,
        DensityBEV_array, 
        density_d,
        spectrum_h,
        stream
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Terma time elapsed: " << milliseconds << " [ms]" << std::endl;


    cudaEventRecord(start);

    fd::DoseComputeCollective(
        nbeamlets,
        1,
        d_beams,
        TermaBEV_array,
        DensityBEV_array,
        DoseBEV_array,
        kernel_h.nTheta,
        kernel_h.nPhi,
        stream
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Dose time elapsed: " << milliseconds << " [ms]" << std::endl;

    size_t nVoxelsBEV = beam_h.fmap_size.x * beam_h.fmap_size.y * beam_h.long_dim;
    size_t pitch_host = beam_h.fmap_size.x * beam_h.fmap_size.y;
    std::vector<float> DoseBEV_h(nVoxelsBEV, 0.);
    checkCudaErrors(cudaMemcpy2D(
        DoseBEV_h.data(), pitch_host*sizeof(float),
        beam_d.DoseBEV, beam_d.DoseBEV_pitch,
        pitch_host*sizeof(float), beam_h.long_dim, cudaMemcpyDeviceToHost));
    
    // clean up
    checkCudaErrors(cudaFree(fluence_array));
    checkCudaErrors(cudaFree(TermaBEV_array));
    checkCudaErrors(cudaFree(DensityBEV_array));
    checkCudaErrors(cudaFree(DoseBEV_array));

    
    // transfer BEV to PVCS. Construct texture memory
    cudaArray* DoseBEV_Arr;
    cudaTextureObject_t DoseBEV_Tex;
    cudaExtent volumeSize = make_cudaExtent(
        beam_h.fmap_size.x, beam_h.fmap_size.y, beam_h.long_dim);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&DoseBEV_Arr, &channelDesc, volumeSize));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(DoseBEV_h.data(),
            beam_h.fmap_size.x * sizeof(float),
            beam_h.fmap_size.x, beam_h.fmap_size.y);
    copyParams.dstArray = DoseBEV_Arr;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = DoseBEV_Arr;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&DoseBEV_Tex, &texRes, &texDescr, NULL));

    // Construct destination array
    cudaExtent DosePVCS_Arr_extent = make_cudaExtent(
        density_d.VolumeDim.x * sizeof(float),
        density_d.VolumeDim.y, density_d.VolumeDim.z);
    cudaPitchedPtr DosePVCS_Arr;
    checkCudaErrors(cudaMalloc3D(&DosePVCS_Arr, DosePVCS_Arr_extent));

    std::cout << "beams[0] angles: " << beam_h.angles << std::endl;
    std::cout << "beams[0] source: " << beam_h.source << std::endl;
    BEV2PVCS(beam_d, density_d, DosePVCS_Arr, DoseBEV_Tex, stream);

    // write result
    size_t pitchedVolume = DosePVCS_Arr.pitch / sizeof(float) * 
        density_d.VolumeDim.y * density_d.VolumeDim.z;
    size_t volume = density_d.VolumeDim.x * density_d.VolumeDim.y *
        density_d.VolumeDim.z;
    std::vector<float> DosePVCS_pitched(pitchedVolume);
    std::vector<float> DosePVCS(volume);
    checkCudaErrors(cudaMemcpy(DosePVCS_pitched.data(), DosePVCS_Arr.ptr,
        pitchedVolume*sizeof(float), cudaMemcpyDeviceToHost));
    fd::pitched2contiguous(DosePVCS, DosePVCS_pitched,
        density_d.VolumeDim.x, density_d.VolumeDim.y, density_d.VolumeDim.z,
        DosePVCS_Arr.pitch / sizeof(float));

    float extent = beam_h.fmap_size.x * beam_h.beamlet_size.x;
    float beamlet_length = FmapOn * beam_h.beamlet_size.x;
    std::string extent_str = formatFloat(extent, 2);
    std::string beamlet_length_str = formatFloat(beamlet_length, 2);
    fs::path DosePVCSFile = fs::path(getarg<std::string>("outputFolder")) / 
        (std::string("DosePVCS_") + beamlet_length_str + 
        std::string("_") + extent_str + std::string(".bin"));
    std::ofstream f(DosePVCSFile.string());
    if (! f.is_open()) {
        std::cerr << "Could not open file " << DosePVCSFile.string() << std::endl;
        return 1;
    }
    f.write((char*)DosePVCS.data(), volume*sizeof(float));
    f.close();

    return 0;
}


bool PlanOptm::beamBundleTestCase(
    std::vector<BeamBundle> beam_bundles,
    fastdose::DENSITY_h& density_h,
    fastdose::DENSITY_d& density_d,
    fastdose::SPECTRUM_h& spectrum_h,
    fastdose::KERNEL_h& kernel_h,
    cudaStream_t stream
) {
    BeamBundle& first_beam_bundle = beam_bundles[0];
    int nBeamlets = first_beam_bundle.fluenceDim.x * first_beam_bundle.fluenceDim.y;
    first_beam_bundle.beams_d.resize(nBeamlets);
    for (int i=0; i<nBeamlets; i++) {
        fd::beam_h2d(first_beam_bundle.beams_h[i], first_beam_bundle.beams_d[i]);
    }

    // preparation
    std::vector<fd::d_BEAM_d> h_beams;
    h_beams.reserve(nBeamlets);
    for (int i=0; i<nBeamlets; i++)
        h_beams.push_back(fd::d_BEAM_d(first_beam_bundle.beams_d[i]));
    fd::d_BEAM_d* d_beams = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beams, nBeamlets*sizeof(fd::d_BEAM_d)));
    checkCudaErrors(cudaMemcpy(d_beams, h_beams.data(),
        nBeamlets*sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice));
    
    // allocate fluence array
    std::vector<float*> h_fluence_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_fluence_array[i] = first_beam_bundle.beams_d[i].fluence;
    float** fluence_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&fluence_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(fluence_array, h_fluence_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));
    
    // allocate Terma_array
    std::vector<float*> h_TermaBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_TermaBEV_array[i] = first_beam_bundle.beams_d[i].TermaBEV;
    float** TermaBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&TermaBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(TermaBEV_array, h_TermaBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));
    
    // allocate DenseBEV_array
    std::vector<float*> h_DensityBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_DensityBEV_array[i] = first_beam_bundle.beams_d[i].DensityBEV;
    float** DensityBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&DensityBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(DensityBEV_array, h_DensityBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    // allocate DoseBEV_array
    std::vector<float*> h_DoseBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_DoseBEV_array[i] = first_beam_bundle.beams_d[i].DoseBEV;
    float** DoseBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&DoseBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(DoseBEV_array, h_DoseBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));
    
    size_t fmap_npixels = first_beam_bundle.subFluenceDim.x *
        first_beam_bundle.subFluenceDim.y;

    // for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    fd::TermaComputeCollective(
        fmap_npixels,
        nBeamlets,
        d_beams,
        fluence_array,
        TermaBEV_array,
        DensityBEV_array,
        density_d,
        spectrum_h,
        stream
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Terma time elapsed: " << milliseconds << " [ms]" << std::endl;


    cudaEventRecord(start);

    fd::DoseComputeCollective(
        fmap_npixels,
        nBeamlets,
        d_beams,
        TermaBEV_array,
        DensityBEV_array,
        DoseBEV_array,
        kernel_h.nTheta,
        kernel_h.nPhi,
        stream
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Dose time elapsed: " << milliseconds << " [ms]" << std::endl;


    // retrieve result
    std::string outputFolder = getarg<std::string>("outputFolder");
    fs::path resultFolder = fs::path(outputFolder) / std::string("canonical");
    if (! fs::exists(resultFolder))
        fs::create_directories(resultFolder);


    // destination array
    cudaExtent DosePVCS_Arr_extent = make_cudaExtent(
        density_d.VolumeDim.x * sizeof(float),
        density_d.VolumeDim.y, density_d.VolumeDim.z);
    cudaPitchedPtr DosePVCS_Arr;
    checkCudaErrors(cudaMalloc3D(&DosePVCS_Arr, DosePVCS_Arr_extent));

    cudaArray* DoseBEV_Arr;
    cudaTextureObject_t DoseBEV_Tex;
    
    for (int i=0; i<nBeamlets; i++) {
        fd::BEAM_d& current_beamlet = first_beam_bundle.beams_d[i];

        // copy the Dose data to CPU
        size_t nVoxels = current_beamlet.fmap_size.x *
            current_beamlet.fmap_size.y * current_beamlet.long_dim;
        size_t pitch_host = current_beamlet.fmap_size.x * 
            current_beamlet.fmap_size.y;
        std::vector<float> DoseBEV_cpu(nVoxels, 0.0f);
        checkCudaErrors(cudaMemcpy2D(
            DoseBEV_cpu.data(), pitch_host*sizeof(float),
            current_beamlet.DoseBEV, current_beamlet.DoseBEV_pitch,
            pitch_host*sizeof(float), current_beamlet.long_dim, cudaMemcpyDeviceToHost));
        
        #if false
            // to ensure that the DoseBEV data has valid value
            fs::path DoseBEVFile = resultFolder / (std::string("DoseBEV_beamlet")
                + std::to_string(i+1) + std::string(".bin"));
            std::ofstream f(DoseBEVFile.string());
            if (! f.is_open()) {
                std::cerr << "Cannot open file: " << DoseBEVFile.string() << std::endl;
                return 1;
            }
            f.write((char*)DoseBEV_cpu.data(), nVoxels*sizeof(float));
            f.close();
            continue;
        #endif

        cudaExtent volumeSize = make_cudaExtent(current_beamlet.fmap_size.x,
            current_beamlet.fmap_size.y, current_beamlet.long_dim);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        checkCudaErrors(cudaMalloc3DArray(&DoseBEV_Arr, &channelDesc, volumeSize));

        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr =
            make_cudaPitchedPtr(DoseBEV_cpu.data(),
                current_beamlet.fmap_size.x * sizeof(float),
                current_beamlet.fmap_size.x, current_beamlet.fmap_size.y);
        copyParams.dstArray = DoseBEV_Arr;
        copyParams.extent = volumeSize;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = DoseBEV_Arr;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = false;
        texDescr.addressMode[0] = cudaAddressModeBorder;
        texDescr.addressMode[1] = cudaAddressModeBorder;
        texDescr.addressMode[2] = cudaAddressModeBorder;
        checkCudaErrors(cudaCreateTextureObject(&DoseBEV_Tex, &texRes, &texDescr, NULL));

        // clear destination array
        checkCudaErrors(cudaMemset3D(DosePVCS_Arr, 0., DosePVCS_Arr_extent));
        // calculate
        // fd::BEV2PVCS(current_beamlet, density_d, DosePVCS_Arr, DoseBEV_Tex, stream);
        fd::BEV2PVCS_SuperSampling(current_beamlet,
            density_d, DosePVCS_Arr, DoseBEV_Tex, 5, 2.0f, stream);
        // write result
        size_t pitchedVolume = DosePVCS_Arr.pitch / sizeof(float) *
            density_d.VolumeDim.y * density_d.VolumeDim.z;
        size_t volume = density_d.VolumeDim.x * density_d.VolumeDim.y *
            density_d.VolumeDim.z;
        std::vector<float> DosePVCS_pitched(pitchedVolume);
        std::vector<float> DosePVCS(volume);
        checkCudaErrors(cudaMemcpy(DosePVCS_pitched.data(), DosePVCS_Arr.ptr,
            pitchedVolume*sizeof(float), cudaMemcpyDeviceToHost));
        fd::pitched2contiguous(DosePVCS, DosePVCS_pitched,
            density_d.VolumeDim.x, density_d.VolumeDim.y, density_d.VolumeDim.z,
            DosePVCS_Arr.pitch / sizeof(float));

        fs::path DosePVCSFile = resultFolder / (std::string("DosePVCS_beamlet") 
            + std::to_string(i+1) + std::string(".bin"));
        std::ofstream f1(DosePVCSFile.string());
        if (! f1.is_open()) {
            std::cerr << "Cannot open file: " << DosePVCSFile << std::endl;
            return 1;
        }
        f1.write((char*)DosePVCS.data(), volume*sizeof(float));
        f1.close();

        // clean up texture
        checkCudaErrors(cudaDestroyTextureObject(DoseBEV_Tex));
        checkCudaErrors(cudaFreeArray(DoseBEV_Arr));
    }

    // further clean up
    checkCudaErrors(cudaFree(DosePVCS_Arr.ptr));
    checkCudaErrors(cudaFree(DoseBEV_array));
    checkCudaErrors(cudaFree(DensityBEV_array));
    checkCudaErrors(cudaFree(TermaBEV_array));
    checkCudaErrors(cudaFree(fluence_array));
    checkCudaErrors(cudaFree(d_beams));

    return 0;
}