#include <iomanip>

#include "helper_math.cuh"
#include "math_constants.h"
#include "fastdose.cuh"

#include "IMRTBeamBundle.cuh"
#include "IMRTArgs.h"
#include "utils.cuh"
#include "geomKernels.cuh"

namespace fd = fastdose;

bool IMRT::BeamBundleInit(std::vector<BeamBundle>& beam_bundles,
    const fd::DENSITY_h& density_h,
    const std::vector<StructInfo>& structs
) {
    // calculate the isocenter
    const StructInfo& PTV_struct = structs[0];
    const auto& PTV_mask = PTV_struct.mask;
    const auto& PTV_size = PTV_struct.size;

    double3 isocenter{0.0, 0.0, 0.0};
    size_t PTV_nvoxels = 0;

    for (int k=0; k<PTV_size.z; k++) {
        for (int j=0; j<PTV_size.y; j++) {
            for (int i=0; i<PTV_size.x; i++) {
                size_t idx = i + PTV_size.x * (j + PTV_size.y * k);
                if (PTV_mask[idx] > 0) {
                    // for now, isocenter is unitless
                    isocenter.x += i + 0.5;
                    isocenter.y += j + 0.5;
                    isocenter.z += k + 0.5;
                    PTV_nvoxels += 1;
                }
            }
        }
    }
    isocenter.x /= PTV_nvoxels;
    isocenter.y /= PTV_nvoxels;
    isocenter.z /= PTV_nvoxels;
    
    // convert unitless isocenter to that of cm
    const std::vector<float>& voxelSize = getarg<std::vector<float>>("voxelSize");
    isocenter.x *= voxelSize[0];
    isocenter.y *= voxelSize[1];
    isocenter.z *= voxelSize[2];
    std::cout << "PTV structure name: " << PTV_struct.name << std::endl
        << "Number of voxels: " << PTV_nvoxels << std::endl
        << "Isocetner: (" << isocenter.x << ", " << isocenter.y 
        << ", " << isocenter.z << ") [cm]" << std::endl << std::endl;;
    
    const float SAD = getarg<float>("SAD");
    const int fluenceDim = getarg<int>("fluenceDim");
    const float beamletSize = getarg<float>("beamletSize");
    const int subFluenceDim = getarg<int>("subFluenceDim");
    const int subFluenceOn = getarg<int>("subFluenceOn");
    const float longSpacing = getarg<float>("longSpacing");

    const std::string& beamlist = getarg<std::string>("beamlist");
    std::ifstream f(beamlist);
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << beamlist << std::endl;
        return 1;
    }
    beam_bundles.clear();
    int nBeamsReserve = getarg<int>("nBeamsReserve");
    beam_bundles.reserve(nBeamsReserve);
    std::string tableRow;
    float3 angles_degree;
    while (std::getline(f, tableRow)) {
        beam_bundles.emplace_back(BeamBundle());
        auto& last_beam_bundle = beam_bundles.back();
        std::istringstream iss(tableRow);
        iss >> angles_degree.x >> angles_degree.y >> angles_degree.z;
        last_beam_bundle.angles = angles_degree * CUDART_PI_F / 180;
        last_beam_bundle.isocenter = make_float3(
            isocenter.x, isocenter.y, isocenter.z);
        last_beam_bundle.beamletSize = make_float2(beamletSize, beamletSize);
        last_beam_bundle.fluenceDim = make_int2(fluenceDim, fluenceDim);
        last_beam_bundle.SAD = SAD;
        last_beam_bundle.longSpacing = longSpacing;
        last_beam_bundle.subFluenceDim = make_int2(subFluenceDim, subFluenceDim);
        last_beam_bundle.subFluenceOn = make_int2(subFluenceOn, subFluenceOn);
    }

    beamletFlagInit(beam_bundles, PTV_mask, density_h);

    // initialize beam information
    for (int i=0; i<beam_bundles.size(); i++) {
        BeamBundle& current = beam_bundles[i];
        current.beamletsInit(density_h);
        std::cout << "Beam " << i+1 << ", number of active beamlets: "
            << current.beams_h.size() << std::endl;
    }
    return 0;
}


bool IMRT::beamletFlagInit(std::vector<BeamBundle>& beam_bundles,
    const std::vector<uint8_t>& PTV_mask,
    const fastdose::DENSITY_h& density_h,
    cudaStream_t stream
) {
    // construct mask texture
    fd::DENSITY_d PTV_density;
    PTV_density.VoxelSize = density_h.VoxelSize;
    PTV_density.VolumeDim = density_h.VolumeDim;
    PTV_density.BBoxStart = density_h.BBoxStart;
    PTV_density.BBoxDim = density_h.BBoxDim;

    std::vector<float> PTV_mask_float(PTV_mask.size(), 0.0f);
    for (int i=0; i<PTV_mask_float.size(); i++) {
        PTV_mask_float[i] = PTV_mask[i];
    }

    cudaExtent volumeSize = make_cudaExtent(PTV_density.VolumeDim.x,
        PTV_density.VolumeDim.y, PTV_density.VolumeDim.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&(PTV_density.densityArray), &channelDesc, volumeSize));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = 
        make_cudaPitchedPtr(PTV_mask_float.data(),
            volumeSize.width * sizeof(float),
            volumeSize.width, volumeSize.height);
    copyParams.dstArray = PTV_density.densityArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = PTV_density.densityArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(
        &PTV_density.densityTex, &texRes, &texDescr, NULL));


    // prepare beams
    std::vector<fd::BEAM_h> beams_init;
    beams_init.resize(beam_bundles.size());
    for (int i=0; i<beam_bundles.size(); i++) {
        fd::BEAM_h& current_beam = beams_init[i];
        const BeamBundle& current_bundle = beam_bundles[i];

        current_beam.isocenter = current_bundle.isocenter;
        current_beam.beamlet_size = current_bundle.beamletSize;
        current_beam.fmap_size = make_uint2(current_bundle.fluenceDim.x,
            current_bundle.fluenceDim.y);
        current_beam.sad = current_bundle.SAD;
        current_beam.angles = current_bundle.angles;
        current_beam.long_spacing = current_bundle.longSpacing;
        current_beam.calc_range(density_h);
        #if false
            std::cout << "Beam bundle " << i+1 << ", long dim: "
                << current_beam.long_dim << std::endl;
        #endif
    }
    #if false
        beams_init[0].fluence.resize(beams_init[0].fmap_size.x * beams_init[0].fmap_size.y);
        std::cout << beams_init[0] << std::endl;
        return 0;
    #endif

    std::vector<fd::d_BEAM_d> beams_h(beams_init.size());
    for (int i=0; i<beams_init.size(); i++) {
        fd::d_BEAM_d& current_dest = beams_h[i];
        const fd::BEAM_h current_source = beams_init[i];
        current_dest.isocenter = current_source.isocenter;
        current_dest.beamlet_size = current_source.beamlet_size;
        current_dest.fmap_size = current_source.fmap_size;
        current_dest.sad = current_source.sad;
        current_dest.angles = current_source.angles;
        current_dest.long_spacing = current_source.long_spacing;
        current_dest.lim_min = current_source.lim_min;
        current_dest.lim_max = current_source.lim_max;
        current_dest.long_dim = current_source.long_dim;
        current_dest.source = current_source.source;
    }

    fd::d_BEAM_d* beams_d = nullptr;
    checkCudaErrors(cudaMalloc((void**)&beams_d, beams_h.size()*sizeof(fd::d_BEAM_d)));
    checkCudaErrors(cudaMemcpy(beams_d, beams_h.data(),
        beams_h.size()*sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice));

    uint8_t* fmapOn = nullptr;
    size_t nElements = beams_h.size() * beams_h[0].fmap_size.x 
        * beams_h[0].fmap_size.y;
    checkCudaErrors(cudaMalloc((void**)&fmapOn, nElements*sizeof(uint8_t)));
    checkCudaErrors(cudaMemset(fmapOn, 0, nElements*sizeof(uint8_t)));

    size_t fmap_npixels = beams_init[0].fmap_size.x * beams_init[0].fmap_size.y;
    dim3 blockSize(((fmap_npixels + WARPSIZE - 1) / WARPSIZE) * WARPSIZE, 1, 1);
    dim3 gridSize(beams_h.size(), 1, 1);
    d_beamletFlagInit<<<gridSize, blockSize, 0, stream>>>(beams_d, fmapOn,
        PTV_density.densityTex, PTV_density.VoxelSize, 3);

    std::vector<uint8_t> fmapOn_h(nElements, 0);
    checkCudaErrors(cudaMemcpy(fmapOn_h.data(), fmapOn, nElements*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    # if false
        // to take a view
        std::cout << beams_h[0].fmap_size << std::endl;
        for (int j=0; j<beams_h[0].fmap_size.y; j++) {
            for (int i=0; i<beams_h[0].fmap_size.x; i++) {
                std::cout << std::setw(6) << (int)fmapOn_h[i + j * beams_h[0].fmap_size.x];
            }
            std::cout << std::endl;
        }
    #endif

    // copy the collective result to individual beam bundles
    for (int i=0; i<beam_bundles.size(); i++) {
        BeamBundle& current = beam_bundles[i];
        current.beamletFlag.resize(fmap_npixels);
        for (int j=0; j<fmap_npixels; j++) {
            int jj = j + i * fmap_npixels;
            current.beamletFlag[j] = fmapOn_h[jj];
        }
    }

    checkCudaErrors(cudaFree(beams_d));
    checkCudaErrors(cudaFree(fmapOn));
    return 0;
}


__global__ void
IMRT::d_beamletFlagInit(
    fd::d_BEAM_d* beams_d, uint8_t* fmapOn,
    cudaTextureObject_t PTVTex, float3 voxelSize,
    int superSampling
) {
    int beam_idx = blockIdx.x;
    fd::d_BEAM_d beam = beams_d[beam_idx];
    int beamlet_x = threadIdx.x % beam.fmap_size.x;
    int beamlet_y = threadIdx.x / beam.fmap_size.x;
    if (threadIdx.y >= beam.fmap_size.y)
        return;
    
    uint8_t* fmap_local = fmapOn + beam_idx * beam.fmap_size.x * beam.fmap_size.y;

    uint8_t current_beamlet_on = 0;
    for (int i=0; i<beam.long_dim; i++) {
        float SXD = beam.lim_min + i * beam.long_spacing;
        for (int j=0; j<superSampling; j++) {
            for (int k=0; k<superSampling; k++) {
                float3 BEV_displacement {
                    (beamlet_x + (j + 0.5f) / superSampling -
                        beam.fmap_size.x * 0.5f) * beam.beamlet_size.x,
                    beam.sad,
                    (beamlet_y + (k + 0.5f) / superSampling -
                        beam.fmap_size.y * 0.5f) * beam.beamlet_size.y
                };
                float3 PVCS_displacement = d_inverseRotateBeamAtOriginRHS(
                    BEV_displacement, beam.angles.x, beam.angles.y, beam.angles.z);
                float3 PVCS_coords = beam.source + PVCS_displacement * SXD / beam.sad;
                float3 PVCS_coords_unitless = PVCS_coords / voxelSize;
                float value = tex3D<float>(PTVTex, PVCS_coords_unitless.x,
                    PVCS_coords_unitless.y, PVCS_coords_unitless.z);
                if (value > 0) {
                    current_beamlet_on = true;
                    break;
                }
            }
            if (current_beamlet_on)
                break;
        }
        if (current_beamlet_on)
            break;
    }
    fmap_local[threadIdx.x] = current_beamlet_on;
}


bool IMRT::BeamBundle::beamletsInit(const fd::DENSITY_h& density_h) {
    int n_active_beamlets = 0;
    int fmap_npixels = this->fluenceDim.x * this->fluenceDim.y;
    for (int i=0; i<fmap_npixels; i++) {
        n_active_beamlets += (this->beamletFlag[i] > 0);
    }
    this->beams_h.resize(n_active_beamlets);
    int count = 0;
    for (int j=0; j<this->fluenceDim.y; j++) {
        for (int i=0; i<this->fluenceDim.x; i++) {
            int idx = i + j * this->fluenceDim.x;
            if (this->beamletFlag[idx]) {
                fd::BEAM_h& localBeam = this->beams_h[count];
                count ++;
                this->beamletInit(localBeam, i, j, density_h);
            }
        }
    }
    return 0;
}


bool IMRT::BeamBundle::beamletInit(fastdose::BEAM_h& beam_h, int idx_x,
    int idx_y, const fastdose::DENSITY_h& density_h
) {
    // If we take out the beamlet, which can be abstracted as a 
    // quadrangular pyramid, whose vertices are the four
    // vertices of the beamlet plus the source. Then we do 
    // this thing: to get the maximum inscribed quadrangular 
    // pyramid whose bottom is a rectangle, and all the four 
    // side edges connecting the bottom vertices and the source
    // are of the equal length, so that the beamlet is in the
    // canonical form, and can be characterized with SAD and
    // fluence map
    
    float2 halfFluenceSize{this->fluenceDim.x * this->beamletSize.x * 0.5f,
        this->fluenceDim.y * this->beamletSize.y * 0.5f};
    float3 vertex0{idx_x * this->beamletSize.x - halfFluenceSize.x,
        this->SAD,
        idx_y * this->beamletSize.y - halfFluenceSize.y};
    float3 vertex1{(idx_x + 1) * this->beamletSize.x - halfFluenceSize.x,
        this->SAD,
        idx_y * this->beamletSize.y - halfFluenceSize.y};
    float3 vertex2{(idx_x + 1) * this->beamletSize.x - halfFluenceSize.x,
        this->SAD,
        (idx_y + 1) * this->beamletSize.y - halfFluenceSize.y};
    float3 vertex3{idx_x * this->beamletSize.x - halfFluenceSize.x,
        this->SAD,
        (idx_y + 1) * this->beamletSize.y - halfFluenceSize.y};
    
    // normalize
    vertex0 = normalize(vertex0);
    vertex1 = normalize(vertex1);
    vertex2 = normalize(vertex2);
    vertex3 = normalize(vertex3);

    // case 1, the centerline is the bisector of vertex0 and vertex2
    float3 centerline_case1 = vertex0 + vertex2;
    centerline_case1 = normalize(centerline_case1);
    float3 vertex0_case1 = vertex0 / dot(vertex0, centerline_case1);
    float3 vertex1_case1 = vertex1 / dot(vertex1, centerline_case1);
    float3 vertex2_case1 = vertex2 / dot(vertex2, centerline_case1);
    float3 vertex3_case1 = vertex3 / dot(vertex3, centerline_case1);

    float dist0_case1 = length(vertex0_case1 - centerline_case1);
    float dist1_case1 = length(vertex1_case1 - centerline_case1);
    float dist2_case1 = length(vertex2_case1 - centerline_case1);
    float dist3_case1 = length(vertex3_case1 - centerline_case1);
    if (abs(dist0_case1 - dist2_case1) > eps_fastdose) {
        std::cerr << "Something wrong!" << std::endl;
        return 1;
    }
    float dist_case1 = min(min(dist0_case1, dist1_case1), dist3_case1);
    

    // case 2, the centerline is the bisector of vertex1 and vertex3
    float3 centerline_case2 = vertex1 + vertex3;
    centerline_case2 = normalize(centerline_case2);
    float3 vertex0_case2 = vertex0 / dot(vertex0, centerline_case2);
    float3 vertex1_case2 = vertex1 / dot(vertex1, centerline_case2);
    float3 vertex2_case2 = vertex2 / dot(vertex2, centerline_case2);
    float3 vertex3_case2 = vertex3 / dot(vertex3, centerline_case2);

    float dist0_case2 = length(vertex0_case2 - centerline_case2);
    float dist1_case2 = length(vertex1_case2 - centerline_case2);
    float dist2_case2 = length(vertex2_case2 - centerline_case2);
    float dist3_case2 = length(vertex3_case2 - centerline_case2);
    if (abs(dist1_case2 - dist3_case2) > eps_fastdose) {
        std::cerr << "Something wrong!" << std::endl;
        return 1;
    }
    float dist_case2 = min(min(dist0_case2, dist1_case2), dist2_case2);


    float3 direction;
    float3 co_direction;
    float fluence_height;
    float fluence_width;
    float SAD_result;
    if (dist_case1 > dist_case2) {
        direction = centerline_case1;
        float3 vertex0_case1_plane = vertex0_case1 - centerline_case1;
        float3 vertex1_case1_plane = vertex1_case1 - centerline_case1;
        vertex0_case1_plane = normalize(vertex0_case1_plane);
        vertex1_case1_plane = normalize(vertex1_case1_plane);
        co_direction = vertex0_case1_plane + vertex0_case1_plane;
        co_direction = normalize(co_direction);
        
        // calculate other physical parameters
        SAD_result = this->SAD / centerline_case1.y;
        float3 fluence_half_height = (vertex0_case1_plane + vertex1_case1_plane) * dist_case1 * 0.5f;
        fluence_height = length(fluence_half_height) * SAD_result * 2;
        float3 fluence_half_width = (vertex0_case1_plane - vertex1_case1_plane) * dist_case1 * 0.5f;
        fluence_width = length(fluence_half_width) * SAD_result * 2;
    } else {
        direction = centerline_case2;
        float3 vertex0_case2_plane = vertex0_case2 - centerline_case2;
        float3 vertex1_case2_plane = vertex1_case2 - centerline_case2;
        vertex0_case2_plane = normalize(vertex0_case2_plane);
        vertex1_case2_plane = normalize(vertex1_case2_plane);
        co_direction = vertex0_case2_plane + vertex1_case2_plane;
        co_direction = normalize(co_direction);

        // calculate other physical parameters
        SAD_result = this->SAD / centerline_case2.y;
        float3 fluence_half_height = (vertex0_case2_plane + vertex1_case2_plane) * dist_case2 * 0.5f;
        fluence_height = length(fluence_half_height) * SAD_result * 2;
        float3 fluence_half_width = (vertex0_case2_plane - vertex1_case2_plane) * dist_case2 * 0.5f;
        fluence_width = length(fluence_half_width) * SAD_result * 2;
    }
    float beamletSad = this->SAD / direction.y;
    
    float3 direction_PVCS = fd::inverseRotateBeamAtOriginRHS(
        direction, this->angles.x, this->angles.y, this->angles.z);
    // calculate the angles of the beamlet
    // so that the SAD axis (0, 1, 0), after fd::inverseRotateBeamAtOriginRHS()
    // is direction_PVCS
    float3 SAD_axis{0., 1., 0.};
    float theta = std::acos(dot(direction_PVCS, SAD_axis));
    float sin_minus_phi = direction_PVCS.z * rsqrt(
        direction_PVCS.z * direction_PVCS.z + 
        direction_PVCS.x * direction_PVCS.x +
        eps_fastdose * eps_fastdose);
    float phi = std::asin(sin_minus_phi);
    
    // case 1: theta, phi
    // case 2: theta, PI - phi
    // case 3: -theta, phi
    // case 4: -theta, PI - phi
    
    float theta_result = theta;
    float phi_result = phi;
    float3 inverse_rotate_direction = fd::inverseRotateBeamAtOriginRHS(
        SAD_axis, theta_result, phi_result, 0.);
    float SAD_alignment = dot(inverse_rotate_direction, direction_PVCS);
    if (SAD_alignment < 1 - eps_fastdose) {
        theta_result = theta;
        phi_result = CUDART_PI_F - phi;
        inverse_rotate_direction = fd::inverseRotateBeamAtOriginRHS(
            SAD_axis, theta_result, phi_result, 0.);
        SAD_alignment = dot(inverse_rotate_direction, direction_PVCS);
        if (SAD_alignment < 1 - eps_fastdose) {
            theta_result = -theta;
            phi_result = phi;
            inverse_rotate_direction = fd::inverseRotateBeamAtOriginRHS(
                SAD_axis, theta_result, phi_result, 0.);
            SAD_alignment = dot(inverse_rotate_direction, direction_PVCS);
            if (SAD_alignment < 1 - eps_fastdose) {
                theta_result = -theta;
                phi_result = CUDART_PI_F - phi;
                inverse_rotate_direction = fd::inverseRotateBeamAtOriginRHS(
                    SAD_axis, theta_result, phi_result, 0.);
                SAD_alignment = dot(inverse_rotate_direction, direction_PVCS);
                if (SAD_alignment < 1 - eps_fastdose) {
                    std::cerr << "theta, phi calculation failed, direction_PVCS: "
                        << direction_PVCS << std::endl;
                    return 1;
                }
            }
        }
    }

    // determine coll angle by examining co_direction
    // firstly, assume the coll angle is 0
    float3 co_direction_PVCS = fd::inverseRotateBeamAtOriginRHS(
        co_direction, this->angles.x, this->angles.y, this->angles.z);
    float3 co_direction_BEV{0., 0., -1.};
    float3 co_direction_inverse_rotate = fd::inverseRotateBeamAtOriginRHS(
        co_direction_BEV, theta_result, phi_result, 0.);
    float cos_coll = dot(co_direction_PVCS, co_direction_inverse_rotate);
    // clamp cos_coll in case of arithmatic instability
    cos_coll = (cos_coll < -1.0f) ? -1.0f : ((cos_coll > 1.0f) ? 1.0f : cos_coll);
    float coll = std::acos(cos_coll);

    // case 1: coll
    // case 2: - coll
    float coll_result = coll;
    co_direction_inverse_rotate = fd::inverseRotateBeamAtOriginRHS(
        co_direction_BEV, theta_result, phi_result, coll_result);
    float co_direction_alignment = dot(
        co_direction_inverse_rotate, co_direction_PVCS);
    if (co_direction_alignment < 1 - eps_fastdose*10) {
        coll_result = - coll;
        co_direction_inverse_rotate = fd::inverseRotateBeamAtOriginRHS(
            co_direction_BEV, theta_result, phi_result, coll_result);
        float co_direction_alignment_ = dot(
            co_direction_inverse_rotate, co_direction_PVCS);
        if (co_direction_alignment_ < 1 - eps_fastdose*10) {
            std::cerr << "coll calculation failed, co-direction: "
                << co_direction_PVCS << ", alignment values: "
                << co_direction_alignment << ", " 
                << co_direction_alignment_ << std::endl;
            return 1;
        }
    }


    beam_h.angles = make_float3(theta_result, phi_result, coll_result);

    float3 displacement_BEV{
        (idx_x + 0.5f - 0.5f * this->fluenceDim.x) * this->beamletSize.x,
        0,
        (idx_y + 0.5f - 0.5f * this->fluenceDim.y) * this->beamletSize.y};
    float3 displacement_PVCS = fd::inverseRotateBeamAtOriginRHS(
        displacement_BEV, this->angles.x, this->angles.y, this->angles.z);
    beam_h.isocenter = this->isocenter + displacement_PVCS;

    beam_h.beamlet_size = make_float2(
        fluence_width / this->subFluenceOn.x,
        fluence_height / this->subFluenceOn.y);
    beam_h.fmap_size = make_uint2(
        static_cast<uint>(this->subFluenceDim.x),
        static_cast<uint>(this->subFluenceDim.y));
    beam_h.sad = SAD_result;
    beam_h.long_spacing = this->longSpacing;

    beam_h.fluence.resize(beam_h.fmap_size.x * beam_h.fmap_size.y);
    std::fill(beam_h.fluence.begin(), beam_h.fluence.end(), 0.);
    int FmapLeadingX = static_cast<int>((beam_h.fmap_size.x - this->subFluenceOn.x) * 0.5f);
    int FmapLeadingY = static_cast<int>((beam_h.fmap_size.y - this->subFluenceOn.y) * 0.5f);
    for (int j=FmapLeadingY; j<FmapLeadingY + this->subFluenceOn.y; j++) {
        for (int i=FmapLeadingX; i<FmapLeadingX + this->subFluenceOn.x; i++) {
            int index = i + j * beam_h.fmap_size.x;
            beam_h.fluence[index] = 1.0f;
        }
    }

    beam_h.calc_range(density_h);

    return 0;
}