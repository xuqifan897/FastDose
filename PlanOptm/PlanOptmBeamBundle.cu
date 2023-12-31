#include <string>
#include <fstream>
#include <algorithm>
#include <math.h>
#include "helper_math.cuh"
#include "math_constants.h"

#include "PlanOptmBeamBundle.cuh"
#include "PlanOptmArgs.cuh"
namespace fd = fastdose;

bool PlanOptm::BeamBundleInit(
    std::vector<BeamBundle>& beam_bundles, const fd::DENSITY_h& density_h
) {
    const std::vector<float>& isocenter = getarg<std::vector<float>>("isocenter");
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

    int count = 0;
    while (std::getline(f, tableRow)) {
        beam_bundles.emplace_back(BeamBundle());
        auto& last_beam_bundle = beam_bundles.back();
        std::istringstream iss(tableRow);
        iss >> last_beam_bundle.angles.x >> last_beam_bundle.angles.y
            >> last_beam_bundle.angles.z;
        last_beam_bundle.isocenter = float3{isocenter[0], isocenter[1], isocenter[2]};
        last_beam_bundle.beamletSize = float2{beamletSize, beamletSize};
        last_beam_bundle.fluenceDim = int2{fluenceDim, fluenceDim};
        last_beam_bundle.SAD = SAD;
        last_beam_bundle.longSpacing = longSpacing;
        last_beam_bundle.subFluenceDim = int2{subFluenceDim, subFluenceDim};
        last_beam_bundle.subFluenceOn = int2{subFluenceOn, subFluenceOn};

        #if false
            if (count == 0) {
                // modify the first beam into canonical form
                last_beam_bundle.angles = make_float3(0.f, 0.f, 0.f);
            }
        #endif

        // convert degree to radian
        last_beam_bundle.angles *= CUDART_PI_F / 180.f;
        if (last_beam_bundle.beamletsInit(density_h))
            return 1;

        #if false
            // for debug purposes
            if (count == 100) {
                // for debug purposes
                float3 minus_SAD_BEV{0.f, -last_beam_bundle.SAD, 0.f};
                float3 beamBundleSource = last_beam_bundle.isocenter + 
                    fd::inverseRotateBeamAtOriginRHS(minus_SAD_BEV,
                        last_beam_bundle.angles.x, last_beam_bundle.angles.y,
                        last_beam_bundle.angles.z);

                float3 SAD_BEV_direction{0.f, 1.f, 0.f};
                float3 SAD_PVCS_beambundle =
                    fd::inverseRotateBeamAtOriginRHS(SAD_BEV_direction,
                    last_beam_bundle.angles.x, last_beam_bundle.angles.y,
                    last_beam_bundle.angles.z);

                for (int i=0; i<last_beam_bundle.beams_h.size(); i++) {
                    std::cout << last_beam_bundle.beams_h[i];
                    std::cout << "Beam bundle isocenter: " << last_beam_bundle.isocenter << std::endl;
                    std::cout << "Beam bundle angles: " << last_beam_bundle.angles << std::endl;
                    std::cout << "Beam bundle source: " << beamBundleSource << std::endl;

                    const float3& angles = last_beam_bundle.beams_h[i].angles;
                    float3 SAD_PVCS_beam = fd::inverseRotateBeamAtOriginRHS(
                        SAD_BEV_direction, angles.x, angles.y, angles.z);
                    float ang_disp = dot(SAD_PVCS_beam, SAD_PVCS_beambundle);
                    ang_disp = std::acos(ang_disp);
                    std::cout << "Angle between beam bundle and beam: " <<
                        ang_disp * 180 / CUDART_PI_F << " degree" << std::endl << std::endl;
                }
                break;
            }
        #endif

        count ++;
    }

    return 0;
}


bool PlanOptm::BeamBundle::beamletsInit(const fd::DENSITY_h& density_h) {
    int n_beamlets = this->fluenceDim.x * this->fluenceDim.x;
    this->beams_h.reserve(n_beamlets);

    for (int j=0; j<this->fluenceDim.y; j++) {
        for (int i=0; i<this->fluenceDim.x; i++) {
            this->beams_h.emplace_back(fd::BEAM_h());
            auto& last_beam = this->beams_h.back();
            if (this->beamletInit(last_beam, i, j, density_h)) {
                return 1;
            }
        }
    }
    return 0;
}


bool PlanOptm::BeamBundle::beamletInit(
    fd::BEAM_h& beam_h, int idx_x, int idx_y,
    const fd::DENSITY_h& density_h
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
        float3 fluence_half_height = (vertex0_case1 + vertex1_case1
            ) * 0.5f - centerline_case1;
        fluence_height = length(fluence_half_height) * SAD_result * 2;
        float3 fluence_half_width = (vertex0_case1 + vertex3_case1
            ) * 0.5 - centerline_case1;
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
        float3 fluence_half_height = (vertex0_case2 + vertex1_case2
            ) * 0.5 - centerline_case2;
        fluence_height = length(fluence_half_height) * SAD_result * 2;
        float3 fluence_half_width = (vertex0_case2 + vertex3_case2
            ) * 0.5 - centerline_case2;
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