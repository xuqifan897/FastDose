#include <string>
#include <fstream>
#include <math.h>
#include "helper_math.cuh"

#include "PlanOptmBeamBundle.cuh"
#include "PlanOptmArgs.cuh"
namespace fd = fastdose;

bool PlanOptm::BeamBundleInit(std::vector<BeamBundle>& beam_bundle) {
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
    beam_bundle.clear();
    std::string tableRow;
    while (std::getline(f, tableRow)) {
        beam_bundle.emplace_back(BeamBundle());
        auto& last_beam_bundle = beam_bundle.back();
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

        last_beam_bundle.beamletsInit();

        // for debut purposes
        break;
    }

    return 0;
}


bool PlanOptm::BeamBundle::beamletsInit() {
    int n_beamlets = this->fluenceDim.x * this->fluenceDim.x;
    this->beams_h.reserve(n_beamlets);
    this->beams_d.reserve(n_beamlets);

    for (int j=0; j<this->fluenceDim.y; j++) {
        for (int i=0; i<this->fluenceDim.x; i++) {
            this->beams_h.emplace_back(fd::BEAM_h());
            auto& last_beam = this->beams_h.back();
            if (this->beamletInit(last_beam, i, j)) {
                return 1;
            }
        }
    }
    return 0;
}


bool PlanOptm::BeamBundle::beamletInit(
    fd::BEAM_h& beam_h, int idx_x, int idx_y
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
    if (dist_case1 > dist_case2) {
        direction = centerline_case1;
        co_direction = vertex0_case1 - centerline_case1;
        co_direction = normalize(co_direction);
    } else {
        direction = centerline_case2;
        co_direction = vertex0_case2 - centerline_case2;
        co_direction = normalize(co_direction);
    }
    float beamletSad = this->SAD / direction.y;
    
    float3 direction_PVCS = fd::inverseRotateBeamAtOriginRHS(
        direction, this->angles.x, this->angles.y, this->angles.z);
    // calculate the angles of the beamlet
    // so that the SAD axis (0, 1, 0), after fd::inverseRotateBeamAtOriginRHS()
    // is direction_PVCS
    float3 SAD_axis{0., 1., 0.};
    float theta = - std::acos(dot(direction_PVCS, SAD_axis));
    float sin_minus_phi = direction_PVCS.z * rsqrt(
        direction_PVCS.z * direction_PVCS.z + 
        direction_PVCS.x * direction_PVCS.x +
        eps_fastdose * eps_fastdose);
    float phi = std::asin(sin_minus_phi);

    float3 SAD_inverse_rotate = fd::inverseRotateBeamAtOriginRHS(
        SAD_axis, theta, phi, 0.);

    #if false
    std::cout << "expected: " << direction_PVCS << ", got: " 
        << SAD_inverse_rotate << std::endl;
    #endif

    return 0;
}