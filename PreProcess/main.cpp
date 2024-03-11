#include "PreProcessArgs.h"
#include "PreProcessInit.h"
#include "rtstruct.h"
#include "PreProcessROI.h"
#include "PreProcessHelper.h"
#include "PreProcessRingStruct.cuh"
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
    if (PreProcess::argparse(argc, argv))
        return 0;

    std::vector<std::string> roi_names;
    PreProcess::getROINamesFromJSON(PreProcess::getarg<
        std::string>("structuresFile"), roi_names);
    std::cout << roi_names << std::endl;

    const std::string& dicomFolder = PreProcess::getarg<std::string>("dicomFolder");
    std::string ptv_name = PreProcess::getarg<std::string>("ptv_name");
    bool verbose = PreProcess::getarg<bool>("verbose");

    PreProcess::RTImage* rtimage = nullptr;
    PreProcess::dicomInit(dicomFolder, &rtimage);

    PreProcess::FloatVolume ctdata {};
    // get dicom data size
    ctdata.size   = make_uint3(rtimage->getDataSize());
    // get voxel dimensions, convert to cm
    ctdata.voxsize     = 0.1f * rtimage->getVoxelSize();
    // get position of voxel 0,0,0 - convert to cm
    ctdata.start   = 0.1f * rtimage->getSliceImagePositionPatient( 0 );
    // HU data volume
    ctdata.set_data(rtimage->getDataArray(), ctdata.nvoxels());

    auto ctLUT = PreProcess::CTLUT();
    if (! PreProcess::getarg<bool>("nolut")) {
        const std::string& ctlutFile = PreProcess::getarg<std::string>("ctlutFile");
        if (ctlutFile != std::string("")) {
            ctLUT.label = "User Specified";
            if (!load_lookup_table(ctLUT, ctlutFile)) {
                char msg[300];
                sprintf(msg, "Failed to read from ct lookup table: \"%s\"", ctlutFile.c_str());
                throw std::runtime_error(msg);
            }
        } else {
            ctLUT.label = "Siemens (default)";
            // LUT from: http://sbcrowe.net/ct-density-tables/
            ctLUT.points.emplace_back("Air",           -969.8f, 0.f    ) ;
            ctLUT.points.emplace_back("Lung 300",      -712.9f, 0.290f ) ;
            ctLUT.points.emplace_back("Lung 450",      -536.5f, 0.450f ) ;
            ctLUT.points.emplace_back("Adipose",       -95.6f,  0.943f ) ;
            ctLUT.points.emplace_back("Breast",        -45.6f,  0.985f ) ;
            ctLUT.points.emplace_back("Water",         -5.6f,   1.000f ) ;
            ctLUT.points.emplace_back("Solid Water",   -1.9f,   1.016f ) ;
            ctLUT.points.emplace_back("Brain",         25.7f,   1.052f ) ;
            ctLUT.points.emplace_back("Liver",         65.6f,   1.089f ) ;
            ctLUT.points.emplace_back("Inner Bone",    207.5f,  1.145f ) ;
            ctLUT.points.emplace_back("B-200",         220.7f,  1.159f ) ;
            ctLUT.points.emplace_back("CB2 30%",       429.9f,  1.335f ) ;
            ctLUT.points.emplace_back("CB2 50%",       775.3f,  1.560f ) ;
            ctLUT.points.emplace_back("Cortical Bone", 1173.7f, 1.823f ) ;
        }
        ctLUT.sort();
        std::cout << ctLUT << std::endl;
    }

    PreProcess::FloatVolume density;  // structure to hold density data
    if (PreProcess::CreateIsoDensity(ctdata, density, &ctLUT)) {
        std::cout << "Failed reading CT data!" << std::endl;
        return 1;
    }
    
    PreProcess::FrameOfReference frameofref {density.size, density.start, density.voxsize};


    // Loat rtstruct data
    PreProcess::RTStruct rtstruct {};
    rtstruct.setDicomDirectory(dicomFolder.c_str());
    if ( !rtstruct.loadDicomInfo(verbose) ) {
        printf(" Couldn't load rtstruct from \"%s\". exiting\n", dicomFolder.c_str());
        return 1;
    }
    rtstruct.loadRTStructInfo(verbose);
    std::cout << "PTV SELECTION" << std::endl;
    bool  target_exact_match = PreProcess::getarg<bool>("target_exact_match");
    int ptv_idx = getROIIndex(rtstruct, ptv_name, target_exact_match, verbose);
    if (ptv_idx < 0) {
        printf("No contour could be matched from search string: %s. exiting\n", ptv_name.c_str());
        return 1;
    } else {
        ptv_name = std::string{rtstruct.getROIName(ptv_idx)};
        if (!roi_names.empty()) { roi_names.front() = ptv_name; } // overwrite structures setting of ptv
        printf("Structure found: #%d - %s\n",ptv_idx+1, ptv_name.c_str());
    }
    PreProcess::StructureSet ptv;
    if (PreProcess::loadStructureSet(ptv, rtstruct, ptv_idx, verbose)) {
        if (!verbose) std::cout << "Failed to load ROI Data for: \""
            << ptv_name <<"\"" << std::endl;
        return 1;
    }


    std::string bbox_name = PreProcess::getarg<std::string>("bbox_name");
    bool use_default_bbox = false;
    PreProcess::StructureSet bbox_roi {};
    int bbox_idx = PreProcess::getROIIndex(rtstruct, bbox_name, true, verbose);
    if (bbox_idx < 0) {
        use_default_bbox = true;
        std::cout << "No contour could be matched from search string: " << bbox_name
            << ". Using full volume: (" << density.size.x << ", " << density.size.y
            << ", " << density.size.z << ")" << std::endl;
    } else {
        if (bbox_idx == ptv_idx) {
            bbox_name = ptv_name;
            std::cout << "Reusing structure: " << ptv_idx+1 << " - \""
                << bbox_name << "\"" << std::endl;
            bbox_roi = ptv;
        } else {
            bbox_name = std::string(rtstruct.getROIName(bbox_idx));
            std::cout << "Structure found: " << bbox_idx+1 << " - \""
                << bbox_name << "\"" << std::endl;
            if (PreProcess::loadStructureSet(bbox_roi, rtstruct, bbox_idx, verbose)) {
                if (! verbose) {
                    std::cout << "Failed to load ROI data for \"" << bbox_name
                        << "\"" << std::endl;
                    use_default_bbox = true;
                }
            }
        }
    }
    
    // process bbox selection
    uint3 calc_bbox_start {};
    uint3 calc_bbox_size {};
    if (use_default_bbox) {
        calc_bbox_start = make_uint3(0, 0, 0);
        calc_bbox_size = density.size;
    } else {
        PreProcess::ArrayProps extents = PreProcess::getROIExtents(bbox_roi, frameofref);
        calc_bbox_start = extents.crop_size;
        calc_bbox_size = extents.crop_size;
    }

    // hotfix - prevent bbox from meeting x and y edges
    calc_bbox_start.x = max(calc_bbox_start.x, 1);
    calc_bbox_start.y = max(calc_bbox_start.y, 1);
    calc_bbox_start.z = max(calc_bbox_start.z, 1);
    calc_bbox_size.x = min(calc_bbox_size.x, frameofref.size.x-2);
    calc_bbox_size.y = min(calc_bbox_size.y, frameofref.size.y-2);
    calc_bbox_size.z = min(calc_bbox_size.z, frameofref.size.z-2);

    std::cout << std::endl;
    printf("-- bbox start: (%3d, %3d, %3d)\n", calc_bbox_start.x,
        calc_bbox_start.y, calc_bbox_start.z);
    printf("-- bbox size:  (%3d, %3d, %3d)\n", calc_bbox_size.x,
        calc_bbox_size.y, calc_bbox_size.z);
    std::cout << std::endl;


    std::cout << "LOADING OPTIMIZATION STRUCTURES" << std::endl;
    PreProcess::ROIMaskList roi_list {};
    PreProcess::ROI_init(roi_list, roi_names,
        rtstruct, frameofref, ctdata, density, verbose);
    if (PreProcess::CreateRingStructure(roi_list, rtstruct, ctdata, density, verbose)) {
        std::cerr << "Error creating ring structure." << std::endl;
        return 1;
    }
    std::cout << "Discovered " << roi_list.size() << " ROIs:\n";
    uint idx = 0;
    for (const auto& v : roi_list.getROINames()) {
        idx++;
        std::cout << "  " << idx << ": " << v << std::endl;
    }
    fs::path roi_list_output(PreProcess::getarg<std::string>("inputFolder"));
    roi_list_output /= "roi_list.h5";
    std::cout << "\nWriting ROI List to \"" << roi_list_output << "\"" << std::endl;
    roi_list.writeToFile(roi_list_output.string());

    fs::path densityFile(PreProcess::getarg<std::string>("inputFolder"));
    densityFile /= std::string("density.raw");
    PreProcess::write_debug_data<float>(density.data(), density.size,
        densityFile.string().c_str(), verbose);

    // save structure list
    fs::path dimFile(PreProcess::getarg<std::string>("inputFolder"));
    dimFile /= PreProcess::getarg<std::string>("dimFile");
    std::ofstream f(dimFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file " << dimFile << std::endl;
        return 1;
    }
    float voxelSize = PreProcess::getarg<float>("voxelSize");
    f << density.size.x << " " << density.size.y << " " << density.size.z
        << "\n" << voxelSize << " " << voxelSize << " " << voxelSize << "\n";
    for (const auto & v : roi_list.getROINames())
        f << v << " ";
    f.close();
    
    return 0;
}