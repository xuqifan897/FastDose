#include <string>
#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <H5Cpp.h>
#include "IMRTInit.cuh"
#include "IMRTArgs.h"
#include "helper_math.cuh"

namespace fd = fastdose;
namespace fs = boost::filesystem;


bool IMRT::StructsInit(std::vector<StructInfo>& structs, bool verbose) {
    std::string structureInfoFile = getarg<std::string>("structureInfo");
    std::ifstream f(structureInfoFile);
    if (! f.is_open()) {
        std::cerr << "Could not open file: " << structureInfoFile << std::endl;
        return 1;
    }

    std::string tableRow;
    bool skipFirst = true;
    while (std::getline(f, tableRow)) {
        // remove the title line
        if (skipFirst) {
            skipFirst = false;
            continue;
        }
        if (tableRow == std::string("\n"))
            break;
        
        std::stringstream issLine(tableRow);
        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(issLine, token, ',')) {
            if (token == std::string("NaN")) {
                tokens.push_back(std::string("0"));
            } else {
                tokens.push_back(token);
            }
        }

        structs.emplace_back(StructInfo());
        StructInfo& last_struct = structs.back();
        last_struct.name = tokens[0];
        last_struct.maxWeights = std::stof(tokens[1]);
        last_struct.maxDose = std::stof(tokens[2]);
        last_struct.minDoseTargetWeights = std::stof(tokens[3]);
        last_struct.minDoseTarget = std::stof(tokens[4]);
        last_struct.OARWeights = std::stof(tokens[5]);
        last_struct.IdealDose = std::stof(tokens[6]);
    }

    // load mask
    const std::string maskFile = getarg<std::string>("masks");
    readMaskFromHDF5(structs, maskFile);

    return 0;
}


bool IMRT::StructsInit_dosecalc(std::vector<StructInfo>& structs, bool verbose) {
    const std::vector<std::string> structures =
        getarg<std::vector<std::string>>("structures");
    for (int i=0; i<structures.size(); i++) {
        structs.emplace_back(StructInfo());
        StructInfo& last_struct = structs.back();
        last_struct.name = structures[i];
        // fill dummy values below
        last_struct.maxWeights = 1.0f;
        last_struct.maxDose = 1.0f;
        last_struct.minDoseTargetWeights = 1.0f;
        last_struct.minDoseTarget = 1.0f;
        last_struct.OARWeights = 1.0f;
        last_struct.IdealDose = 1.0f;
    }

    // load mask
    const std::string maskFile = getarg<std::string>("masks");
    readMaskFromHDF5(structs, maskFile);

    return 0;
}


bool IMRT::readMaskFromHDF5(std::vector<StructInfo>& structs, const std::string& h5file) {
    struct IndexedString {
        uint16_t idx;
        std::string str;
    };
    struct OpData {
        OpData(H5::Group& g) : h5group(g) {};
        std::list<IndexedString> groups={};
        H5::Group& h5group;
    };

    H5::H5File h5file_(h5file, H5F_ACC_RDONLY);
    H5::Group rootgroup = h5file_.openGroup("/");
    OpData opdata(rootgroup);
    int iter_idx = 0;  // iter_count is returned here

    auto opFunc = [](hid_t loc_id, const char* name, void* opdata) -> herr_t {
        // iterator body
        // construct an IndexedString for each group and add to "groups" list
        OpData* data = static_cast<OpData*>(opdata);
        H5::Group roi_group = data->h5group.openGroup(name);
        auto att = roi_group.openAttribute("index");
        uint16_t index; att.read(H5::PredType::NATIVE_UINT16, &index);
        data->groups.push_back( IndexedString{index, std::string(name)} );
        return 0;
    };
    rootgroup.iterateElems(".", &iter_idx, opFunc, (void*)&opdata);

    // sort (index, groupname) list on index ascending
    opdata.groups.sort( [](IndexedString& a, IndexedString& b) -> bool {
        // true if a belongs before b
        return (a.idx <= b.idx);
        } );

    # if false
        // for debug purposes
        // to show all structure names
        for (auto v : opdata.groups) {
            std::cout << "Structure " << v.idx << ":" << v.str << std::endl;
        }
        return 0;
    #endif

    const std::vector<int>& phantomDim = getarg<std::vector<int>>("phantomDim");
    for(int i=0; i<structs.size(); i++) {
        const std::string struct_name = structs[i].name;
        std::string key_name;
        bool struct_found = false;
        int struct_index;

        if (struct_name == std::string("PTV")) {
            struct_found = true;
            struct_index = 1;
            for (auto v : opdata.groups) {
                if (struct_index == v.idx) {
                    key_name = v.str;
                    break;
                }
            }
        } else if (struct_name == std::string("BODY")) {
            struct_found = true;
            struct_index = 2;
            for (auto v : opdata.groups) {
                if (struct_index == v.idx) {
                    key_name = v.str;
                    break;
                }
            }
        } else {
            key_name = struct_name;
            for (auto v : opdata.groups) {
                if (struct_name == v.str) {
                    struct_index = v.idx;
                    struct_found = true;
                    break;
                }
            }
        }

        if (! struct_found) {
            std::cerr << "Structure: " << struct_name << " not found" << std::endl << std::endl;
            continue;
        }

        H5::Group roi_group = rootgroup.openGroup(key_name);
        // read attribute - name
        {
            auto att = roi_group.openAttribute("name");
            H5::DataType str_t = att.getDataType();
            H5std_string buff("");
            att.read(str_t, buff);
        }

        // read group - ArrayProps
        {
            hsize_t tuple3_dims[] = {3};
            H5::ArrayType tuple3_native_t(H5::PredType::NATIVE_UINT, 1, tuple3_dims);
            H5::Group props_group = roi_group.openGroup("ArrayProps");

            uint temp[3];
            auto att = props_group.openAttribute("size");
            att.read(tuple3_native_t, temp);
            ARR3VECT(structs[i].size, temp);
            if (structs[i].size.x != phantomDim[0] || 
                structs[i].size.y != phantomDim[1] || 
                structs[i].size.z != phantomDim[2]
            ) {
                std::cerr << "Phantom dimension and structure dimension doesn't match!" << std::endl;
                std::cerr << structs[i].size << " != phantomDim" << std::endl;
                return 1;
            } 

            att = props_group.openAttribute("crop_size");
            att.read(tuple3_native_t, temp);
            ARR3VECT(structs[i].crop_size, temp);

            att = props_group.openAttribute("crop_start");
            att.read(tuple3_native_t, temp);
            ARR3VECT(structs[i].crop_start, temp);
        }

        // read datasaet - dense mask
        {
            H5::DataSet dset = roi_group.openDataSet("mask");
            H5::DataSpace dspace = dset.getSpace();
            hsize_t N;
            dspace.getSimpleExtentDims(&N, nullptr);
            std::vector<uint8_t> mask_temp(N, 0);
            dset.read(mask_temp.data(), H5::PredType::NATIVE_UINT8, dspace, dspace);

            const auto& size = structs[i].size;
            const auto& crop_size = structs[i].crop_size;
            const auto& crop_start = structs[i].crop_start;

            size_t mask_size = size.x * size.y * size.z;
            structs[i].mask.resize(mask_size);
            std::fill(structs[i].mask.begin(), structs[i].mask.end(), 0);

            for (int kk=0; kk<crop_size.z; kk++) {
                int k_full = kk + crop_start.z;
                for (int jj=0; jj<crop_size.y; jj++) {
                    int j_full = jj + crop_start.y;
                    for (int ii=0; ii<crop_size.x; ii++) {
                        int i_full = ii + crop_start.x;
                        size_t idx_crop = ii + crop_size.x * (jj + crop_size.y * kk);
                        size_t idx_full = i_full + size.x * (j_full + size.y * k_full);
                        structs[i].mask[idx_full] = mask_temp[idx_crop];
                    }
                }
            }
        }

        if (true) {
            std::cout << structs[i] << std::endl << std::endl;
        }
    }
    return 0;
}


std::ostream& IMRT::operator<<(std::ostream& os, const StructInfo& obj) {
    os << "Structure: " << obj.name << std::endl;
    os << "maxWeights: " << std::scientific << std::setprecision(2) << obj.maxWeights << std::endl;
    os << "maxDose: " << std::scientific << obj.maxDose << std::endl;
    os << "minDoseTargetWeights: " << obj.minDoseTargetWeights << std::endl;
    os << "minDoseTarget: " << obj.minDoseTarget << std::endl;
    os << "OARWeights: " << obj.OARWeights << std::endl;
    os << "IdealDose: " << obj.IdealDose << std::endl;
    
    size_t voxels_on = 0;
    size_t mask_size = obj.size.x * obj.size.y * obj.size.z;
    for (size_t i=0; i<mask_size; i++)
        voxels_on += (obj.mask[i] > 0);
    os << "Number of active voxels: " << voxels_on;
    
    return os;
}


bool IMRT::densityInit(fd::DENSITY_h& density_h, fd::DENSITY_d& density_d,
    const std::vector<StructInfo>& structs
) {
    const std::vector<float>& voxelSize = getarg<std::vector<float>>("voxelSize");
    const std::vector<int>& phantomDim = getarg<std::vector<int>>("phantomDim");

    uint3 bbox_start, bbox_size;
    const std::string& bbox_name = getarg<std::string>("bboxROI");
    const StructInfo* bbox_ptr = nullptr;
    for (int i=0; i<structs.size(); i++) {
        if (structs[i].name == bbox_name) {
            bbox_ptr = & structs[i];
            break;
        }
    }
    if (bbox_ptr == nullptr) {
        std::cerr << "The bounding box structure " << bbox_name << " was not found" << std::endl;
        return 1;
    }
    std::cout << "Bounding box structure name: " << bbox_name << std::endl;
    const std::vector<uint8_t>& bbox = bbox_ptr->mask;
    const uint3& shape = structs[1].size;
    getBBox(bbox, shape, bbox_start, bbox_size);
    std::cout << "";
    
    density_h.VoxelSize = float3{voxelSize[0], voxelSize[1], voxelSize[2]};
    density_h.VolumeDim = uint3{(uint)phantomDim[0], (uint)phantomDim[1], (uint)phantomDim[2]};
    density_h.BBoxStart = bbox_start;
    density_h.BBoxDim = bbox_size;
    std::cout << "BBoxStart: " << bbox_start << ", BBoxDim: "
        << bbox_size << std::endl << std::endl;

    size_t volumeSize = phantomDim[0] * phantomDim[1] * phantomDim[2];
    density_h.density.resize(volumeSize);

    const std::string& densityFile = getarg<std::string>("density");
    std::ifstream f(densityFile);
    if (! f.is_open()) {
        std::cerr << "Could not open file: " << densityFile << std::endl;
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


bool IMRT::getBBox(const std::vector<uint8_t>& bbox, const uint3 shape,
    uint3& bbox_start, uint3& bbox_size
) {
    // initialize
    uint3 bbox_end{0, 0, 0};
    bbox_start = shape - 1;
    for (int i=0; i<shape.x; i++) {
        for (int j=0; j<shape.y; j++) {
            for (int k=0; k<shape.z; k++) {
                size_t idx = i + shape.x * (j + shape.y * k);
                if (bbox[idx] > 0) {
                    bbox_start.x = min(bbox_start.x, i);
                    bbox_start.y = min(bbox_start.y, j);
                    bbox_start.z = min(bbox_start.z, k);

                    bbox_end.x = max(bbox_end.x, i);
                    bbox_end.y = max(bbox_end.y, j);
                    bbox_end.z = max(bbox_end.z, k);
                }
            }
        }
    }
    bbox_size = bbox_end + 1 - bbox_start;
    return 0;
}


bool IMRT::specInit(fastdose::SPECTRUM_h& spectrum_h) {
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


bool IMRT::kernelInit(fastdose::KERNEL_h& kernel_h) {
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


bool IMRT::ParamsInit(Params& params) {
    std::string paramsFile = IMRT::getarg<std::string>("params");
    std::ifstream f(paramsFile);
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << paramsFile << std::endl;
        return 1;
    }
    std::vector<std::vector<std::string>> data;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::vector<std::string> row;
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(value);
        }
        data.push_back(row);
    }
    
    std::vector<std::string> keys {
        "beamWeight",
        "gamma",
        "eta",
        "numBeamsWeWant",
        "stepSize",
        "maxIter",
        "showTrigger",
        "ChangeWeightsTrigger",
        "pruneTrigger"
    };

    if (data.size() != keys.size()) {
        std::cerr << "We expect " << keys.size() << " lines, but the file has "
            << data.size() << " lines." << std::endl;
        return 1;
    }
    for (int i=0; i<keys.size(); i++) {
        const std::vector<std::string> row = data[i];
        if (row[0] != keys[i]) {
            std::cerr << "Key unmatch: " << row[0] << " != " << keys[i] << std::endl;
            return 1;
        }
    }
    params.beamWeight = std::stof(data[0][1]);
    params.gamma = std::stof(data[1][1]);
    params.eta = std::stof(data[2][1]);
    params.numBeamsWeWant = std::stoi(data[3][1]);
    params.stepSize = std::stof(data[4][1]);
    params.maxIter = std::stoi(data[5][1]);
    params.showTrigger = std::stoi(data[6][1]);
    params.changeWeightsTrigger = std::stoi(data[7][1]);

    params.pruneTrigger.resize(data[8].size() - 1);
    for (int i=0; i<params.pruneTrigger.size(); i++)
        params.pruneTrigger[i] = std::stoi(data[8][i+1]);

    std::vector<std::string> publicMemberValues;
    std::stringstream valueStream;
#define MEMBER(type, name) \
    valueStream << std::scientific << std::setprecision(2) << params.name; \
    publicMemberValues.push_back(#name + std::string(": ") + valueStream.str()); \
    valueStream.str("");

    DECLARE_PUBLIC_MEMBERS
    for (int i=0; i<params.pruneTrigger.size(); i++)
        valueStream << params.pruneTrigger[i] << ", ";
    publicMemberValues.push_back("pruneTrigger: " + valueStream.str());
#undef MEMBER

    std::cout << std::scientific << std::setprecision(2) <<
        "Optimization parameters: " << std::endl;
    for (const std::string& value : publicMemberValues) {
        std::cout << "    " << value << std::endl;
    }
    std::cout << std::endl;

    return 0;
}