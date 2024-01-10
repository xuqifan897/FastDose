#include <string>
#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <H5Cpp.h>
#include "IMRTInit.cuh"
#include "IMRTArgs.h"

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


bool IMRT::densityInit(fd::DENSITY_h& density_h, fd::DENSITY_d& density_d) {
    const std::vector<float>& voxelSize = getarg<std::vector<float>>("voxelSize");
    const std::vector<int>& phantomDim = getarg<std::vector<int>>("phantomDim");
}