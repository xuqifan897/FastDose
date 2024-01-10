#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <H5Cpp.h>
namespace fs = boost::filesystem;

// Convert cuda vector types to c-style arrays
#define VECT2ARR(a, v) a[0] = v.x; a[1] = v.y;
#define VECT3ARR(a, v) a[0] = v.x; a[1] = v.y; a[2] = v.z;
// Convert c-style array to cuda vector types
#define ARR2VECT(v, a) v.x = a[0]; v.y = a[1];
#define ARR3VECT(v, a) v.x = a[0]; v.y = a[1]; v.z = a[2];

// std::cout formatting
#define FORMAT_3VEC(v) "("<<v.x<<", "<<v.y<<", "<<v.z<<")"
#define FORMAT_2VEC(v) "("<<v.x<<", "<<v.y<<")"


bool getMask(std::vector<uint8_t>& array, const fs::path& fileName, const uint3& shape) {
    size_t size = shape.x * shape.y * shape.z;
    array.resize(size);
    std::vector<uint8_t> temp(size, 0);

    std::ifstream f(fileName.string());
    if (! f.is_open()) {
        std::cerr << "Could not open file: " << fileName << std::endl;
        return 1;
    }
    f.read((char*)(temp.data()), size*sizeof(uint8_t));
    f.close();

    // transpose
    for (int k=0; k<shape.z; k++) {
        for (int j=0; j<shape.y; j++) {
            for (int i=0; i<shape.x; i++) {
                size_t idx_target = i + shape.x * (j + shape.y * k);
                size_t idx_source = j + shape.y * (i + shape.x * k);
                array[idx_target] = temp[idx_source];
            }
        }
    }
    return 0;
}


int main(int argc, char** argv) {
    std::string structPath("/data/qifan/FastDoseWorkplace/BOOval/LUNG/fullStruct");
    std::string outputFile("/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/structs.h5");
    uint3 shape{220, 220, 149};

    std::vector<std::string> structures;
    try {
        fs::directory_iterator dir_iter(structPath);
        for (const auto& entry : dir_iter) {
            if (fs::is_regular_file(entry.status())) {
                fs::path file_local(entry.path().string());
                std::string structName = file_local.stem().string();
                structures.push_back(structName);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // move the PTV and BODY to the front
    std::string target = std::string("BODY");
    auto remove_iterator = std::remove(structures.begin(), structures.end(), target);
    structures.erase(remove_iterator, structures.end());
    structures.insert(structures.begin(), target);

    target = std::string("PTV");
    remove_iterator = std::remove(structures.begin(), structures.end(), target);
    structures.erase(remove_iterator, structures.end());
    structures.insert(structures.begin(), target);
    for (int i=0; i<structures.size(); i++) {
        std::cout << structures[i] << std::endl;
    }

    H5::H5File h5file(outputFile, H5F_ACC_TRUNC);
    H5::Group rootgroup = h5file.openGroup("/");
    uint index = 0;
    for (const auto& structName : structures) {
        ++index;
        auto roi_group = rootgroup.createGroup(structName);

        // create attribute - name
        {
            H5::DataSpace scalarspace;
            H5::StrType str_t{H5::PredType::C_S1, structName.length()+1};
            auto att = roi_group.createAttribute("name", str_t, scalarspace);
            att.write(str_t, structName);
        }

        // write array props to group
        {
            H5::DataSpace scalarspace {};
            hsize_t tuple3_dims[] = { 3 };
            H5::DataSpace tuple3(1, tuple3_dims);
            H5::ArrayType tuple3_native_t(H5::PredType::NATIVE_UINT, 1, tuple3_dims);
            H5::ArrayType tuple3_t(H5::PredType::STD_U16LE, 1, tuple3_dims);

            auto array_props_group = roi_group.createGroup("ArrayProps");
            uint temp[3];

            VECT3ARR(temp, shape);
            auto att = array_props_group.createAttribute("size", tuple3_t, scalarspace);
            att.write(tuple3_native_t, temp);

            // crop size is the same as size
            att = array_props_group.createAttribute("crop_size", tuple3_t, scalarspace);
            att.write(tuple3_native_t, temp);

            temp[0] = 0;
            temp[1] = 0;
            temp[2] = 0;
            att = array_props_group.createAttribute("crop_start", tuple3_t, scalarspace);
            att.write(tuple3_native_t, temp);
        }
        
        // create dataset - dense mask
        {
            hsize_t dims[] = {shape.x * shape.y * shape.z};
            H5::DataSpace simplespace(1, dims);
            auto dset = roi_group.createDataSet("mask", H5::PredType::STD_U8LE, simplespace);

            fs::path file(structPath);
            file /= (structName + std::string(".bin"));
            std::vector<uint8_t> data;
            if (getMask(data, file, shape)) {
                std::cerr << "Mask load error." << std::endl;
                return 1;
            }
            dset.write(data.data(), H5::PredType::NATIVE_UINT8);
        }

        H5::DataSpace scalarspace {};
        H5::Attribute att = roi_group.createAttribute("index",
            H5::PredType::STD_U16LE, scalarspace);
        att.write(H5::PredType::NATIVE_UINT, &index);
    }
}