#include "PreProcessInit.h"
#include "PreProcessArgs.h"
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <boost/filesystem.hpp>
#include "helper_math.h"

namespace fs = boost::filesystem;

bool PreProcess::getROINamesFromJSON(const std::string& json_path,
    std::vector<std::string>& target) {
    FILE* fp;
    if ( (fp = fopen(json_path.c_str(), "r")) != NULL ) {
        char readbuf[5000];
        rapidjson::FileReadStream is{fp, readbuf, sizeof(readbuf)};
        rapidjson::Document jdoc;
        jdoc.ParseStream(is);

        if (jdoc.IsObject()) {
            rapidjson::Value::ConstMemberIterator ptv_itr = jdoc.FindMember("ptv");
            if (ptv_itr != jdoc.MemberEnd() && ptv_itr->value.IsString()) {
                target.emplace_back(ptv_itr->value.GetString());
            } else {
                throw std::exception();
            }

            rapidjson::Value::ConstMemberIterator oar_itr = jdoc.FindMember("oar");
            if (oar_itr != jdoc.MemberEnd()) {
                if (oar_itr->value.IsArray()) {
                    for (auto& v : oar_itr->value.GetArray()) {
                        if (v.IsString()) {
                            target.emplace_back( v.GetString() );
                        }
                    }
                } else if (oar_itr->value.IsString()) {
                    target.emplace_back( oar_itr->value.GetString() );
                }
            }
        }
        fclose(fp);
    }
    return 0;
}

bool PreProcess::dicomInit(const std::string& dicomFolder, RTImage** rtimage, bool verbose) {
    *rtimage = new RTImage(dicomFolder, verbose);
    const std::array<float, 6>& imageOrientationPatient = (*rtimage)->getSliceImageOrientationPatient(0);
    std::array<float, 6> orient_hfs = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    if (imageOrientationPatient != orient_hfs) {
        std::cout << "Warning: CT Data orientation does not match standard orientation (HFS). Please check your orientation to be sure it is what you expect: [" <<
            imageOrientationPatient[0] << "\\" <<
            imageOrientationPatient[1] << "\\" <<
            imageOrientationPatient[2] << "\\" <<
            imageOrientationPatient[3] << "\\" <<
            imageOrientationPatient[4] << "\\" <<
            imageOrientationPatient[5] << "]" <<
            std::endl;
    }
    return 0;
}

std::ostream& operator<<(std::ostream& out, const PreProcess::FrameOfReference& frame) {
  out << "Frame:" << std::endl
      << "  Size:    ("<<frame.size.x    <<", "<< frame.size.y    <<", "<< frame.size.z <<")"<< std::endl
      << "  Start:   ("<<frame.start.x   <<", "<< frame.start.y   <<", "<< frame.start.z <<")"<< std::endl
      << "  Spacing: ("<<frame.spacing.x <<", "<< frame.spacing.y <<", "<< frame.spacing.z <<")"<< std::endl;
  return out;
}

void PreProcess::CTLUT::sort() {
    std::sort(points.begin(), points.end(),
            [] (const LUTPOINT& p1, const LUTPOINT& p2) { return (p1.hunits < p2.hunits); }
            );
}


std::ostream& operator<<(std::ostream& os, const PreProcess::CTLUT& ctlut) {
    os << "CT Lookup Table ("<<ctlut.label<<"):"<<std::endl;
    os << "---------------------------------------------" << std::endl;
    if (!ctlut.points.size()) {
        os << "  EMPTY" << std::endl;
    } else {
        os << "   #     HU#    g/cm3   rel   label          " << std::endl;
        os << "  ---  -------  -----  -----  ---------------" << std::endl;
        char buf[100];
        int ii = 0;
        for (const auto& pt : ctlut.points) {
            ii++;
            sprintf(buf, "  %3d  %7.1f  %5.3f  %5.3f  %s", ii, pt.hunits, pt.massdens, pt.reledens, pt.label.c_str());
            os << buf << std::endl;
        }
    }
    return os;
}


bool PreProcess::load_lookup_table(CTLUT& lut, std::string filepath, int verbose) {
    /* LUT file format:
     * Each line should contain the material name, CT number, corresponding electron density, and optionally the density relative to water (currently unused)
     *    name   CT#      density  rel_density
     *    <str>  <float>  <float>  <float>
     * lines beginning with "#" are ignored as comments, empty lines between entries are ignored, lines are read until EOF
     * */
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "Cannot open lookup table file: \""<<filepath<<"\" for reading" << std::endl;
        std::cout << "failed with error ("<<errno<<"): " << std::strerror(errno) << std::endl;
        return false;
    }

    std::string line;
    int currentline = 0;
    while (std::getline(file, line)) {
        currentline++;

        // test for empty
        if (line.empty()) {
            if (verbose>=2) { std::cout << "ignoring empty line: "<<currentline<<std::endl; }
            continue;
        }

        // test for comment
        if (is_comment_string(line, '#')) {
            if (verbose>=2) { std::cout << "ignoring comment on line "<<currentline<<std::endl; }
            continue;
        }

        // split line into fields
        std::vector<std::string> fields{};
        tokenize_string(line, fields, " ");
        if (fields.size() < 3 ) {
            std::cout << "LUT spec on line " << currentline << " is invalid. Please check documentation for valid specification" << std::endl;
            return false;
        }

        // parse fields
        lut.points.emplace_back(fields[0], std::stof(fields[1]), std::stof(fields[2]));
        if (verbose>=2) {
            std::cout << "added LUT point on line " << currentline<< ": "<<
                lut.points.back().label<<" "<<lut.points.back().hunits<<" "<<lut.points.back().massdens<< std::endl;
        }
    }

    return 0;
}


int PreProcess::_write_file_version(H5::Group& h5group, uint ftmagic,
    uint ftversionmajor, uint ftversionminor) {
    auto fgroup = h5group.createGroup("filetype");
    H5::DataSpace scalarspace;
    {
        auto att = fgroup.createAttribute("ftmagic", H5::PredType::STD_U8LE, scalarspace);
        att.write(H5::PredType::NATIVE_UINT, &ftmagic);
    }
    {
        auto att = fgroup.createAttribute("ftversionmajor", H5::PredType::STD_U8LE, scalarspace);
        att.write(H5::PredType::NATIVE_UINT, &ftversionmajor);
    }
    {
        auto att = fgroup.createAttribute("ftversionminor", H5::PredType::STD_U8LE, scalarspace);
        att.write(H5::PredType::NATIVE_UINT, &ftversionminor);
    }
    return 1;
}


bool PreProcess::is_comment_string(const std::string& str, const char comment_char) {
        size_t firstchar = str.find_first_not_of(" \t");
        if (firstchar != std::string::npos && str[firstchar] == comment_char) {
            return true;
        }
        return false;
    }

void PreProcess::tokenize_string(const std::string& str,
    std::vector<std::string>& tokens, const std::string& delims) {
    // skip delims at beginning
    size_t lastpos = str.find_first_not_of(delims, 0);
    // Find one after end of first token (next delim position)
    size_t nextpos = str.find_first_of(delims, lastpos);
    while (nextpos != std::string::npos || lastpos != std::string::npos) {
        // add token to vector
        tokens.push_back(str.substr(lastpos, nextpos-lastpos));
        // update positions
        lastpos = str.find_first_not_of(delims, nextpos);
        nextpos = str.find_first_of(delims, lastpos);
    }
}


bool PreProcess::ctdataInitMode1(PreProcess::FloatVolume& ctdata) {
    // get dicom data size
    const std::vector<int>& ctdata_size = PreProcess::getarg<std::vector<int>>("shape");
    ctdata.size = make_uint3(ctdata_size[0], ctdata_size[1], ctdata_size[2]);
    // get voxel dimensions, convert to cm
    float voxelSize_scalar = PreProcess::getarg<float>("voxelSize");
    ctdata.voxsize = make_float3(voxelSize_scalar, voxelSize_scalar, voxelSize_scalar);
    ctdata.start = make_float3(0.f, 0.f, 0.f);

    // prepare data
    std::vector<uint16_t> phantomData_;
    std::vector<float> phantomData;
    const std::string& phantomPath = PreProcess::getarg<std::string>("phantomPath");
    std::ifstream f(phantomPath);
    if (f.is_open()) {
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        size_t numElements = file_size / sizeof(uint16_t);
        if (numElements != ctdata_size[0] * ctdata_size[1] * ctdata_size[2]) {
            std::cerr << "The phantom array has " << numElements << " elements, "
            "inconsistent with the phantom size " << ctdata_size << std::endl;
            return 1;
        }
        phantomData_.resize(numElements);
        f.seekg(0, std::ios::beg);
        f.read((char*)phantomData_.data(), numElements*sizeof(uint16_t));
        f.close();
    } else {
        std::cerr << "Error: Unable to open file: " << phantomPath << std::endl;
        return 1;
    }

    float RescaleSlope = getarg<float>("RescaleSlope");
    float RescaleIntercept = getarg<float>("RescaleIntercept");
    phantomData.resize(phantomData_.size());
    for (int i=0; i<phantomData.size(); i++) {
        float value = phantomData_[i] * RescaleSlope + RescaleIntercept;
        value = std::min(value, DATAMAX);
        value = std::max(value, DATAMIN);
        phantomData[i] = value;
    }

    ctdata.set_data(phantomData.data(), ctdata.nvoxels());
    return 0;
}


bool PreProcess::densityInitMode1(FloatVolume& density,
    const FloatVolume& ctdata) {
    auto ctLUT = CTLUT();
    if (! getarg<bool>("nolut")) {
        const std::string& ctlutFile = getarg<std::string>("ctlutFile");
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
    if (CreateIsoDensity(ctdata, density, &ctLUT)) {
        std::cout << "Failed reading CT data!" << std::endl;
        return 1;
    }
    return 0;
}


bool PreProcess::ROIInitModel1(ROIMaskList& roi_list, const FloatVolume& ctdata) {
    std::vector<std::string> roi_names;
    PreProcess::getROINamesFromJSON(PreProcess::getarg<
        std::string>("structuresFile"), roi_names);
    std::cout << roi_names << std::endl;

    std::string ptv_name = PreProcess::getarg<std::string>("ptv_name");
    std::string bbox_name = PreProcess::getarg<std::string>("bbox_name");

    fs::path maskFolder(getarg<std::string>("maskFolder"));
    size_t numElements = ctdata.size.x * ctdata.size.y * ctdata.size.z;
    for (const std::string& name : roi_names) {
        fs::path file = maskFolder / (name + std::string(".bin"));
        if (fs::is_regular_file(file)) {
            std::vector<uint8_t> roi_mask(numElements);
            std::ifstream f(file.string());
            if (! f.is_open()) {
                std::cerr << "Cannot open file: " << file;
                return 1;
            }
            f.seekg(0, std::ios::end);
            size_t numElementsRef = f.tellg() / sizeof(uint8_t);
            if (numElementsRef != numElements) {
                std::cerr << "The number of elements of the file " << numElementsRef
                    << " is inconsistent with the mask shape: (" << ctdata.size.x 
                    << ", " << ctdata.size.y << ", " << ctdata.size.z << ")" << std::endl;
                return 1;
            }
            f.seekg(0, std::ios::beg);
            f.read((char*)roi_mask.data(), numElements);
            f.close();

            // For simplicity, we include all the mask array without cropping
            ArrayProps roi_bbox;
            roi_bbox.size = ctdata.size;
            roi_bbox.crop_size = ctdata.size;
            roi_bbox.crop_start = make_uint3(0, 0, 0);
            roi_list.push_back(new DenseROIMask(name, roi_mask, roi_bbox));
            std::cout << "Creating ROI Mask: " << name << std::endl;
        }
    }
    return 0;
}