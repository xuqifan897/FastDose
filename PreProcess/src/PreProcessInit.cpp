#include "PreProcessInit.h"
#include "PreProcessArgs.h"
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include "helper_math.h"

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