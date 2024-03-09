#ifndef __RTIMAGE_H__
#define __RTIMAGE_H__

#include <string>
#include <array>
#include <vector>

#include <cuda_runtime.h>
#include <helper_math.h>
#include "dcmtk/dcmdata/dctk.h"
#include "H5Cpp.h"

#include "PreProcessHelper.h"
#include "rtstruct.h"

#define RTIMAGE_SOP_CLASS_UID "1.2.840.10008.5.1.4.1.1.481.1"
#define CTIMAGE_SOP_CLASS_UID "1.2.840.10008.5.1.4.1.1.2"
#define MRIMAGE_SOP_CLASS_UID "1.2.840.10008.5.1.4.1.1.4"

#ifndef UINT16_MAX
#define UINT16_MAX 65535
#endif

#ifndef CONVERSION_VEC_ARRAY
#define CONVERSION_VEC_ARRAY
// Convert cuda vector types to c-style arrays
#define VECT2ARR(a, v) a[0] = v.x; a[1] = v.y;
#define VECT3ARR(a, v) a[0] = v.x; a[1] = v.y; a[2] = v.z;
// Convert c-style array to cuda vector types
#define ARR2VECT(v, a) v.x = a[0]; v.y = a[1];
#define ARR3VECT(v, a) v.x = a[0]; v.y = a[1]; v.z = a[2];

// std::cout formatting
#define FORMAT_3VEC(v) "("<<v.x<<", "<<v.y<<", "<<v.z<<")"
#define FORMAT_2VEC(v) "("<<v.x<<", "<<v.y<<")"
#endif

namespace PreProcess {
    template<typename T>
    int write_debug_data(T *mat, uint3 count, const char* filename, bool verbose=false) {
        return write_debug_data<T>(mat, make_int3(count), filename, verbose);
    }
    template<typename T>
    int write_binary_data(T *mat, uint3 count, const char* filename, bool verbose=false) {
        return write_binary_data<T>(mat, make_int3(count), filename, verbose);
    }
    template<typename T>
    int write_debug_data(T *mat, int3 count, const char* filename, bool verbose=false) {
        char binfile[1024];
        sprintf(binfile,"%s/%s.raw", get_username().c_str(),filename);
        return write_binary_data<T>(mat, count, binfile, verbose);
    }

    class RTImage {
    public:
        RTImage() {}
        RTImage(const std::string& dname, bool verbose=false) : dicom_dir{dname} {
            loadDicomInfo(verbose);
            loadRTImageData(verbose);
        }
        ~RTImage();

        class SLICE_DATA {
        public:
            std::string filename;
            std::string sop_instance_uid;
            std::string reference_frame_uid;
            int instance_number;
            float slice_location;
            std::string patient_position;
            float3 image_position_patient;
            std::array<float, 6> image_orientation_patient;

            SLICE_DATA() {}

            // copy constructor
            SLICE_DATA(const SLICE_DATA& copyfrom) {
                filename = copyfrom.filename;
                sop_instance_uid = copyfrom.sop_instance_uid;
                reference_frame_uid = copyfrom.reference_frame_uid;
                instance_number = copyfrom.instance_number;
                slice_location = copyfrom.slice_location;
                image_position_patient = copyfrom.image_position_patient;
                patient_position = std::string(copyfrom.patient_position);
                image_orientation_patient = copyfrom.image_orientation_patient;
            }
        };

        bool loadDicomInfo(bool verbose=false);
        int  loadRTImageData(bool verbose=false, bool flipXY=false);
        void saveRTImageData (const std::string& outpath, bool anonymize_switch );
        int  saveRTImageData (const std::string& outpath, float *newData, bool anonymize_switch );
        bool importSOPClassUID(const std::string& fname);
        void importPatientPosition(unsigned int i);
        void importImagePositionPatient( unsigned int i );
        void importInstanceNumber( unsigned int i );
        void importPatientInfo();
        void anonymize( DcmDataset *dataset );

        /* GETTERS */
        std::string  getDicomDirectory()                          { return dicom_dir; };
        float        getArrayVoxel(int i, int j, int k)           { return data_array[i + data_size.x*(j + data_size.y*k)]; };
        unsigned int getImageCount()                              { return image_count; };
        int3         getDataSize()                                { return data_size; };
        float3       getVoxelSize()                               { return voxel_size; };
        float        getRescaleSlope()                            { return rescale_slope; };
        float        getRescaleIntercept()                        { return rescale_intercept; };
        float        getDataMin()                                 { return data_min; };
        float        getDataMax()                                 { return data_max; };
        float*       getDataArray()                               { return data_array; };
        float        getWindowCenter()                            { return window_center; };
        float        getWindowWidth()                             { return window_width; };
        int          getSliceInstanceNumber(unsigned int i)       { return slice[i].instance_number; };
        std::string  getSlicePatientPosition(unsigned int i)      { return slice[i].patient_position; }
        float3       getSliceImagePositionPatient(unsigned int i) { return slice[i].image_position_patient; };
        std::array<float, 6> getSliceImageOrientationPatient(unsigned int i) { return slice[i].image_orientation_patient; }
        std::string  getSliceSOPInstanceUID(unsigned int i)       { return slice[i].sop_instance_uid; };
        std::string  getSliceReferenceFrameUID(unsigned int i)    { return slice[i].reference_frame_uid; };
        std::string  getSliceFilename(unsigned int i)             { return slice[i].filename; };

        /* SETTERS */
        void setDicomDirectory(const std::string& dname)                                   { dicom_dir = dname; }
        void setSliceSOPInstanceUID( unsigned int i, const std::string& uid)               { slice[i].sop_instance_uid = uid; }
        void setSliceFilename(unsigned int i, const std::string& fname)                    { slice[i].filename = fname; }
        void setArrayVoxel(int x, int y, int z, float v)                                   { data_array[x + data_size.x*(y + data_size.y*z)] = v; };
        void setDataSize(int x, int y, int z)                                              { data_size = int3{x,y,z}; }
        void setVoxelSize(float v_x, float v_y, float v_z)                                 { voxel_size = float3{v_x, v_y, v_z}; }
        void setDataOrigin(float v_x, float v_y, float v_z)                                { data_origin = float3{v_x, v_y, v_z}; }
        void setRescaleSlope(float v)                                                      { rescale_slope = v; };
        void setRescaleIntercept(float v)                                                  { rescale_intercept = v; };
        void setDataMin(float v)                                                           { data_min = v; };
        void setDataMax(float v)                                                           { data_max = v; };
        void setWindowCenter(float v)                                                      { window_center = v; };
        void setWindowWidth(float v)                                                       { window_width = v; };
        void setSliceInstanceNumber(unsigned int i, int n)                                 { slice[i].instance_number = n; };
        void setSliceImagePositionPatient(unsigned int i, float v_x, float v_y, float v_z) { slice[i].image_position_patient = float3{v_x, v_y, v_z}; }

        // Type of image
        enum class SOPTYPE { RTIMAGE, CT, MR };
        SOPTYPE sop_type;
        std::string getSOPTypeName() { return getSOPTypeName(sop_type); }
        std::string getSOPTypeName(SOPTYPE t) {
            switch(t) {
                case SOPTYPE::RTIMAGE  :  return "RTIMAGE";
                case SOPTYPE::CT       :  return "CT";
                case SOPTYPE::MR       :  return "MR";
                default                :  return "unknown";
            }
        }

        // data ordering - see https://public.kitware.com/IGSTKWIKI/index.php/DICOM_data_orientation
        enum class DATAORDER { LPS, RAS };
        DATAORDER data_order;

    protected:
        std::string dicom_dir;
        std::string dicom_date;
        std::string pt_series_description;
        std::string pt_name;
        std::string pt_id;
        std::string pt_study_id;
        std::string pt_study_instance_uid;
        std::string pt_series_instance_uid;

        unsigned int image_count;
        SLICE_DATA *slice;

        int3 data_size;
        float3 voxel_size;
        float3 data_origin;
        float slice_thickness;

        std::string orient{};
        float3 orient_x;
        float3 orient_y;

        float window_center;
        float window_width;
        float rescale_slope;
        float rescale_intercept;

        float data_min;
        float data_max;
        float *data_array;
    };

    struct FrameOfReference {
        uint3  size;    // data array dimensions
        float3 start;   // scanner coords of first voxel in data array [unit: mm]
        float3 spacing; // voxel size in [unit: mm]
        uint nvoxels() const { return size.x*size.y*size.z; }
        float3 end() const { return make_float3(start.x + spacing.x*size.x, start.y + spacing.y*size.y, start.z + spacing.z*size.z); }
    };

    typedef unsigned int uint;
    int _write_file_version(H5::Group&, uint, uint, uint);

    template <typename T>
    class Volume {
    public:
        Volume() {}
        // copy constructor
        Volume(const FrameOfReference& frame) :
            start{frame.start}, size{frame.size}, voxsize{frame.spacing}, _vect{std::vector<T>(frame.nvoxels())}
            {}
        Volume(T* data, const FrameOfReference& frame) :
            start{frame.start}, size{frame.size}, voxsize{frame.spacing}, _vect{std::vector<T>(data, data+frame.nvoxels())}
            {}

        float3 start;                // coords of first voxel in scanner coord system (GCS)
        float3 voxsize;              // voxel dimensions in mm
        uint3  size;                 // volume shape (units of integer voxels)
        std::vector<T> _vect;        // linear array flattened in C-Major order (Depth->Row->Column)

        T* data()  {return _vect.data();};
        const T* data() const {return _vect.data();};
        // copy count elements from ptr to vector
        void set_data(T* ptr, uint count) { _vect = std::vector<T>(ptr, ptr+count); }
        // initialize to size=count with zeros
        void set_data(uint count) { _vect = std::vector<T>(count); }

        inline uint nvoxels() { return size.x * size.y * size.z; }
        inline uint mem_size() { return nvoxels() * sizeof(T); }

        const FrameOfReference get_frame() const {
        return FrameOfReference{size, start, voxsize};
        }

        // OP OVERLOADS
        T& operator [](uint idx) { return _vect[idx]; }
        const T& operator [](uint idx) const { return _vect[idx]; }
        T& operator [](int idx) { return _vect[idx]; }
        const T& operator [](int idx) const { return _vect[idx]; }
        T& operator [](uint3 coords) { return at(coords); }
        const T& operator [](uint3 coords) const { return at(coords); }
        T& operator [](int3 coords) { return at(coords); }
        const T& operator [](int3 coords) const { return at(coords); }
        T& at(uint idx) { return _vect.at(idx); }
        const T& at(uint idx) const { return _vect.at(idx); }
        T& at(uint x, uint y, uint z) { return _vect.at(x + size.x*(y + size.y*z)); }
        const T& at(uint x, uint y, uint z) const { return _vect.at(x + size.x*(y + size.y*z)); }
        T& at(uint3 coords) { return at(coords.x, coords.y, coords.z); }
        const T& at(uint3 coords) const { return at(coords.x, coords.y, coords.z); }
        T& at(int3 coords) { return at(coords.x, coords.y, coords.z); }
        const T& at(int3 coords) const { return at(coords.x, coords.y, coords.z); }
        bool check_bounds(uint idx) { return (idx < nvoxels()); }
        bool check_bounds(uint x, uint y, uint z) { return (x < size.x && y < size.y && z < size.z); }
        bool check_bounds(uint3 coords) { return (coords.x < size.x && coords.y < size.y && coords.z < size.z); }
        bool check_bounds(int3 coords) { return (coords.x>=0 && coords.y>=0 && coords.z>=0 &&
                coords.x < size.x && coords.y < size.y && coords.z < size.z); }

        // FILE I/O
        static int _readFromHDF5(Volume<T>& vol, H5::Group& h5group, const std::string& dset_name="data") {
            // read data array
            auto dset = h5group.openDataSet(dset_name);
            _readTypedH5Dataset(vol._vect, vol.size, dset);

            { // read start coords
                auto att = dset.openAttribute("dicom_start_cm");
                float temp[3];
                att.read(H5::PredType::NATIVE_FLOAT, &temp);
                ARR3VECT(vol.start, temp);
            } { // read voxelsize
                auto att = dset.openAttribute("voxel_size_cm");
                float temp[3];
                att.read(H5::PredType::NATIVE_FLOAT, &temp);
                ARR3VECT(vol.voxsize, temp);
            }
            return true;
        }
        static int readFromFile(Volume<T>& vol, const std::string& infile, const std::string& dset_name="data", int verbose=false) {
            try {
                vol = Volume<T>{};
                auto h5file = H5::H5File(infile, H5F_ACC_RDONLY);
                H5::Group rootgroup = h5file.openGroup("/");
                if (!Volume<T>::_readFromHDF5(vol, rootgroup, dset_name)) {
                    if (verbose) { std::cerr << "Failed to read Volume from \""<<infile<<"\""<<std::endl; }
                    return false;
                }
            } catch (H5::FileIException &file_exists_error) {
                if (verbose) { std::cerr << "Failed to read Volume from \""<<infile<<"\""<<std::endl; }
                return false;
            }
            return true;
        }

        static int _writeToHDF5(H5::Group& h5group, const T* data, const FrameOfReference& frame, const std::string& dset_name) {

            // write dose float-volume to dataset
            H5::DataSet dset = _createTypedH5Dataset(h5group, data, frame, dset_name);

            // add attributes
            {
                hsize_t dims[] = {3};
                auto dspace = H5::DataSpace(1, dims);
                auto att = dset.createAttribute("dicom_start_cm", H5::PredType::IEEE_F32LE, dspace);
                float temp[3];
                VECT3ARR(temp, frame.start);
                att.write(H5::PredType::NATIVE_FLOAT, &temp);
            } {
                hsize_t dims[] = {3};
                auto dspace = H5::DataSpace(1, dims);
                auto att = dset.createAttribute("voxel_size_cm", H5::PredType::IEEE_F32LE, dspace);
                float temp[3];
                VECT3ARR(temp, frame.spacing);
                att.write(H5::PredType::NATIVE_FLOAT, &temp);
            }
            return true;
        }
        int writeToFile(const std::string& outfile, const std::string& dset_name="volume") {
            FrameOfReference frame = { size, start, voxsize };
            Volume<T>::writeToFile(outfile, _vect.data(), frame, dset_name);
            return 0;
        }
        static int writeToFile(const std::string& outfile, T* data, FrameOfReference& frame, const std::string& dset_name="volume") {
            auto h5file = H5::H5File(outfile, H5F_ACC_TRUNC);
            auto rootgroup = h5file.openGroup("/");
            _write_file_version(rootgroup, FTMAGIC, FTVERSIONMAJOR, FTVERSIONMINOR);
            return _writeToHDF5(rootgroup, data, frame, dset_name);
        }
        int writeToRawFile(const std::string& outfile, bool verbose=false) {
            return write_binary_data<T>(this->data(), size, outfile.c_str(), verbose);
        }

    protected:
        static const uint FTMAGIC = 0x2A;
        static const uint FTVERSIONMAJOR = 1;
        static const uint FTVERSIONMINOR = 0;

        static H5::DataSet typedH5DatasetFactory(H5::Group& h5group, const T* data, const FrameOfReference& frame, const std::string& dset_name, const H5::PredType& fileDataType, const H5::PredType& memDataType) {
            hsize_t dims_mem[] = {frame.nvoxels()};
            hsize_t dims_file[] = {frame.size.z, frame.size.y, frame.size.x};
            auto dspace_mem = H5::DataSpace(1, dims_mem);
            auto dspace_file = H5::DataSpace(3, dims_file);
            H5::DataSet dset = h5group.createDataSet(dset_name, fileDataType, dspace_file);
            dset.write(data, memDataType, dspace_mem, dspace_file);
            return dset;
        }

        static void typedH5DatasetLoader(std::vector<T>& outvect, uint3& outsize, H5::DataSet& h5dset, const H5::PredType& memDataType) {
            auto file_space = h5dset.getSpace();
            hsize_t N = file_space.getSimpleExtentNpoints();
            std::unique_ptr<T[]> temp(new T[N]);
            hsize_t dims[] = {N};
            auto mem_space = H5::DataSpace(1, dims);
            h5dset.read(temp.get(), memDataType, mem_space, file_space);
            outvect.clear();
            outvect.assign(&temp[0], &temp[N]);
            hsize_t file_dims[3];
            file_space.getSimpleExtentDims(file_dims);
            outsize.x = file_dims[2];
            outsize.y = file_dims[1];
            outsize.z = file_dims[0];
        }

        // These need to be specialized for each templated Volume type that will be used
        static H5::DataSet _createTypedH5Dataset(H5::Group& h5group, const T* data, const FrameOfReference& frame, const std::string& dset_name);
        static void _readTypedH5Dataset(std::vector<T>&, uint3&, H5::DataSet&);
    };

    // represents a single point in the lookup table
    struct LUTPOINT {
        LUTPOINT(std::string label, float hunits, float massdens, float reledens=0)
            : label(label), hunits(hunits), massdens(massdens), reledens(reledens) {}

        std::string label;
        float hunits;     // [HU]
        float massdens;   // [g/cm^3]
        float reledens;   // electron density relative to water
    };

    // collection of LUTPOINTS with convenience methods and interpolation
    struct CTLUT {
    public:
        enum class INTERPTYPE {
            LINEAR,
        };

        CTLUT(INTERPTYPE interp_style=INTERPTYPE::LINEAR) : label(""), interp_style(interp_style) {};
        CTLUT(std::string label, INTERPTYPE interp_style=INTERPTYPE::LINEAR)
            : label(label), interp_style(interp_style) {}

        std::string label;
        INTERPTYPE interp_style;

        std::vector<LUTPOINT> points;
        void sort();

        // friend std::ostream& operator<<(std::ostream& os, const CTLUT& ctlut);
    };

    bool load_lookup_table(CTLUT& lut, std::string filepath, int verbose=0);
    bool is_comment_string(const std::string& str, const char comment_char);
    void tokenize_string(const std::string& str,
        std::vector<std::string>& tokens, const std::string& delims);

    bool closeto(float a, float b, float tolerance=1e-6f);
    ArrayProps getROIExtents(const StructureSet& roi,
        const FrameOfReference& frame, bool verbose=false);
}

std::ostream& operator<<(std::ostream& os, const PreProcess::CTLUT& ctlut);
std::ostream& operator<<(std::ostream& out, const PreProcess::FrameOfReference& frame);

#endif // __RTIMAGE_H__