#ifndef __PREPROCESSHELPER_H__
#define __PREPROCESSHELPER_H__

#include <string>
#include "rtimages.h"

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

#ifdef __CUDA_ARCH__
    #define CUDEV_FXN __host__ __device__
#else
    #define CUDEV_FXN
#endif

CUDEV_FXN inline int3 float2int_floor(const float3& a) {
    return int3 { (int)floorf(a.x), (int)floorf(a.y), (int)floorf(a.z) };
}
CUDEV_FXN inline uint3 float2uint_floor(const float3& a) {
    return uint3 { (uint)floorf(a.x), (uint)floorf(a.y), (uint)floorf(a.z) };
}
CUDEV_FXN inline int3 float2int_ceil(const float3& a) {
    return int3 { (int)ceilf(a.x), (int)ceilf(a.y), (int)ceilf(a.z) };
}
CUDEV_FXN inline uint3 float2uint_ceil(const float3& a) {
    return uint3 { (uint)ceilf(a.x), (uint)ceilf(a.y), (uint)ceilf(a.z) };
}

// NARROWING OPS
CUDEV_FXN inline unsigned int product(const uint3& a) { return a.x * a.y * a.z; }
CUDEV_FXN inline int product(const int3& a) { return a.x * a.y * a.z; }

namespace PreProcess {
    template <class T>
    class Volume;
    typedef Volume<float>   FloatVolume;
    typedef Volume<short>   ShortVolume;
    typedef Volume<char>    CharVolume;
    typedef Volume<uint8_t> BoolVolume;

    struct CTLUT;

    std::string get_username();
    bool CreateIsoDensity(const FloatVolume& source, FloatVolume& target,
        CTLUT* ctlut, bool verbose=false);

    __global__ void
    cudaMakeIsotropicWithLUT( float *iso, float3 voxelSize, float iso_voxel, uint3 iso_size, 
        float* lut_hunits, float* lut_massdens, int nlut, cudaTextureObject_t texObj);

    // Describes sub-volume for iterating or calculating
    struct ArrayProps {
        uint3 size;                   // size of full dicom volume
        uint3 crop_size;              // size of calc_bbox
        uint3 crop_start;             // start indices of calc_bbox relative to original dicom volume size
        // FrameOfReference frame {};       // NOT IN USE YET - MIGRATE size to frame.size
        CUDEV_FXN uint3 crop_end() { return crop_start + crop_size; }
        uint nvoxels() { return crop_size.x * crop_size.y * crop_size.z; }

        static int _readFromHDF5(ArrayProps& props, H5::Group& h5group) {
            // tuple dataspaces/datatypes
            hsize_t tuple3_dims[] = { 3 };
            H5::ArrayType tuple3_native_t(H5::PredType::NATIVE_UINT, 1, tuple3_dims);

            // read attributes
            {
                uint temp[3];
                auto att = h5group.openAttribute("size");
                att.read(tuple3_native_t, temp);
                ARR3VECT(props.size, temp);
            }
            {
                uint temp[3];
                auto att = h5group.openAttribute("crop_size");
                att.read(tuple3_native_t, temp);
                ARR3VECT(props.crop_size, temp);
            }
            {
                uint temp[3];
                auto att = h5group.openAttribute("crop_start");
                att.read(tuple3_native_t, temp);
                ARR3VECT(props.crop_start, temp);
            }
            return true;
        }
        int _writeToHDF5(H5::Group& h5group) const {
            // tuple dataspaces/datatypes
            H5::DataSpace scalarspace {};
            hsize_t tuple3_dims[] = { 3 };
            H5::DataSpace tuple3(1, tuple3_dims);
            H5::ArrayType tuple3_native_t(H5::PredType::NATIVE_UINT, 1, tuple3_dims);
            H5::ArrayType tuple3_t(H5::PredType::STD_U16LE, 1, tuple3_dims);

            // write attributes
            {
                uint temp[3];
                VECT3ARR(temp, size)
                auto att = h5group.createAttribute("size", tuple3_t, scalarspace);
                att.write(tuple3_native_t, temp);
            }
            {
                uint temp[3];
                VECT3ARR(temp, crop_size)
                auto att = h5group.createAttribute("crop_size", tuple3_t, scalarspace);
                att.write(tuple3_native_t, temp);
            }
            {
                uint temp[3];
                VECT3ARR(temp, crop_start)
                auto att = h5group.createAttribute("crop_start", tuple3_t, scalarspace);
                att.write(tuple3_native_t, temp);
            }
            return true;
        }
    };

    // template<typename T>
    // int write_debug_data(T *mat, uint3 count, const char* filename, bool verbose=false) {
    //     return write_debug_data<T>(mat, make_int3(count), filename, verbose);
    // }
    // template<typename T>
    // int write_binary_data(T *mat, uint3 count, const char* filename, bool verbose=false) {
    //     return write_binary_data<T>(mat, make_int3(count), filename, verbose);
    // }
    // template<typename T>
    // int write_debug_data(T *mat, int3 count, const char* filename, bool verbose=false) {
    //     const std::string& inputFolder = getarg<std::string>("inputFolder");
    //     char binfile[1024];
    //     sprintf(binfile,"%s/%s.raw", inputFolder, filename);
    //     return write_binary_data<T>(mat, count, binfile, verbose);
    // }

    template<typename T>
    int write_binary_data(T *mat, int3 count, const char* filename, bool verbose) {
        // TODO: convert to c++ ofstream with metadata header/footer
        FILE *binout;

        if (verbose) {
            std::cout << "binary data file name is \"" << filename << "\" with size: ("<<count.x << ", "<<count.y<<", "<<count.z<<")"<<std::endl;
        }

        if ( (binout = fopen(filename,"wb")) == NULL) {
            printf("write_binary_data() failed with error (%d): %s\n", errno, std::strerror(errno));
            return(-1); }

        size_t sizer = count.x * count.y * count.z;

        unsigned int entries = fwrite( (const void*)mat, sizeof(float), sizer, binout);

        if ( entries != sizer){
            printf("  Binary file has unexpected size! (%d / %lu)\n", entries, sizer);
            fclose(binout);
            return -2;
        }

        fclose(binout);
        return(1);
    }
    template<typename T>
    int write_debug_data(T *mat, uint3 count, const char* filename, bool verbose=false) {
        return write_binary_data<T>(mat, make_int3(count), filename, verbose);
    }
}

#endif