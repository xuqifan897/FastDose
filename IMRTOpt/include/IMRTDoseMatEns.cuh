#ifndef __IMRTDOSEMATENS_CUH__
#define __IMRTDOSEMATENS_CUH__
#include <string>
#include <vector>
#include "IMRTDoseMat.cuh"
#include "IMRTDoseMatEigen.cuh"
#include "IMRTDebug.cuh"

namespace IMRT {
    class MatCSR64;
    class MatCSR_Eigen;
    class MatCSREnsemble {
    public:
        MatCSREnsemble(
            const std::vector<size_t> numRowsPerMat_,
            size_t numColsPerMat_,
            size_t estBufferSize  // estimated buffer size, in elements, not byte
        );

        MatCSREnsemble(size_t numColsPerMat_):
            numColsPerMat(numColsPerMat_),
            d_offsetsBuffer(nullptr),
            d_columnsBuffer(nullptr),
            d_valuesBuffer(nullptr),
            d_constructBuffer(nullptr)
        {}

        ~MatCSREnsemble();

        bool addMat(float* d_dense, size_t numRows, size_t numCols);
        bool tofile(const std::string& resultFolder);
        bool fromfile(const std::string& resultFolder);

        friend class MatCSR64;
        friend MatCSR_Eigen;
        friend bool sparseValidation(const MatCSREnsemble* matEns);
        friend bool conversionValidation(
            const MatCSR64& mat, const MatCSREnsemble& matEns);

    private:
        std::vector<cusparseSpMatDescr_t> matA_array;
        std::vector<size_t> NonZeroElements;
        std::vector<size_t> CumuNonZeroElements;

        std::vector<size_t> numRowsPerMat;
        std::vector<size_t> CumuNumRowsPerMat;
        // the starting index of d_offsetsBuffer of each matrix
        std::vector<size_t> OffsetBufferIdx;

        size_t numMatrices;
        size_t numColsPerMat;
        size_t* d_offsetsBuffer;

        size_t bufferSize;  // in number of elements, not bytes
        size_t* d_columnsBuffer;
        float* d_valuesBuffer;

        size_t constructBufferSize;  // in bytes
        float* d_constructBuffer;
    };
}

#endif