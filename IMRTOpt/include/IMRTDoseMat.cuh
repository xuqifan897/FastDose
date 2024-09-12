#ifndef __IMRTDOSEMAT_CUH__
#define __IMRTDOSEMAT_CUH__

#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "cusparse.h"

#include "IMRTBeamBundle.cuh"
#include "IMRTDoseMatEns.cuh"
#include "IMRTDoseMatEigen.cuh"

#define TIMING true
#define BeamsPerBatch 100
#define pitchModule 64

#define checkCusparse(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at %s:%d with error: %s (%d)\n",         \
               __FILE__, __LINE__, cusparseGetErrorString(status), status);   \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}

namespace IMRT {
    class MatCSREnsemble;

    class MatCSR64 {
    public:
        MatCSR64() : matA(nullptr), d_csr_offsets(nullptr),
            d_csr_columns(nullptr), d_csr_values(nullptr),
            d_buffer_spmv(nullptr), nnz(0) {}

        ~MatCSR64() {
            if (this->matA != nullptr) {
                checkCusparse(cusparseDestroySpMat(this->matA));
                this->matA = nullptr;
            }
            if (this->d_csr_offsets != nullptr) {
                checkCudaErrors(cudaFree(this->d_csr_offsets));
                this->d_csr_offsets = nullptr;
            }
            if (this->d_csr_columns != nullptr) {
                checkCudaErrors(cudaFree(this->d_csr_columns));
                this->d_csr_columns = nullptr;
            }
            if (this->d_csr_values != nullptr) {
                checkCudaErrors(cudaFree(this->d_csr_values));
                this->d_csr_values = nullptr;
            }
            if (this->d_buffer_spmv != nullptr) {
                checkCudaErrors(cudaFree(this->d_buffer_spmv));
                this->d_buffer_spmv = nullptr;
            }
        }

        bool dense2sparse(float* d_dense, size_t num_rows, size_t num_cols, size_t ld);
        bool fuseEnsemble(MatCSREnsemble& matEns);

        cusparseSpMatDescr_t matA;
        size_t* d_csr_offsets;
        size_t* d_csr_columns;
        float* d_csr_values;
        void* d_buffer_spmv;
        size_t numRows;
        size_t numCols;
        int64_t nnz;
    };

    bool DoseMatConstruction(
        std::vector<BeamBundle>& beam_bundles,
        fastdose::DENSITY_d& density_d,
        fastdose::SPECTRUM_h& spectrum_h,
        fastdose::KERNEL_h& kernel_h,
        MatCSREnsemble** matEns,
        cudaStream_t stream=0
    );


    bool MatOARSlicing(
        const MatCSR64& fullMat, MatCSR64& sliceMat, MatCSR64& sliceMatT,
        const std::vector<StructInfo>& structs);

    
    class MatCSR32 {
    public:
        MatCSR32() : matA(nullptr), d_csr_offsets(nullptr),
            d_csr_columns(nullptr), d_csr_values(nullptr),
            d_buffer_spmv(nullptr), nnz(0) {}

        ~MatCSR32() {
            if (this->matA != nullptr) {
                checkCusparse(cusparseDestroySpMat(matA));
                this->matA = nullptr;
            }
            if (this->d_csr_offsets != nullptr) {
                checkCudaErrors(cudaFree(this->d_csr_offsets));
                this->d_csr_offsets = nullptr;
            }
            if (this->d_csr_columns != nullptr) {
                checkCudaErrors(cudaFree(this->d_csr_columns));
                this->d_csr_columns = nullptr;
            }
            if (this->d_csr_values != nullptr) {
                checkCudaErrors(cudaFree(this->d_csr_values));
                this->d_csr_values = nullptr;
            }
            if (this->d_buffer_spmv != nullptr) {
                checkCudaErrors(cudaFree(this->d_buffer_spmv));
                this->d_buffer_spmv = nullptr;
            }
        }

        bool fuseEnsemble(MatCSREnsemble& matEns);

        cusparseSpMatDescr_t matA;
        int* d_csr_offsets;
        int* d_csr_columns;
        float* d_csr_values;
        void* d_buffer_spmv;
        int numRows;
        int numCols;
        int64_t nnz;
    };

    bool MatCSR32_fromfile(const std::string resultFolder,
        size_t numColsPerMat, const std::vector<StructInfo>& structs);
    bool getOARFilter(MatCSR32& matFilter, MatCSR32& matFilterT,
        const std::vector<StructInfo>& structs, size_t nVoxels);
    bool OARFiltering(std::vector<MatCSR32>& OARMatricesT, const std::vector<MatCSR32>& matricesT,
        const MatCSR32& matFilter, const MatCSR32& matFilterT,
        int** d_bufferOffsets, int** d_bufferColumns, float** d_bufferValues);
    bool OARFiltering(const std::vector<std::string>& doseMatFolders, const std::vector<StructInfo>& structs,
        std::vector<MatCSR_Eigen>& MatricesT_full, std::vector<MatCSR_Eigen>& VOIMatrices,
        std::vector<MatCSR_Eigen>& VOIMatricesT, Weights_h& weights,
        const std::string& ptv, const std::vector<float>* referenceDose);
}

#endif