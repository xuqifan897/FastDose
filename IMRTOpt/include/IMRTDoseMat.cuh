#ifndef __IMRTDOSEMAT_CUH__
#define __IMRTDOSEMAT_CUH__

#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "cusparse.h"

#include "IMRTBeamBundle.cuh"
#include "IMRTDoseMatEns.cuh"
#include "IMRTDoseMatEigen.cuh"

#define TIMING true
#define pitchModule 64

#define checkCusparse(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
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
}

#endif