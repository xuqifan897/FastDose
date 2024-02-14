#ifndef __IMRTOPTIMIZE_VAR_CUH__
#define __IMRTOPTIMIZE_VAR_CUH__
#include "IMRTDoseMat.cuh"
#include "IMRTDoseMatEigen.cuh"

namespace IMRT {
    // estimate the size of the total matrix.
    size_t sizeEstimate(
        const std::vector<MatCSR_Eigen>& VOIMatrices,
        const std::vector<MatCSR_Eigen>& VOIMatricesT,
        const std::vector<MatCSR_Eigen>& SpFluenceGrad,
        const std::vector<MatCSR_Eigen>& SpFluenceGradT);

    class MatReservior {
        // this class stores the data for matrices
    public:
        // load data from CPU to GPU
        bool load(const std::vector<MatCSR_Eigen>& source);
        // assemble the reservior to a single matrix
        bool assemble_row_block(MatCSR64& target, const std::vector<uint8_t>& flags) const;
        bool assemble_col_block(MatCSR64& target, const std::vector<uint8_t>& flags) const;
        std::vector<MatCSR64> reservior;
    };

    __global__ void
    d_assembly_row_block(size_t* d_csr_offsets,
        size_t* cumu_row, size_t* cumu_nnz, size_t numMatrices);

    bool MatReservior_dev(
        const std::vector<MatCSR_Eigen>& VOIMatrices,
        const std::vector<MatCSR_Eigen>& VOIMatricesT,
        const std::vector<MatCSR_Eigen>& SpFluenceGrad,
        const std::vector<MatCSR_Eigen>& SpFluenceGradT);

    bool MatReservior_dev_col(
        const std::vector<MatCSR_Eigen>& VOIMatrices,
        const std::vector<MatCSR_Eigen>& VOIMatricesT,
        const std::vector<MatCSR_Eigen>& SpFluenceGrad,
        const std::vector<MatCSR_Eigen>& SpFluenceGradT);
}

#endif