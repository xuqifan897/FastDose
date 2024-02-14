#ifndef __IMRTOPTIMIZE_VAR_H__
#define __IMRTOPTIMIZE_VAR_H__
#include "IMRTDoseMatEigen.cuh"
#include <vector>

namespace IMRT {
    bool assemble_col_block_meta(
        size_t& numRows, size_t& numCols, size_t& total_nnz, size_t& numMatrices,
        std::vector<size_t>& cumuNnz, const std::vector<uint8_t>& flags,
        const std::vector<MatCSR_Eigen>& reservior_h);
}

#endif