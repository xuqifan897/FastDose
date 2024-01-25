#ifndef __IMRTDOSEMATEIGEN_CUH__
#define __IMRTDOSEMATEIGEN_CUH__

#include "IMRTDoseMatEns.cuh"
#include "cuda_runtime.h"
#include <Eigen/Sparse>
#define EigenIdxType int64_t

namespace IMRT {
    class MatCSREnsemble;
    class StorageTransparent : public Eigen::internal::CompressedStorage<float, EigenIdxType> {
    public:
        __inline__ EigenIdxType** getIndices() {
            return &(this->m_indices);
        }

        __inline__ float** getValues() {
            return &(this->m_values);
        }

        __inline__ Eigen::Index& getSize() {
            return this->m_size;
        }

        __inline__ Eigen::Index& getAllocatedSize() {
            return this->m_allocatedSize;
        }
    };

    class MatCSR_Eigen : public Eigen::SparseMatrix<float, Eigen::RowMajor, EigenIdxType> {
    public:
        __inline__ Index& getRows() {
            if (this->IsRowMajor) {
                return this->m_outerSize;
            } else {
                return this->m_innerSize;
            }
        }

        __inline__ Index& getCols() {
            if (this->IsRowMajor) {
                return this->m_innerSize;
            } else {
                return this->m_outerSize;
            }
        }

        __inline__ EigenIdxType** getOffset() {
            return & this->m_outerIndex;
        }

        __inline__ const EigenIdxType* getIndices() const {
            return this->m_data.indexPtr();
        }

        __inline__ const float* getValues() const {
            return this->m_data.valuePtr();
        }

        __inline__ Eigen::Index getNnz() const {
            return this->m_data.size();
        }

        bool customInit(
            EigenIdxType nRows, EigenIdxType nCols, EigenIdxType nnz,
            EigenIdxType* offsets, EigenIdxType* columns, float* values);
        
        bool fromEnsemble(MatCSREnsemble& source);
        bool fromfile(const std::string& resultFolder, size_t numCols);
        MatCSR_Eigen transpose();
    };

    bool MatOARSlicing(const MatCSR_Eigen& matrixT, MatCSR_Eigen& A,
        MatCSR_Eigen& AT, const std::vector<StructInfo>& structs);
}

IMRT::MatCSR_Eigen operator*(const IMRT::MatCSR_Eigen& a, const IMRT::MatCSR_Eigen& b);

#endif