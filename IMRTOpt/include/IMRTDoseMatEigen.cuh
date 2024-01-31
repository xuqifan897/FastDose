#ifndef __IMRTDOSEMATEIGEN_H__
#define __IMRTDOSEMATEIGEN_H__

// #include "IMRTDoseMatEns.cuh"
// #include "cuda_runtime.h"
#include <Eigen/Sparse>
#define EigenIdxType int64_t
#define slicingTiming true
#define ParallelGroupSize 4096

namespace IMRT {
    class MatCSREnsemble;
    class StructInfo;
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

        __inline__ const Index& getRows() const {
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

        __inline__ const Index& getCols() const {
            if (this->IsRowMajor) {
                return this->m_innerSize;
            } else {
                return this->m_outerSize;
            }
        }

        __inline__ EigenIdxType** getOffset() {
            return & this->m_outerIndex;
        }

        __inline__ const EigenIdxType* getOffset() const {
            return this->m_outerIndex;
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
        MatCSR_Eigen transpose() const;
    };

    bool parallelSpGEMM(const std::string& resultFolder, const MatCSR_Eigen& filter,
        const MatCSR_Eigen& filterT, std::vector<MatCSR_Eigen>& OARMatrices,
        std::vector<MatCSR_Eigen>& OARMatricesT);
    
    bool parallelMatCoalease(MatCSR_Eigen& OARmat, MatCSR_Eigen& OARmatT,
        const std::vector<MatCSR_Eigen>& OARMatrices,
        const std::vector<MatCSR_Eigen>& OARMatricesT);

    bool MatOARSlicing(const MatCSR_Eigen& matrixT, MatCSR_Eigen& A,
        MatCSR_Eigen& AT, const std::vector<StructInfo>& structs);

    bool OARFiltering(const std::string& resultFolder,
        const std::vector<StructInfo>& structs);

    bool getStructFilter(MatCSR_Eigen& filter, MatCSR_Eigen& filterT,
        const std::vector<StructInfo>& structs);
    
    // size in bytes
    bool readBlockParallel(const std::string& filename, void** pointer, EigenIdxType* size);

    // start, end in bytes
    void readBlockParallelFunc(const std::string& filename,
        char* buffer, size_t start, size_t end);

    bool test_parallelSpGEMM(const std::vector<MatCSR_Eigen>& OARMatrices,
        const std::vector<MatCSR_Eigen>& OARMatricesT,
        const std::vector<MatCSR_Eigen>& matricesT,
        const MatCSR_Eigen& filter);

    bool test_OARMat_OARMatT(const MatCSR_Eigen& OARMat, const MatCSR_Eigen& OARMatT);
}

IMRT::MatCSR_Eigen operator*(const IMRT::MatCSR_Eigen& a, const IMRT::MatCSR_Eigen& b);

#endif