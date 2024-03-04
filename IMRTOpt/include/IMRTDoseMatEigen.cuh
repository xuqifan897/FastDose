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
    
    bool parallelMatCoalease(MatCSR_Eigen& VOImat, MatCSR_Eigen& VOImatT,
        const std::vector<MatCSR_Eigen>& VOIMatrices,
        const std::vector<MatCSR_Eigen>& VOIMatricesT);
    
    class MatCSR64;
    bool MatOARSlicing(const MatCSR_Eigen& matrixT, MatCSR_Eigen& A,
        MatCSR_Eigen& AT, const std::vector<StructInfo>& structs);

    class Weights_h; class Weights_d;
    bool OARFiltering(const std::string& resultFolder,
        const std::vector<StructInfo>& structs,
        MatCSR64& SpVOImat, MatCSR64& SpVOImatT,
        Weights_h& weights, Weights_d& weights_d);

    bool fluenceGradInit(MatCSR64& SpFluenceGrad, MatCSR64& SpFluenceGradT,
        std::vector<uint8_t>& fluenceArray, const std::string& fluenceMapPath,
        int fluenceDim);

    bool matFuseFunc(
        std::vector<MatCSR_Eigen*>& VOIMatrices,
        std::vector<MatCSR_Eigen*>& VOIMatricesT,
        std::vector<MatCSR_Eigen*>& SpFluenceGrad,
        std::vector<MatCSR_Eigen*>& SpFluenceGradT,
        MatCSR_Eigen& VOIMat_Eigen,
        MatCSR_Eigen& VOIMatT_Eigen,
        MatCSR_Eigen& D_Eigen,
        MatCSR_Eigen& DTrans_Eigen);

    bool diagBlock(MatCSR_Eigen& target, const std::vector<MatCSR_Eigen*>& source);

    bool fluenceGradInit(
        std::vector<IMRT::MatCSR_Eigen>& SpFluenceGrad,
        std::vector<IMRT::MatCSR_Eigen>& SpFluenceGradT,
        std::vector<uint8_t>& fluenceArray,
        const std::string& fluenceMapPath);

    bool getStructFilter(MatCSR_Eigen& filter, MatCSR_Eigen& filterT,
        const std::vector<StructInfo>& structs, Weights_h& weights);
    
    // size in bytes
    bool readBlockParallel(const std::string& filename, void** pointer, EigenIdxType* size);

    // start, end in bytes
    void readBlockParallelFunc(const std::string& filename,
        char* buffer, size_t start, size_t end);
    
    bool Eigen2Cusparse(const MatCSR_Eigen& source, MatCSR64& dest);
    bool DxyInit(IMRT::MatCSR_Eigen& Dxy, size_t size);
    bool IdentityInit(IMRT::MatCSR_Eigen& Id, size_t size);
    bool KroneckerProduct(const MatCSR_Eigen& A,
        const MatCSR_Eigen& B, MatCSR_Eigen& C);
    bool filterConstruction(MatCSR_Eigen& filter, const std::vector<uint8_t>& array);

    bool test_parallelSpGEMM(const std::vector<MatCSR_Eigen>& OARMatrices,
        const std::vector<MatCSR_Eigen>& OARMatricesT,
        const std::vector<MatCSR_Eigen>& matricesT,
        const MatCSR_Eigen& filter);

    bool test_OARMat_OARMatT(const MatCSR_Eigen& OARMat, const MatCSR_Eigen& OARMatT);

    bool test_KroneckerProduct();
    bool test_filterConstruction();
    bool testWeights(const Weights_h& weights_h);
}

IMRT::MatCSR_Eigen operator*(const IMRT::MatCSR_Eigen& a, const IMRT::MatCSR_Eigen& b);

#endif