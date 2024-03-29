#ifndef __IMRTOPTIMIZE_VAR_CUH__
#define __IMRTOPTIMIZE_VAR_CUH__
#include "IMRTDoseMat.cuh"
#include "IMRTDoseMatEigen.cuh"
#include "cublas_v2.h"

namespace IMRT {
    template<class T>
    class array_1d {
    public:
        ~array_1d();
        bool resize(size_t new_size);
        bool copy(const array_1d<T>& old);
        friend std::ostream& operator<<(std::ostream& os, const array_1d<T>& obj) {
            std::vector<T> array_host(obj.size);
            checkCudaErrors(cudaMemcpy(array_host.data(), obj.data,
                obj.size*sizeof(T), cudaMemcpyDeviceToHost));
            for (size_t i=0; i<obj.size; i++)
                os << array_host[i] << "  ";
            os << "\n";
            return os;
        }

        T* data = nullptr;
        cusparseDnVecDescr_t vec = nullptr;
        size_t size = 0;
        size_t init_flag = false;
    };

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
        bool assemble_col_block(MatCSR64& target, const std::vector<MatCSR_Eigen>& reservior_h,
            const std::vector<uint8_t>& flags) const;
        bool assemble_diag(MatCSR64& target, const std::vector<uint8_t>& flags) const;
        std::vector<MatCSR64> reservior;
    };

    __global__ void
    d_assembly_row_block(size_t* d_csr_offsets,
        size_t* cumu_row, size_t* cumu_nnz, size_t numMatrices);
    __global__ void
    d_assembly_col_block(size_t* d_csr_offsets, size_t* d_csr_columns, float* d_csr_values,
        size_t** source_offsets, size_t** source_columns, float** source_values,
        size_t* source_columns_offset, size_t numRows, size_t numMatrices);
    __global__ void
    d_assemble_diag(size_t* d_csr_offsets, size_t* d_csr_columns, float* d_csr_values,
        size_t** source_offsets, size_t** source_columns, float** source_values,
        size_t* rowOff, size_t* colOff, size_t* nnzOff, size_t numMatrices);

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

    bool MatReservior_dev_diag(
        const std::vector<MatCSR_Eigen>& VOIMatrices,
        const std::vector<MatCSR_Eigen>& VOIMatricesT,
        const std::vector<MatCSR_Eigen>& SpFluenceGrad,
        const std::vector<MatCSR_Eigen>& SpFluenceGradT);
    
    // This function verifies whether the two input matrices
    // are transposes of each other
    bool transposeConsistency(const MatCSR64& A, const MatCSR64& ATrans,
        const cusparseHandle_t& handle);
    
    template<typename T> class array_1d;

    bool beamWeightsInit_func(
        const MatReservior& VOIRes, array_1d<float>& beamWeightsInit,
        size_t ptv_voxels, size_t oar_voxels, const cusparseHandle_t& handle,
        const cublasHandle_t& cublas_handle);

    bool beamWeightsInit_func(
        const std::vector<const MatCSR_Eigen*>& VOIMatrices,
        Eigen::VectorXf& beamWeightsInit,
        size_t ptv_voxels, size_t oar_voxels);

    // this function calculates tHat from nrm2 in-place.
    __global__ void d_proxL2Onehalf_calc_tHat(float* buffer, float* tau, size_t size);

    class calc_rhs {
    public:
        calc_rhs();
        // allocate enough space for calculation
        // bufferSize in the number of elements.
        bool customInit(size_t bufferSize);
        bool evaluate(float& rhs, float gy, float t, const array_1d<float>& gradAty,
            const array_1d<float>& x, const array_1d<float>& y,
            const cublasHandle_t& cublas_handle);

        array_1d<float> firstTerm;
        array_1d<float> x_minus_y;
    };

    class calc_loss {
    public:
        calc_loss();
        bool customInit(size_t nBeams);
        bool evaluate(
            float& cost, float gx, const array_1d<float>& beamWeights,
            const array_1d<float>& beamNorms, const cublasHandle_t& cublas_handle);
        
        array_1d<float> buffer_sqrt;
        array_1d<float> buffer_mul;
    };

    bool elementWiseSquare(float* source, float* target, size_t size);
    __global__ void d_elementWiseSquare(float* source, float* target, size_t size);

    // performs target[i] = source[i] ^ power
    bool elementWisePower(float* source, float* target, float power, size_t size);
    __global__ void d_elementWisePower(float* source, float* target, float power, size_t size);

    // performs c[i] = a[i] * b[i] * t.
    bool elementWiseMul(float* a, float* b, float* c, float t, size_t size);
    __global__ void d_elementWiseMul(float* a, float* b, float* c, float t, size_t size);
    
    bool bufferAllocate(MatCSR64& target, const array_1d<float>& input,
        const array_1d<float>& output, const cusparseHandle_t& handle);
    
    
    // performs tHat(alpha > 2 * root6 / 9) = 0; nrm2 is used to tell whether
    // the alpha element is nan. alpha element is nan when nrm2 is 0.
    bool tHat_step2(float* tHat, float* alpha, float* nrm2, size_t size);
    __global__ void d_tHat_step2(float* tHat, float* alpha, float* nrm2, size_t size);


    // performs tHat(alpha > 2 * root6 / 9) = 0; nrm2 is used to tell whether
    // the alpha element is nan. alpha element is nan when nrm2 is 0.
    bool tHat_step2(float* tHat, float* alpha, float* nrm2, size_t size);
    __global__ void d_tHat_step2(float* tHat, float* alpha, float* nrm2, size_t size);

    // performs prox = tHat' * g0 in an element-wise manner
    // as prox has the same structure as g0, just provide prox_values
    bool calc_prox_tHat_g0(size_t numRows, size_t numCols, size_t nnz,
        float* tHat, size_t* g0_offsets, size_t* g0_columns,
        float* g0_values, float* prox_values);
    __global__ void d_calc_prox_tHat_g0(size_t numRows, size_t numCols, size_t nnz,
        float* tHat, size_t* g0_offsets, size_t* g0_columns,
        float* g0_values, float* prox_values);
    
    // performs target[i] = sqrtf(source[i])
    bool elementWiseSqrt(float* source, float* target, size_t size);
    __global__ void d_elementWiseSqrt(float* source, float* target, size_t size);

    // performs source[i] = a * source[i];
    bool elementWiseScale(float* source, float a, size_t size);
    __global__ void d_elementWiseScale(float* source, float a, size_t size);

    // overload, performs target[i] = a * source[i]
    bool elementWiseScale(float* source, float* target, float a, size_t size);
    __global__ void d_elementWiseScale(float* source, float* target, float a, size_t size);

    // performs target[i] = source[i] > a;
    bool elementWiseGreater(float* source, float* target, float a, size_t size);
    __global__ void d_elementWiseGreater(
        float* source, float* target, float a, size_t size);

    bool beamSort(const std::vector<float>& beamNorms_h,
        std::vector<std::pair<int, float>>& beamNorms_pair);
}
#endif