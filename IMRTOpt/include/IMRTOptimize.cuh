#ifndef __IMRTOPTIMIZE_CUH__
#define __IMRTOPTIMIZE_CUH__
#include <vector>
#include <cublas_v2.h>
#include "IMRTInit.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTOptimize_var.cuh"

#define SHOW_VAR(obj, var) viewArray(obj, var, __FILE__, __LINE__)
#define BOO_IMRT_DEBUG true

#define checkCublas(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

namespace IMRT {
    bool StructsTrim(std::vector<StructInfo>& structs_trimmed,
        const std::vector<StructInfo>& structs);
    bool imdilate(std::vector<uint8_t>& dest,
        const std::vector<uint8_t>& source, uint3 shape, uint3 kernelSize);

    class Weights_h {
    public:
        std::vector<float> maxDose;  // PTV and OAR
        std::vector<float> maxWeightsLong;  // PTV and OAR
        std::vector<float> minDoseTarget;  // PTV only
        std::vector<float> minDoseTargetWeights;  // PTV only
        std::vector<float> OARWeightsLong;  // OAR only

        int voxels_PTV;
        int voxels_OAR;
    };

    bool caseInsensitiveStringCompare(char c1, char c2);
    bool containsCaseInsensitive(const std::string& data, const std::string& pattern);
    bool structComp(const std::tuple<StructInfo, bool, size_t>& a,
        const std::tuple<StructInfo, bool, size_t>& b);
    bool testContains();
    bool test_imdilate();
    bool test_cusparseSlicing();

    // minimize sum_{b=1}^{numBeams} beamWeights(b) * || x_b ||_2
    // + 0.5 * mu * || (A_0 x - minDose_target)_- ||_2^2
    // + sum_{i=0}^{numOars} 0.5 * alpha_i || (A_i x - d_vecs{i})_+ ||_2^2
    // + sum_{i=1}^{numOars} 0.5 * beta_i || A_i x ||_2^2 + eta || Dx ||_1^gamma
    // subject to x >= 0
    bool BOO_IMRT_L2OneHalf_cpu_QL(MatCSR64& A, MatCSR64& ATrans,
        MatCSR64& D, MatCSR64& DTrans, const Weights_d& weights_d,
        const Params& params, const std::vector<uint8_t>& fluenceArray);

    bool BOO_IMRT_L2OneHalf_gpu_QL(
        const std::vector<MatCSR_Eigen>& VOIMatrices,
        const std::vector<MatCSR_Eigen>& VOIMatricesT,
        const std::vector<MatCSR_Eigen>& SpFluenceGrad,
        const std::vector<MatCSR_Eigen>& SpFluenceGradT,
        const Weights_h& weights_h,
        const Params& params,
        const std::vector<uint8_t>& fluenceArray,
        // results
        std::vector<float>& xFull,
        std::vector<float>& costs,
        std::vector<int>& activeBeams,
        std::vector<float>& activeNorms,
        std::vector<std::pair<int, std::vector<int>>>& topN);
        

    class Weights_d {
    public:
        bool fromHost(const Weights_h& source);
        array_1d<float> maxDose;
        array_1d<float> maxWeightsLong;
        array_1d<float> minDoseTarget;
        array_1d<float> minDoseTargetWeights;
        array_1d<float> OARWeightsLong;
    
        int voxels_PTV;
        int voxels_OAR;
    };

    class eval_g {
    public:
        eval_g(size_t ptv_voxels, size_t oar_voxels, size_t d_rows_max);
        ~eval_g();
        // adapt to the changing dimension
        void resize(size_t D_rows_current_);
        // assume the handle and the stream are bound
        float evaluate(const MatCSR64& A, const MatCSR64& D,
            const array_1d<float>& x, float gamma,
            const cusparseHandle_t& handle,
            
            float* maxDose,
            const float* minDoseTarget,
            const float* minDoseTargetWeights,
            const float* maxWeightsLong,
            const float* OARWeightsLong,
            float eta);
    private:
        size_t PTV_voxels;
        size_t OAR_voxels;
        size_t D_rows_max;
        size_t D_rows_current;
        float alpha;
        float beta;

        cudaStream_t stream1 = nullptr;
        cudaStream_t stream2 = nullptr;
        cudaStream_t stream3 = nullptr;
        cudaStream_t stream4 = nullptr;
        cudaStream_t stream5 = nullptr;

        array_1d<float> Ax;
        array_1d<float> prox1;
        array_1d<float> prox2;
        array_1d<float> term3;
        array_1d<float> term4;
        array_1d<float> prox4;

        array_1d<float> sumProx1;
        array_1d<float> sumProx2;
        array_1d<float> sumTerm3;
        array_1d<float> sumProx4;
        array_1d<float> sumProx4Term4;

        cublasHandle_t cublasHandle = nullptr;
    };

    bool arrayInit(array_1d<float>& arr, size_t size);
    void arrayRand01(array_1d<float>& arr);

    __global__ void
    d_calc_prox1(float* prox1_data, float* Ax_data, const float* minDoseTargetData, size_t size);
    __global__ void
    d_calc_prox2(float* prox2_data, float* Ax_data, const float* maxDose, size_t size);
    __global__ void
    d_prox1Norm(float* y, float* x, float t, size_t size);
    __device__ int
    d_sign(float x);
    __global__ void
    d_ATimesBSquare(float* C, const float* A, float* B, size_t size);
    __global__ void
    d_calcSumProx4(float* sumProx4Data, float* prox4Data, size_t size);
    __global__ void
    d_calcProx4Term4(float* sumProx4Term4Data, float* prox4Data,
        float* term4Data, size_t size);


    class eval_grad {
    public:
        eval_grad(size_t ptv_voxels, size_t oar_voxels,
            size_t numBeamlets_max_, size_t d_rows_max_);
        ~eval_grad();
        void resize(size_t numBeamlets_current_, size_t D_rows_current_);
        float evaluate(const MatCSR64& A, const MatCSR64& ATrans,
            const MatCSR64& D, const MatCSR64& DTrans,
            const array_1d<float>& x, array_1d<float>& grad, float gamma,
            const cusparseHandle_t& handle,
            
            float* maxDose,
            const float* minDoseTarget,
            const float* minDoseTargetWeights,
            const float* maxWeightsLong,
            const float* OARWeightsLong,
            float eta);
    private:
        size_t PTV_voxels;
        size_t OAR_voxels;
        size_t numBeamlets_max;
        size_t numBeamlets_current;
        size_t D_rows_max;
        size_t D_rows_current;
        float alpha;
        float beta;

        cudaStream_t stream1 = nullptr;
        cudaStream_t stream2 = nullptr;
        cudaStream_t stream3 = nullptr;
        cudaStream_t stream4 = nullptr;
        cudaStream_t stream5 = nullptr;

        array_1d<float> Ax;
        array_1d<float> prox1;
        array_1d<float> prox2;
        array_1d<float> term3;
        array_1d<float> term4;
        array_1d<float> prox4;

        array_1d<float> sumProx1;
        array_1d<float> sumProx2;
        array_1d<float> sumTerm3;
        array_1d<float> sumProx4;
        array_1d<float> sumProx4Term4;

        array_1d<float> grad_term1_input;
        array_1d<float> grad_term1_output;
        array_1d<float> grad_term2_input;
        array_1d<float> grad_term2_output;

        cublasHandle_t cublasHandle = nullptr;
    };

    // performs a = b
    void copy_array_1d(array_1d<float>& a, const array_1d<float>& b);
    // performs c = alpha * a + beta * b;
    void linearComb_array_1d(float alpha, const array_1d<float>& a,
        float beta, const array_1d<float>& b, array_1d<float>& c);
    __global__ void
    d_linearComb(float* c, float alpha, float* a, float beta, float* b, size_t size);
    __global__ void
    d_calc_grad_term1_input(float* output,
        size_t PTV_voxels, const float* minDoseTargetWeights, float* prox1Data,
        size_t OAR_voxels, const float* OARWeightsLong, float* term3Data,
        const float* maxWeightsLong, float* prox2Data);
    __global__ void
    d_calc_grad_term2_input(float* output, float* term4Data,
        float* prox4Data, float eta_over_gamma, size_t size);
    __global__ void
    d_elementWiseAdd(float* c, float* a, float* b, size_t size);


    bool arrayInit_group1(const std::vector<array_1d<float>*>& array_group1,
        size_t numBeamlets_max);

    // allocate enough memory space for the sparse matrices in array_group2
    bool arrayInit_group2(const std::vector<MatCSR64*>& array_group2,
        const std::vector<uint8_t>& fluenceArray, int nBeams, int fluenceDim);
    
    class resize_group2 {
    public:
        ~resize_group2() {
            if (this->bufferSize > 0 && this->buffer != nullptr) {
                checkCudaErrors(cudaFree(this->buffer));
            }
            if (this->handle != nullptr) {
                checkCusparse(cusparseDestroy(this->handle));
            }
        }
        bool evaluate(const std::vector<MatCSR64*>& array_group2,
            const array_1d<float>& fluenceArray, int nBeams, int fluenceDim);
        
        size_t bufferSize = 0;
        void* buffer = nullptr;
        cusparseHandle_t handle = nullptr;
    };

    bool arrayToMatScatter(const array_1d<float>& source, MatCSR64& target);

    // this function resizes the matrices and vectors according to the active beams
    bool DimensionReduction(const std::vector<uint8_t>& active_beams,
        const MatReservior& VOIRes, const std::vector<MatCSR_Eigen>& VOIRes_h,
        const MatReservior& VOIResT, const MatReservior& FGRes, const MatReservior& FGResT,
        MatCSR64** A, MatCSR64** ATrans, MatCSR64** D, MatCSR64** DTrans,
        eval_g& operator_eval_g, eval_grad& operator_eval_grad,
        const std::vector<array_1d<float>*>& array_group1,
        const std::vector<MatCSR64*>& array_group2,
        resize_group2& operator_resize_group2,
        const std::vector<uint8_t>& fluenceArray, int fluenceDim,
        const cusparseHandle_t& handle);

    // this function performs the element-wise operation: target[i] = max(target[i], value);
    bool elementWiseMax(MatCSR64& target, float value);
    __global__ void d_elementWiseMax(float* target, float value, size_t size);

    bool assignmentTest();
}
#define sanityCheck true


#endif