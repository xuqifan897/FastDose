#ifndef __IMRTOPTIMIZE_CUH__
#define __IMRTOPTIMIZE_CUH__
#include <vector>
#include "IMRTInit.cuh"
#include "IMRTDoseMat.cuh"

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

    class Weights_d {
    public:
        bool fromHost(const Weights_h& source);
        ~Weights_d();
        float* maxDose = nullptr;
        float* maxWeightsLong = nullptr;
        float* minDoseTarget = nullptr;
        float* minDoseTargetWeights = nullptr;
        float* OARWeightsLong = nullptr;
    
        int voxels_PTV;
        int voxels_OAR;
    };
    bool caseInsensitiveStringCompare(char c1, char c2);
    bool containsCaseInsensitive(const std::string& data, const std::string& pattern);
    bool structComp(const std::tuple<StructInfo, bool, size_t>& a,
        const std::tuple<StructInfo, bool, size_t>& b);
    bool testContains();
    bool test_imdilate();

    // minimize sum_{b=1}^{numBeams} beamWeights(b) * || x_b ||_2
    // + 0.5 * mu * || (A_0 x - minDose_target)_- ||_2^2
    // + sum_{i=0}^{numOars} 0.5 * alpha_i || (A_i x - d_vecs{i})_+ ||_2^2
    // + sum_{i=1}^{numOars} 0.5 * beta_i || A_i x ||_2^2 + eta || Dx ||_1^gamma
    // subject to x >= 0
    bool BOO_IMRT_L2OneHalf_cpu_QL(MatCSR64& A, MatCSR64& ATrans,
        const MatCSR64& D, const MatCSR64& DTrans, const Weights_d& weights_d,
        const Params& params, const std::vector<uint8_t>& fluenceArray);

    template<class T>
    class array_1d {
    public:
        ~array_1d() {
            if (this->data != nullptr) {
                checkCudaErrors(cudaFree(this->data));
                this->data = nullptr;
            }
            if (this->vec != nullptr) {
                checkCusparse(cusparseDestroyDnVec(this->vec));
                this->vec = nullptr;
            }
        }
        array_1d<T>& operator=(const array_1d<T>& other);
        T* data = nullptr;
        cusparseDnVecDescr_t vec = nullptr;
        size_t size = 0;
    };

    bool beamWeightsInit(
        const Params& params, const Weights_d& weights_d,
        MatCSR64& A, MatCSR64& ATrans, const std::vector<uint8_t>& fluenceArray,
        array_1d<float>& beamWeights, array_1d<uint8_t>& BeamletLog,
        array_1d<float>& input_d, array_1d<float>& output_d
    );

    // // using static variables to avoid recurrent initialization and destruction.
    // // Be cautious when using multi-threading.
    // float eval_g(const MatCSR64& A, const MatCSR64& D,
    //     const array_1d<float>& x, bool initFlag=false,
    //     size_t ptv_voxels=0, size_t oar_voxels=0);


    class eval_g {
    public:
        eval_g(size_t ptv_voxels, size_t oar_voxels);
        float evaluate();
    private:
        size_t PTV_voxels;
        size_t OAR_voxels;
        array_1d<float> Ax;
        array_1d<float> prox1;
        array_1d<float> prox2;
        array_1d<float> term3;
        array_1d<float> term4;
        array_1d<float> prox4;
    };


    bool assignmentTest();
}



#endif