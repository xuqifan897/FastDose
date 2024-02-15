#include <string>
#include <tuple>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include "IMRTOptimize.cuh"
#include "IMRTInit.cuh"
#include "IMRTDoseMatEigen.cuh"
#include "IMRTArgs.h"

bool IMRT::StructsTrim(std::vector<StructInfo>& structs_trimmed,
    const std::vector<StructInfo>& structs) {
    std::cout << "Remove the overlapping part of the OARs with the PTV, "
        "exclude the parts that are irrelevant in the optimization. " 
        "Assuming that the zeroth structure is PTV, the first structure is BODY"
        << std::endl;
    
    const StructInfo& PTV = structs[0];
    const StructInfo& BODY = structs[1];
    const std::vector<uint8_t> PTV_mask = PTV.mask;
    uint3 size = PTV.size;

    std::vector<uint8_t> PTV_dilate;
    uint3 kernelSize{3, 3, 3};
    imdilate(PTV_dilate, PTV_mask, size, kernelSize);

    std::vector<uint8_t> IsPTV(structs.size(), 0);
    IsPTV[0] = 1;
    std::vector<uint8_t> IsOAR(structs.size(), 0);
    for (int i=0; i<structs.size(); i++) {
        const StructInfo& local_struct = structs[i];
        if (local_struct.maxWeights < eps_fastdose
            && local_struct.minDoseTargetWeights < eps_fastdose
            && local_struct.OARWeights < eps_fastdose) {
            IsOAR[i] = 0;
            continue;
        }
        IsOAR[i] = 1;
    }
    return 0;
}


bool IMRT::imdilate(std::vector<uint8_t>& dest,
    const std::vector<uint8_t>& source, uint3 shape, uint3 kernelSize) {
    // firstly, all the elements of kernelSize should be odd numbers
    if (kernelSize.x % 2 == 0 || kernelSize.y % 2 == 0 || kernelSize.z % 2 == 0) {
        std::cerr << "At least one kernelSize element is even. kernelSize: "
            << kernelSize << std::endl;
        return 1;
    }
    dest.resize(source.size());
    std::fill(dest.begin(), dest.end(), 0);

    uint3 halfKernelSize {(kernelSize.x - 1) / 2, (kernelSize.y - 1) / 2,
        (kernelSize.z - 1) / 2};

    for (int k=0; k<shape.z; k++) {
        int k_begin = std::max(0, k - (int)halfKernelSize.z);
        int k_end = std::min((int)shape.z, k + (int)halfKernelSize.z + 1);
        for (int j=0; j<shape.y; j++) {
            int j_begin = std::max(0, j - (int)halfKernelSize.y);
            int j_end = std::min((int)shape.y, j + (int)halfKernelSize.y + 1);
            for (int i=0; i<shape.x; i++) {
                size_t idx_source = i + shape.x * (j + shape.y * k);
                if (source[idx_source]) {
                    // dilate around the voxel
                    int i_begin = std::max(0, i - (int)halfKernelSize.x);
                    int i_end = std::min((int)shape.x, i + (int)halfKernelSize.x + 1);


                    for (int kk=k_begin; kk<k_end; kk++) {
                        for (int jj=j_begin; jj<j_end; jj++) {
                            for (int ii=i_begin; ii<i_end; ii++) {
                                size_t idx_dest = ii + shape.x * (jj + shape.y * kk);
                                dest[idx_dest] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}


bool IMRT::test_imdilate(){
    uint3 shape {10, 10, 3};
    std::vector<uint8_t> middle_slice {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    std::vector<uint8_t> source(shape.x * shape.y * shape.z, 0);
    std::copy(middle_slice.begin(), middle_slice.end(), source.begin() + shape.x * shape.y);

    std::vector<uint8_t> result;
    uint3 kernelSize{3, 3, 3};
    if (imdilate(result, source, shape, kernelSize)) {
        std::cerr << "The function imdilate() error." << std::endl;
        return 1;
    }
    for (int k=0; k<shape.z; k++) {
        for (int j=0; j<shape.y; j++) {
            for (int i=0; i<shape.x; i++) {
                size_t idx = i + shape.x * (j + shape.y * k);
                std::cout << (int)result[idx] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
    return 0;
}

bool IMRT::caseInsensitiveStringCompare(char c1, char c2) {
    return std::tolower(c1) == std::tolower(c2);
}

bool IMRT::containsCaseInsensitive(const std::string& data, const std::string& pattern) {
    auto it = std::search(data.begin(), data.end(),
        pattern.begin(), pattern.end(),
        caseInsensitiveStringCompare);
    
    return it != data.end();
}

bool IMRT::testContains() {
    std::vector<std::string> dataArray{"happy", "HaPpY", "Hobby", "hapPY"};
    std::string pattern {"APP"};
    for (const std::string& data : dataArray) {
        std::cout << "find(\"" << data << "\", \"" << pattern << "\") = "
            << containsCaseInsensitive(data, pattern) << std::endl;;
    }
    return 0;
}


bool IMRT::structComp(const std::tuple<StructInfo, bool, size_t>& a,
    const std::tuple<StructInfo, bool, size_t>& b) {
    return (std::get<1>(a) && !std::get<1>(b));
}


bool IMRT::testWeights(const Weights_h& weights) {
    std::cout << std::fixed << std::setprecision(0);
    for (size_t i=0; i<weights.maxDose.size(); i++)
        std::cout << weights.maxDose[i] << " ";
    std::cout << "\n maxDose size: " << weights.maxDose.size() << "\n" << std::endl;

    for (size_t i=0; i< weights.maxWeightsLong.size(); i++)
        std::cout << weights.maxWeightsLong[i] << " ";
    std::cout << "\nmaxWeightsLong size: " << weights.maxWeightsLong.size() << "\n" << std::endl;

    for (size_t i=0; i<weights.minDoseTarget.size(); i++)
        std::cout << weights.minDoseTarget[i] << " ";
    std::cout << "\nminDoseTarget size: " << weights.minDoseTarget.size() << "\n" << std::endl;

    for (size_t i=0; i<weights.minDoseTargetWeights.size(); i++)
        std::cout << weights.minDoseTargetWeights[i] << " ";
    std::cout << "\nminDoseTargetWeights: " << weights.minDoseTargetWeights.size()
        << "\n" << std::endl;
    
    for (size_t i=0; i<weights.OARWeightsLong.size(); i++)
        std::cout << weights.OARWeightsLong[i] << " ";
    std::cout << "\nOARWeightsLong: " << weights.OARWeightsLong.size() << "\n" << std::endl;
    return 0;
}


bool IMRT::Weights_d::fromHost(const Weights_h& source) {
    // sanity check
    if (source.maxDose.size() != source.voxels_PTV + source.voxels_OAR
        || source.maxWeightsLong.size() != source.voxels_PTV + source.voxels_OAR
        || source.minDoseTarget.size() != source.voxels_PTV
        || source.minDoseTargetWeights.size() != source.voxels_PTV
        || source.OARWeightsLong.size() != source.voxels_OAR) {
        std::cerr << "There is inconsistency between the weights array sizes." << std::endl;
        return 1;
    }

    if (this->maxDose || this->maxWeightsLong || this->minDoseTarget
        || this->minDoseTargetWeights || this->OARWeightsLong) {
        std::cerr << "All weight arrays are supposed to be nullptr "
            "before initialization." << std::endl;
        return 1;
    }

    checkCudaErrors(cudaMalloc((void**)&this->maxDose, source.maxDose.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&this->maxWeightsLong,
        source.maxWeightsLong.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&this->minDoseTarget,
        source.minDoseTarget.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&this->minDoseTargetWeights,
        source.minDoseTargetWeights.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&this->OARWeightsLong,
        source.OARWeightsLong.size()*sizeof(float)));

    checkCudaErrors(cudaMemcpy(this->maxDose, source.maxDose.data(),
        source.maxDose.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->maxWeightsLong, source.maxWeightsLong.data(),
        source.maxWeightsLong.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->minDoseTarget, source.minDoseTarget.data(),
        source.minDoseTarget.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->minDoseTargetWeights, source.minDoseTargetWeights.data(),
        source.minDoseTargetWeights.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->OARWeightsLong, source.OARWeightsLong.data(),
        source.OARWeightsLong.size()*sizeof(float), cudaMemcpyHostToDevice));

    this->voxels_PTV = source.voxels_PTV;
    this->voxels_OAR = source.voxels_OAR;
    return 0;
}

IMRT::Weights_d::~Weights_d() {
    if (this->maxDose)
        checkCudaErrors(cudaFree(this->maxDose));
    if (this->maxWeightsLong)
        checkCudaErrors(cudaFree(this->maxWeightsLong));
    if (this->minDoseTarget)
        checkCudaErrors(cudaFree(this->minDoseTarget));
    if (this->minDoseTargetWeights)
        checkCudaErrors(cudaFree(this->minDoseTargetWeights));
    if (this->OARWeightsLong)
        checkCudaErrors(cudaFree(this->OARWeightsLong));
}


bool IMRT::BOO_IMRT_L2OneHalf_cpu_QL(MatCSR64& A, MatCSR64& ATrans,
    MatCSR64& D, MatCSR64& DTrans, const Weights_d& weights_d,
    const Params& params, const std::vector<uint8_t>& fluenceArray
) {
    // initialization
    int pruneTrigger = 40;
    float reductionFactor = 0.5f;
    array_1d<float> beamWeights_d;
    array_1d<uint8_t> BeamletLog0_d;
    array_1d<float> xkm1;
    array_1d<float> output_d;

    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    
    // beamWeightsInit(params, weights_d, A, ATrans, D, DTrans, fluenceArray,
    //     beamWeights_d, BeamletLog0_d, xkm1, output_d);
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Weights initialiazation time elapsed: " 
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif

    return 0;


    cusparseHandle_t handle;
    checkCusparse(cusparseCreate(&handle));
    int numBeamlets = A.numCols;
    array_1d<float> vkm1;
    int numBeamsLast = beamWeights_d.size;
    eval_g operator_eval_g(weights_d.voxels_PTV, weights_d.voxels_OAR, D.numRows);
    
    vkm1 = xkm1;
    // evaluate all-zero cost
    checkCudaErrors(cudaMemset(vkm1.data, 0, vkm1.size*sizeof(float)));
    float all_zero_cost = operator_eval_g.evaluate(A, D, vkm1,
        weights_d.maxDose, params.gamma, handle,
        
        weights_d.minDoseTarget,
        weights_d.minDoseTargetWeights,
        weights_d.maxWeightsLong,
        weights_d.OARWeightsLong,
        params.eta);

    std::cout << "All-zero cost is: " << all_zero_cost << std::endl;

    
    // clean up
    checkCusparse(cusparseDestroy(handle));

    return 0;
}