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
    const MatCSR64& D, const MatCSR64& DTrans, const Weights_d& weights_d,
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
    
    beamWeightsInit(params, weights_d, A, ATrans, fluenceArray,
        beamWeights_d, BeamletLog0_d, xkm1, output_d);
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Weights initialiazation time elapsed: " 
            << duration.count() * 0.001f << " [s]" << std::endl;
    #endif


    int numBeamlets = A.numCols;
    array_1d<float> vkm1;
    int numBeamsLast = beamWeights_d.size;
    
    vkm1 = xkm1;

    return 0;
}


bool IMRT::beamWeightsInit(
    const Params& params, const Weights_d& weights_d,
    MatCSR64& A, MatCSR64& ATrans, const std::vector<uint8_t>& fluenceArray,
    array_1d<float>& beamWeights_d, array_1d<uint8_t>& BeamletLog0_d,
    array_1d<float>& input_d, array_1d<float>& output_d
) {
    std::vector<float> input_h(A.numCols);
    // initialize input_h to to the range [0, 1]
    for (int i=0; i<input_h.size(); i++)
        input_h[i] = std::rand() / RAND_MAX;
    std::vector<float> output_h(A.numRows, 1.0f);
    checkCudaErrors(cudaMalloc((void**)&input_d.data, input_h.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&output_d.data, output_h.size()*sizeof(float)));
    checkCudaErrors(cudaMemcpy(input_d.data, input_h.data(),
        input_h.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(output_d.data, output_h.data(),
        output_h.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCusparse(cusparseCreateDnVec(&input_d.vec, input_h.size(), input_d.data, CUDA_R_32F));
    checkCusparse(cusparseCreateDnVec(&output_d.vec, output_h.size(), output_d.data, CUDA_R_32F));
    input_d.size = input_h.size();
    output_d.size = output_h.size();

    #if true
        // sanity check
        size_t total_beamlets = 0;
        for (int i=0; i<fluenceArray.size(); i++)
            total_beamlets += fluenceArray[i] > 0;
        if (total_beamlets != A.numCols) {
            std::cerr << "The number of columns of matrix A should be the same as "
                "the total number of beamlets. However, A.numCols = " << A.numCols
                << ", total_beamlets = " << total_beamlets << std::endl;
            return 1;
        }
    #endif

    cusparseHandle_t handle = nullptr;
    checkCusparse(cusparseCreate(&handle));

    // allocate SpMV buffer memory
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.matA, input_d.vec, &beta, output_d.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc((void**)&A.d_buffer_spmv, bufferSize));
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, ATrans.matA, output_d.vec, &beta, input_d.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc((void**)&ATrans.d_buffer_spmv, bufferSize));

    // count the number of beamlets per beam
    int fluenceDim = getarg<int>("fluenceDim");
    int numElementsPerBB = fluenceDim * fluenceDim;
    if (fluenceArray.size() % numElementsPerBB != 0) {
        std::cerr << "The fluence array size, " << fluenceArray.size() 
            << ", is supposed to be a multiple of the number of elements per beam, "
            << numElementsPerBB << std::endl;
        return 1;
    }
    int numBeams = fluenceArray.size() / numElementsPerBB;
    std::vector<int> ActiveBeamletsPerBeam(numBeams, 0);
    for (int i=0; i<numBeams; i++) {
        size_t offset = i * numElementsPerBB;
        for (int j=0; j<numElementsPerBB; j++)
            ActiveBeamletsPerBeam[i] += fluenceArray[offset + j] > 0;
    }

    std::vector<float> beamWeights_h(numBeams);
    size_t idx_start = 0;
    for (int i=0; i<numBeams; i++) {
        int current_beamlets = ActiveBeamletsPerBeam[i];
        std::fill(input_h.begin(), input_h.end(), 0.0f);
        std::fill(input_h.begin() + idx_start, input_h.begin() +
            idx_start + current_beamlets, 1.0f);
        idx_start += current_beamlets;
        checkCudaErrors(cudaMemcpy(input_d.data, input_h.data(),
            input_h.size()*sizeof(float), cudaMemcpyHostToDevice));
        
        checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A.matA, input_d.vec, &beta, output_d.vec, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, A.d_buffer_spmv));
        
        // we only care about the PTV dose
        checkCudaErrors(cudaMemcpy(output_h.data(), output_d.data,
            weights_d.voxels_PTV * sizeof(float), cudaMemcpyDeviceToHost));
        // sum it over
        float PTV_sum = 0.0f;
        for (int j=0; j<weights_d.voxels_PTV; j++)
            PTV_sum += output_h[j];
        float PTV_mean = PTV_sum / weights_d.voxels_PTV;
        beamWeights_h[i] = sqrtf(PTV_mean / current_beamlets);
    }

    // beamWeights_h regularization
    float beamWeights_max = 0.0f;
    for (int i=0; i<numBeams; i++)
        beamWeights_max = std::max(beamWeights_max, beamWeights_h[i]);
    for (int i=0; i<numBeams; i++) {
        beamWeights_h[i] /= beamWeights_max;
        beamWeights_h[i] = std::max(beamWeights_h[i], 0.1f);
        beamWeights_h[i] *= params.beamWeight;
    }

    if (beamWeights_d.data != nullptr || BeamletLog0_d.data != nullptr) {
        std::cerr << "The data vectors of beamWeights_d and BeamletLog0_d "
            "are supposed to be nullptr." << std::endl;
        return 1;
    }
    checkCudaErrors(cudaMalloc((void**)&beamWeights_d.data, beamWeights_h.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&BeamletLog0_d.data, fluenceArray.size()*sizeof(uint8_t)));
    checkCudaErrors(cudaMemcpy(beamWeights_d.data, beamWeights_h.data(),
        beamWeights_h.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(BeamletLog0_d.data, fluenceArray.data(),
        fluenceArray.size()*sizeof(uint8_t), cudaMemcpyHostToDevice));
    beamWeights_d.size = beamWeights_h.size();
    BeamletLog0_d.size = fluenceArray.size();

    #if false
        for debug purposes
        std::cout << "beamWeights:\n";
        for (int i=0; i<beamWeights_h.size(); i++) {
            std::cout << "beam " << i << ": " << beamWeights_h[i] << ",  ";
        }
        std::cout << "\n\nBeamletLog0:" << std::endl;

        for (int i=0; i<beamWeights_h.size(); i++) {
            std::cout << "beam " << i << "\n";
            size_t BeamletLog0_offset = i * numElementsPerBB;
            for (size_t j=0; j<numElementsPerBB; j++) {
                std::cout << (int)fluenceArray[BeamletLog0_offset + j] << "  ";
            }
            std::cout << "\n" << std::endl;
        }
    #endif

    checkCusparse(cusparseDestroy(handle));
    return 0;
}