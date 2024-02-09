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
    
    beamWeightsInit(params, weights_d, A, ATrans, D, DTrans, fluenceArray,
        beamWeights_d, BeamletLog0_d, xkm1, output_d);
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


bool IMRT::beamWeightsInit(
    const Params& params, const Weights_d& weights_d,
    MatCSR64& A, MatCSR64& ATrans, MatCSR64& D, MatCSR64& DTrans,
    const std::vector<uint8_t>& fluenceArray,
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


    array_1d<float> DTrans_input;
    DTrans_input.size = D.numRows;
    checkCudaErrors(cudaMalloc((void**)&DTrans_input.data, DTrans_input.size*sizeof(float)));
    std::vector<float> DTrans_input_h(DTrans_input.size, 1.0f);
    checkCudaErrors(cudaMemcpy(DTrans_input.data, DTrans_input_h.data(),
        DTrans_input.size*sizeof(float), cudaMemcpyHostToDevice));
    checkCusparse(cusparseCreateDnVec(&DTrans_input.vec,
        DTrans_input.size, DTrans_input.data, CUDA_R_32F));

    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, D.matA, input_d.vec, &beta, DTrans_input.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc((void**)&D.d_buffer_spmv, bufferSize));

    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, DTrans.matA, DTrans_input.vec, &beta, input_d.vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc((void**)&DTrans.d_buffer_spmv, bufferSize));

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

    MatCSR64 mat_slicing;
    cusparseDnVecDescr_t vec_input = nullptr;
    float* vec_input_values = nullptr;
    cusparseDnVecDescr_t vec_output;
    float* vec_output_values = nullptr;

    std::vector<size_t> offsets_h(ATrans.numRows + 1, 0);
    checkCudaErrors(cudaMemcpy(offsets_h.data(), ATrans.d_csr_offsets,
        (ATrans.numRows + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));

    size_t numRows_max = 0;
    size_t numRows_cumu = 0;
    for (int i=0; i<numBeams; i++) {
        size_t current_beamlets = ActiveBeamletsPerBeam[i];
        size_t current_nnz = offsets_h[numRows_cumu + current_beamlets] - offsets_h[numRows_cumu];
        numRows_cumu += current_beamlets;
        numRows_max = max(numRows_max, current_beamlets);
    }
    checkCudaErrors(cudaMalloc((void**)&vec_input_values, numRows_max*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&vec_output_values, ATrans.numCols*sizeof(float)));

    size_t* mat_slicing_offsets = nullptr;
    checkCudaErrors(cudaMalloc((void**)&mat_slicing_offsets, (numRows_max + 1) * sizeof(size_t)));

    std::vector<float> vec_input_values_h(numRows_max, 1.0f);
    checkCudaErrors(cudaMemcpy(vec_input_values, vec_input_values_h.data(),
        numRows_max*sizeof(float), cudaMemcpyHostToDevice));
    
    checkCusparse(cusparseCreateDnVec(&vec_output, ATrans.numCols, vec_output_values, CUDA_R_32F));
    bufferSize = 0;

    numRows_cumu = 0;
    cublasHandle_t cublasHandle = nullptr;
    checkCublas(cublasCreate(&cublasHandle));
    for (int i=0; i<numBeams; i++) {
        size_t current_beamlets = ActiveBeamletsPerBeam[i];
        size_t nnz_idx_begin = offsets_h[numRows_cumu];
        size_t nnz_idx_end = offsets_h[numRows_cumu + current_beamlets];

        mat_slicing.numRows = current_beamlets;
        mat_slicing.numCols = ATrans.numCols;
        mat_slicing.nnz = nnz_idx_end - nnz_idx_begin;
        mat_slicing.d_csr_columns = ATrans.d_csr_columns + nnz_idx_begin;
        mat_slicing.d_csr_values = ATrans.d_csr_values + nnz_idx_begin;

        std::vector<size_t> mat_slicing_offsets_h(mat_slicing.numRows+1);
        std::copy(&offsets_h[numRows_cumu], &offsets_h[numRows_cumu + current_beamlets + 1],
            mat_slicing_offsets_h.begin());
        size_t first_value = mat_slicing_offsets_h[0];
        for (size_t j=0; j<mat_slicing_offsets_h.size(); j++)
            mat_slicing_offsets_h[j] -= first_value;
        checkCudaErrors(cudaMemcpy(mat_slicing_offsets, mat_slicing_offsets_h.data(),
            mat_slicing_offsets_h.size()*sizeof(size_t), cudaMemcpyHostToDevice));
        mat_slicing.d_csr_offsets = mat_slicing_offsets;

        numRows_cumu += current_beamlets;

        checkCusparse(cusparseCreateCsr(
            &mat_slicing.matA, mat_slicing.numRows, mat_slicing.numCols, mat_slicing.nnz,
            mat_slicing.d_csr_offsets, mat_slicing.d_csr_columns, mat_slicing.d_csr_values,
            CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        checkCusparse(cusparseCreateDnVec(&vec_input, current_beamlets, vec_input_values, CUDA_R_32F));

        size_t current_buffer_size = 0;
        checkCusparse(cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, mat_slicing.matA, vec_input, &beta, vec_output, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &current_buffer_size));

        if (current_buffer_size > bufferSize) {
            if (mat_slicing.d_buffer_spmv != nullptr)
                checkCudaErrors(cudaFree(mat_slicing.d_buffer_spmv));
            checkCudaErrors(cudaMalloc((void**)&mat_slicing.d_buffer_spmv, current_buffer_size));
            #if false
                // for debug purposes
                std::cout << "buffer re-allocation: " << bufferSize << " -> " << current_buffer_size << std::endl;
            #endif
            bufferSize = current_buffer_size;
        }

        checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, mat_slicing.matA, vec_input, &beta, vec_output,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, mat_slicing.d_buffer_spmv));
        
        // sum over PTV
        float PTV_sum = 0.0f;
        checkCublas(cublasSasum(cublasHandle, weights_d.voxels_PTV,
            vec_output_values, 1, &PTV_sum));
        
        float PTV_mean = PTV_sum / weights_d.voxels_PTV;
        beamWeights_h[i] = sqrtf(PTV_mean / current_beamlets);

        checkCusparse(cusparseDestroyDnVec(vec_input));
        checkCusparse(cusparseDestroySpMat(mat_slicing.matA));

    }

    checkCublas(cublasDestroy(cublasHandle));
    mat_slicing.matA = nullptr;
    mat_slicing.d_csr_offsets = nullptr;
    mat_slicing.d_csr_columns = nullptr;
    mat_slicing.d_csr_values = nullptr;
    checkCudaErrors(cudaFree(mat_slicing.d_buffer_spmv));
    mat_slicing.d_buffer_spmv = nullptr;
    checkCusparse(cusparseDestroyDnVec(vec_output));
    checkCudaErrors(cudaFree(vec_input_values));
    checkCudaErrors(cudaFree(vec_output_values));
    checkCudaErrors(cudaFree(mat_slicing_offsets));

    // beamWeights_h regularization
    float beamWeights_max = 0.0f;
    for (int i=0; i<numBeams; i++)
        beamWeights_max = std::max(beamWeights_max, beamWeights_h[i]);
    for (int i=0; i<numBeams; i++) {
        beamWeights_h[i] /= beamWeights_max;
        beamWeights_h[i] = std::max(beamWeights_h[i], 0.1f);
        beamWeights_h[i] *= params.beamWeight;
    }

    #if false
        // for debug purposes
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

    checkCusparse(cusparseDestroy(handle));
    return 0;
}