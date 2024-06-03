#include <string>
#include <tuple>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "IMRTOptimize.cuh"
#include "IMRTInit.cuh"
#include "IMRTDoseMatEigen.cuh"
#include "IMRTOptimize_var.cuh"
#include "IMRTArgs.h"
#include "IMRTDebug.cuh"

bool IMRT::BOO_IMRT_L2OneHalf_gpu_QL (
    const MatCSR64& A, const MatCSR64& ATrans,
    const MatCSR64& D, const MatCSR64& DTrans,
    // parameters
    array_1d<float>& beamWeights, const array_1d<float>& maxDose,
    const array_1d<float>& minDoseTarget, const array_1d<float>& minDoseTargetWeights,
    const array_1d<float>& maxWeightsLong, const array_1d<float>& OARWeightsLong,
    size_t numBeamletsPerBeam, float gamma, float eta, int showTrigger, int changeWeightsTrigger,
    // variable
    int& k_global, int iters_global, const std::vector<int>& pruneTrigger, int numBeamsWeWant,
    float& theta_km1, float& tkm1, array_1d<float>& xkm1, array_1d<float>& vkm1,
    MatCSR64& x2d, MatCSR64& x2dprox,
    // for result logging
    std::vector<float>& loss_cpu, std::vector<float>& nrm_cpu,
    float& numActiveBeamsStrict, bool& stop
) {
    stop = false;
    size_t ptv_voxels = minDoseTarget.size;
    size_t oar_voxels = OARWeightsLong.size;
    size_t d_rows = D.numRows;
    size_t numBeams = beamWeights.size;
    size_t numBeamlets = xkm1.size;

    cusparseHandle_t handle_cusparse;
    checkCusparse(cusparseCreate(&handle_cusparse));
    cublasHandle_t handle_cublas;
    checkCublas(cublasCreate(&handle_cublas));

    eval_grad operator_eval_grad(ptv_voxels, oar_voxels, d_rows, numBeamlets);
    eval_g operator_eval_g(ptv_voxels, oar_voxels, d_rows);
    proxL2Onehalf_QL_gpu operator_proxL2Onehalf;
    operator_proxL2Onehalf.customInit(x2d, handle_cusparse);

    array_1d<float> x, y, x_minus_y, v, in, gradAty;
    std::vector<array_1d<float>*> array_pointers{&x, &y, &x_minus_y, &v, &in, &gradAty};
    for (auto a : array_pointers)
        arrayInit(*a, numBeamlets);

    array_1d<float> nrm, nrm_sqrt, t_times_beamWeights, activeBeamsStrict;
    arrayInit(nrm, numBeams);
    arrayInit(nrm_sqrt, numBeams);
    arrayInit(t_times_beamWeights, numBeams);
    arrayInit(activeBeamsStrict, numBeams);

    // find the interval
    int iters_local = iters_global;
    for (int i=0; i<pruneTrigger.size(); i++)
        if (pruneTrigger[i] > k_global) {
            iters_local = pruneTrigger[i] - k_global;
            break;
        }
    
    int k_local = 0;
    float reductionFactor = 0.5f;
    float t = tkm1, theta, gx, gy, rhs;
    float gradAty_dot_x_minus_y, x_minus_y_norm_square;
    float cost;
    for (; k_global<iters_global && k_local<iters_local; k_global++, k_local++) {
        if (k_global < 50 || k_global % 5 == 0)
            t /= reductionFactor;
        bool accept_t = false;
        while (! accept_t) {
            if (k_global > 1) {
                float a = tkm1;
                float b = t * theta_km1 * theta_km1;
                float c = -b;
                theta = (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
                linearComb_array_1d(1.0f-theta, xkm1, theta, vkm1, y);
            } else {
                theta = 1.0f;
                y.copy(xkm1);
            }
            gy = operator_eval_grad.evaluate(
                A, ATrans, D, DTrans, y, gradAty, gamma, handle_cusparse, handle_cublas,
                maxDose.data, minDoseTarget.data, minDoseTargetWeights.data,
                maxWeightsLong.data, OARWeightsLong.data, eta);

            linearComb_array_1d(1.0f, y, -t, gradAty, in);
            arrayToMatScatter(in, x2d);
            elementWiseMax(x2d, 0.0f);
            elementWiseScale(beamWeights.data, t_times_beamWeights.data, t, numBeams);

            operator_proxL2Onehalf.evaluate(
                x2d, t_times_beamWeights, x2dprox, nrm, handle_cusparse);
            
            checkCudaErrors(cudaMemcpy(x.data, x2dprox.d_csr_values,
                numBeamlets*sizeof(float), cudaMemcpyDeviceToDevice));
            
            gx = operator_eval_g.evaluate(
                A, D, x, gamma, handle_cusparse, handle_cublas,
                maxDose.data, minDoseTarget.data, minDoseTargetWeights.data,
                maxWeightsLong.data, OARWeightsLong.data, eta);
            
            // x_minus_y = x - y
            linearComb_array_1d(1.0f, x, -1.0f, y, x_minus_y);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCublas(cublasSdot(handle_cublas, numBeamlets,
                gradAty.data, 1, x_minus_y.data, 1, &gradAty_dot_x_minus_y));
            checkCublas(cublasSdot(handle_cublas, numBeamlets,
                x_minus_y.data, 1, x_minus_y.data, 1, &x_minus_y_norm_square));
            rhs = gy + gradAty_dot_x_minus_y + (0.5f / t) * x_minus_y_norm_square;
            if (gx <= rhs)
                accept_t = true;
            else
                t *= reductionFactor;
        }

        float One_over_theta = 1.0f / theta;
        linearComb_array_1d(One_over_theta, x, (1.0f - One_over_theta), xkm1, v);

        theta_km1 = theta;
        tkm1 = t;
        xkm1.copy(x);
        vkm1.copy(v);

        // nrm_sqrt = sqrt(nrm)
        elementWiseSqrt(nrm.data, nrm_sqrt.data, numBeams);
        // nrm_sqrt = beamWeights * nrm_sqrt
        elementWiseMul(beamWeights.data, nrm_sqrt.data, nrm_sqrt.data, 1.0f, numBeams);
        // cost = sum(nrm_sqrt)
        checkCublas(cublasSasum(handle_cublas, numBeams, nrm_sqrt.data, 1, &cost));
        // cost = cost + gx
        cost += gx;

        loss_cpu[k_global] = cost;
        
        elementWiseGreater(nrm.data, activeBeamsStrict.data, 1e-4f, numBeams);
        checkCublas(cublasSasum(handle_cublas, numBeams,
            activeBeamsStrict.data, 1, &numActiveBeamsStrict));
        if (numActiveBeamsStrict - eps_fastdose <= numBeamsWeWant) {
            stop = true;
            break;
        }
        #if false
        else if (abs(numActiveBeamsStrict - numBeamsWeWant - 1) < eps_fastdose) {
            if (abs(loss_cpu[k_global] - loss_cpu[k_global-1]) < 1e-5f * loss_cpu[k_global]) {
                stop = true;
                break;
            }
        } else if (numActiveBeamsStrict <= numBeamsWeWant * 1.05f)
            if (abs(loss_cpu[k_global] - loss_cpu[k_global-1]) < 1e-7f * loss_cpu[k_global]) {
                stop = true;
                break;
            }
        #endif
        if ((k_global + 1) % showTrigger == 0) {
            std::cout << "Iteration: " << k_global << ", cost: " << cost
                << ", t: " << t << ", numActiveBeamsStrict: " << (int)numActiveBeamsStrict
                << std::endl;
        }

        if ((k_global + 1) % changeWeightsTrigger == 0) {
            if (numActiveBeamsStrict >= 2 * numBeamsWeWant) {
                elementWiseScale(beamWeights.data, 2.0f, numBeams);
            } else if (numActiveBeamsStrict >= 1.5f * numBeamsWeWant) {
                elementWiseScale(beamWeights.data, 1.5f, numBeams);
            } else if (numActiveBeamsStrict >= 1.05f * numBeamsWeWant) {
                elementWiseScale(beamWeights.data, 1.2f, numBeams);
            } else if (numActiveBeamsStrict < numBeamsWeWant) {
                elementWiseScale(beamWeights.data, 0.3333f, numBeams);
            }
        }
    }
    if (k_global == iters_global)
        stop = true;
    
    if (nrm_cpu.size() != nrm.size) {
        std::cerr << "nrm_cpu size is supposed to be equal to nrm size." << std::endl;
        return 1;
    }
    checkCudaErrors(cudaMemcpy(nrm_cpu.data(), nrm.data,
        numBeams*sizeof(float), cudaMemcpyDeviceToHost));
    #if false
    // for debug purposes
        std::cout << "End of optimization GPU\ngx:" << gx << "\nnrm:\n\n" << nrm << "\n\n"
            << "x:\n" << x << std::endl;
    #endif
    return 0;
}


bool IMRT::BeamOrientationOptimization(
    const std::vector<MatCSR_Eigen>& VOIMatrices,
    const std::vector<MatCSR_Eigen>& VOIMatricesT,
    const std::vector<MatCSR_Eigen>& SpFluenceGrad,
    const std::vector<MatCSR_Eigen>& SpFluenceGradT,
    const Weights_h& weights_h, const Params& params_h,
    const std::vector<uint8_t>& fluenceArray, 
    std::vector<float>& costs, std::vector<int>& activeBeams,
    std::vector<float>& activeNorms
) {
    size_t numBeams = VOIMatrices.size();
    if (fluenceArray.size() % numBeams != 0) {
        std::cerr << "fluenceArray size should be a multiple of numBeams." << std::endl;
        return 1;
    }
    size_t numBeamletsPerBeam = fluenceArray.size() / numBeams;
    if (VOIMatricesT.size() != numBeams ||
        SpFluenceGrad.size() != numBeams ||
        SpFluenceGradT.size() != numBeams) {
        std::cerr << "The sizes of VOIMatrices, VOIMatricesT, SpFluenceGrad, "
            "SpFluenceGradT should be equal" << std::endl;
        return 1;
    }
    activeBeams.resize(numBeams);
    for (size_t i=0; i<numBeams; i++)
        activeBeams[i] = i;

    std::vector<const MatCSR_Eigen*> VOIMatrices_ptr(numBeams, nullptr);
    std::vector<const MatCSR_Eigen*> VOIMatricesT_ptr(numBeams, nullptr);
    std::vector<const MatCSR_Eigen*> SpFluenceGrad_ptr(numBeams, nullptr);
    std::vector<const MatCSR_Eigen*> SpFluenceGradT_ptr(numBeams, nullptr);
    size_t numBeamlets = 0;
    for (size_t i=0; i<numBeams; i++) {
        VOIMatrices_ptr[i] = &VOIMatrices[i];
        numBeamlets += VOIMatrices[i].getCols();
    }
    
    // prepare CPU weights
    Eigen::VectorXf xkm1_cpu(numBeamlets), vkm1_cpu(numBeamlets),
        beamWeights_cpu(numBeams);
    beamWeightsInit_func(VOIMatrices_ptr, beamWeights_cpu,
        weights_h.voxels_PTV, weights_h.voxels_OAR);
    beamWeights_cpu *= params_h.beamWeight;
    std::srand(10086);
    for (size_t i=0; i<numBeamlets; i++)
        xkm1_cpu(i) = (float)std::rand() / RAND_MAX;
    vkm1_cpu = xkm1_cpu;

    // prepare constant GPU weights
    array_1d<float> maxDose, minDoseTarget,
        minDoseTargetWeights, maxWeightsLong, OARWeightsLong;
    std::vector<array_1d<float>*> weight_targets{&maxDose, &minDoseTarget,
        &minDoseTargetWeights, &maxWeightsLong, &OARWeightsLong};
    std::vector<const std::vector<float>*> weight_sources{&weights_h.maxDose,
        &weights_h.minDoseTarget, &weights_h.minDoseTargetWeights,
        &weights_h.maxWeightsLong, &weights_h.OARWeightsLong};
    for (int i=0; i<weight_targets.size(); i++) {
        array_1d<float>& target = *weight_targets[i];
        const std::vector<float>& source = *weight_sources[i];
        size_t local_size = source.size();
        arrayInit(target, local_size);
        checkCudaErrors(cudaMemcpy(target.data, source.data(),
            local_size*sizeof(float), cudaMemcpyHostToDevice));
    }

    // prepare scalars
    float theta_km1, tkm1 = params_h.stepSize;
    int k_global = 0;

    #if slicingTiming
        std::chrono::_V2::system_clock::time_point time0, time1, time2;
        std::chrono::milliseconds duration;
        time0 = std::chrono::high_resolution_clock::now();
    #endif
    std::cout << std::scientific << "\n\nBeam Orientation Optimization starts." << std::endl;
    while (true) {
        #if slicingTiming
            time1 = std::chrono::high_resolution_clock::now();
        #endif
        // generate the matrices according to activeBeams
        size_t numActiveBeams = activeBeams.size();
        VOIMatrices_ptr.resize(numActiveBeams);
        VOIMatricesT_ptr.resize(numActiveBeams);
        SpFluenceGrad_ptr.resize(numActiveBeams);
        SpFluenceGradT_ptr.resize(numActiveBeams);
        for (size_t i=0; i<numActiveBeams; i++) {
            size_t beamIdx = activeBeams[i];
            VOIMatrices_ptr[i] = &VOIMatrices[beamIdx];
            VOIMatricesT_ptr[i] = &VOIMatricesT[beamIdx];
            SpFluenceGrad_ptr[i] = &SpFluenceGrad[beamIdx];
            SpFluenceGradT_ptr[i] = &SpFluenceGradT[beamIdx];
        }
        MatCSR_Eigen A_Eigen, ATrans_Eigen, D_Eigen, DTrans_Eigen;
        if (parallelMatCoalesce(A_Eigen, ATrans_Eigen, VOIMatrices_ptr, VOIMatricesT_ptr)) {
            std::cerr << "Error coaleasing the beam-wise dose loading matrices "
                "into a single matrix." << std::endl;
            return 1;
        }
        if (diagBlock(D_Eigen, SpFluenceGrad_ptr) ||
            diagBlock(DTrans_Eigen, SpFluenceGradT_ptr)) {
            std::cerr << "Error coaleasing the beam-wise fluence map gradient operators "
                "into a single matrix" << std::endl;
            return 1;
        }
        #if slicingTiming
            time2 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
            std::cout << "    Coaleasing beam-wise matrices time Elapsed: "
                << duration.count() * 1e-3f << " [s]\n";
        #endif

        // Loading CPU matrices to GPU
        MatCSR64 A, ATrans, D, DTrans;
        if (Eigen2Cusparse(A_Eigen, A) || Eigen2Cusparse(ATrans_Eigen, ATrans) ||
            Eigen2Cusparse(D_Eigen, D) || Eigen2Cusparse(DTrans_Eigen, DTrans)) {
            std::cerr << "Error loading CPU matrices to GPU" << std::endl;
            return 1;
        }
        array_1d<float> xkm1, vkm1, beamWeights;
        arrayInit(xkm1, xkm1_cpu);
        arrayInit(vkm1, vkm1_cpu);
        arrayInit(beamWeights, beamWeights_cpu);
        #if slicingTiming
            time1 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time2);
            std::cout << "    Loading matrices and vectors to GPU time elapsed: "
                << duration.count() * 1e-3f << " [s]\n";
        #endif

        // preparing x2d and x2dprox.
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor>
            x2d_dense_host(numActiveBeams, numBeamletsPerBeam);
        for (size_t i=0; i<numActiveBeams; i++) {
            size_t beamIdx = activeBeams[i];
            size_t offset_j = beamIdx * numBeamletsPerBeam;
            for (size_t j=0; j<numBeamletsPerBeam; j++)
                x2d_dense_host(i, j) = fluenceArray[offset_j + j] > 0;
        }
        MatCSR_Eigen x2d_sparse_host;
        Eigen::SparseMatrix<float, Eigen::RowMajor, EigenIdxType>
            *x2d_sparse_host_ptr = &x2d_sparse_host;
        *x2d_sparse_host_ptr = x2d_dense_host.sparseView().pruned();
        MatCSR64 x2d, x2dprox;
        Eigen2Cusparse(x2d_sparse_host, x2d);
        Eigen2Cusparse(x2d_sparse_host, x2dprox);

        activeNorms.resize(numActiveBeams);
        #if slicingTiming
            time2 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
            std::cout << "    Preparing x2d, x2dprox time elapsed: "
                << duration.count() * 1e-3f << " [s]" << std::endl;
        #endif

        // begin optimization
        float numActiveBeamsStrict = 0;
        bool stop = false;
        costs.resize(params_h.maxIter);
        if (BOO_IMRT_L2OneHalf_gpu_QL (
            A, ATrans, D, DTrans,
            beamWeights, maxDose, minDoseTarget, minDoseTargetWeights,
            maxWeightsLong, OARWeightsLong, numBeamletsPerBeam,
            params_h.gamma, params_h.eta, params_h.showTrigger, params_h.changeWeightsTrigger,
            k_global, params_h.maxIter, params_h.pruneTrigger, params_h.numBeamsWeWant,
            theta_km1, tkm1, xkm1, vkm1, x2d, x2dprox,
            costs, activeNorms, numActiveBeamsStrict, stop)) {
            std::cerr << "BOO_IMRT_L2OneHalf_gpu_QL error." << std::endl;
            return 1;
        }
        // beam reduction following.
        DimReduction(activeBeams, beamWeights_cpu, xkm1_cpu, vkm1_cpu,
            beamWeights, xkm1, vkm1, activeNorms, numActiveBeamsStrict, VOIMatrices);
        if (stop)
            break;
    }

    std::cout << "\n\nBeam Orientation Optimization finished." << std::endl;
    #if slicingTiming
        time1 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Optimization iterations: " << k_global + 1 << ", time elapsed: "
            << duration.count() * 1e-3f << " [s]" << std::endl;
    #endif

    std::cout << "Selected beams:\n";
    for (int i=0; i<activeBeams.size(); i++) {
        std::cout << activeBeams[i] << "  ";
    }
    std::cout << "\n\n";

    return 0;
}


bool IMRT::polish_BOO_IMRT_gpu(
    const MatCSR64& A, const MatCSR64& ATrans,
    const MatCSR64& D, const MatCSR64& DTrans,
    // parameters
    const array_1d<float>& maxDose,
    const array_1d<float>& minDoseTarget, const array_1d<float>& minDoseTargetWeights,
    const array_1d<float>& maxWeightsLong, const array_1d<float>& OARWeightsLong,
    float gamma, float eta, int& k_global, int iters_global, int iters_local,
    float& theta_km1, float& tkm1, array_1d<float>& xkm1, array_1d<float>& vkm1,
    std::vector<float>& loss_cpu
) {
    size_t ptv_voxels = minDoseTarget.size;
    size_t oar_voxels = OARWeightsLong.size;
    size_t d_rows = D.numRows;
    size_t numBeamlets = xkm1.size;

    cusparseHandle_t handle_cusparse;
    checkCusparse(cusparseCreate(&handle_cusparse));
    cublasHandle_t handle_cublas;
    checkCublas(cublasCreate(&handle_cublas));

    eval_grad operator_eval_grad(ptv_voxels, oar_voxels, d_rows, numBeamlets);
    eval_g operator_eval_g(ptv_voxels, oar_voxels, d_rows);

    array_1d<float> x, y, x_minus_y, v, gradAty;
    std::vector<array_1d<float>*> array_pointers{&x, &y, &x_minus_y, &v, &gradAty};
    for (auto a : array_pointers)
        arrayInit(*a, numBeamlets);

    int k_local = 0;
    float reductionFactor = 0.5f;
    float t = tkm1, theta, gx, gy, rhs;
    float gradAty_dot_x_minus_y, x_minus_y_norm_square;
    for (; k_global<iters_global && k_local<iters_local; k_global++, k_local++) {
        if (k_global < 50 || k_global % 5 == 0)
            t /= reductionFactor;
        bool accept_t = false;
        while (! accept_t) {
            if (k_global > 1) {
                float a = tkm1;
                float b = t * theta_km1 * theta_km1;
                float c = -b;
                theta = (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
                linearComb_array_1d(1.0f-theta, xkm1, theta, vkm1, y);
            } else {
                theta = 1.0f;
                y.copy(xkm1);
            }
            gy = operator_eval_grad.evaluate(
                A, ATrans, D, DTrans, y, gradAty, gamma, handle_cusparse, handle_cublas,
                maxDose.data, minDoseTarget.data, minDoseTargetWeights.data,
                maxWeightsLong.data, OARWeightsLong.data, eta);

            linearComb_array_1d(1.0f, y, -t, gradAty, x);
            elementWiseMax(x, 0.0f);

            gx = operator_eval_g.evaluate(
                A, D, x, gamma, handle_cusparse, handle_cublas,
                maxDose.data, minDoseTarget.data, minDoseTargetWeights.data,
                maxWeightsLong.data, OARWeightsLong.data, eta);

            linearComb_array_1d(1.0f, x, -1.0f, y, x_minus_y);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCublas(cublasSdot(handle_cublas, numBeamlets,
                gradAty.data, 1, x_minus_y.data, 1, &gradAty_dot_x_minus_y));
            checkCublas(cublasSdot(handle_cublas, numBeamlets,
                x_minus_y.data, 1, x_minus_y.data, 1, &x_minus_y_norm_square));
            rhs = gy + gradAty_dot_x_minus_y + (0.5f / t) * x_minus_y_norm_square;
            if (gx <= rhs)
                accept_t = true;
            else
                t *= reductionFactor;
        }

        float One_over_theta = 1.0f / theta;
        linearComb_array_1d(One_over_theta, x, (1.0f - One_over_theta), xkm1, v);

        theta_km1 = theta;
        tkm1 = t;
        xkm1.copy(x);
        vkm1.copy(v);

        loss_cpu[k_global] = gx;
    }
    return 0;
}


bool IMRT::FluencePolish(
    const std::vector<MatCSR_Eigen>& VOIMatrices,
    const std::vector<MatCSR_Eigen>& VOIMatricesT,
    const std::vector<MatCSR_Eigen>& SpFluenceGrad,
    const std::vector<MatCSR_Eigen>& SpFluenceGradT,
    const Weights_h& weights_h, const Params& params_h,
    const std::vector<uint8_t>& fluenceArray,
    const std::vector<int>& activeBeams,
    std::vector<float>& costs, Eigen::VectorXf& xFull
) {
    size_t numBeams = activeBeams.size();

    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    // Coalescing active beam-wise matrices
    MatCSR_Eigen A_Eigen, ATrans_Eigen, D_Eigen, DTrans_Eigen;
    std::vector<const MatCSR_Eigen*> VOIMatrices_ptr(numBeams, nullptr);
    std::vector<const MatCSR_Eigen*> VOIMatricesT_ptr(numBeams, nullptr);
    for (int i=0; i<numBeams; i++) {
        size_t beamIdx = activeBeams[i];
        VOIMatrices_ptr[i] = &VOIMatrices[beamIdx];
        VOIMatricesT_ptr[i] = &VOIMatricesT[beamIdx];
    }
    if (parallelMatCoalesce(A_Eigen, ATrans_Eigen, VOIMatrices_ptr, VOIMatricesT_ptr)) {
        std::cerr << "Error coaleasing the beam-wise dose loading matrices "
            "into a single matrix." << std::endl;
        return 1;
    }
    std::vector<const MatCSR_Eigen*> SpFluenceGrad_ptr(numBeams, nullptr);
    std::vector<const MatCSR_Eigen*> SpFluenceGradT_ptr(numBeams, nullptr);
    for (size_t i=0; i<numBeams; i++) {
        size_t beamIdx = activeBeams[i];
        SpFluenceGrad_ptr[i] = &SpFluenceGrad[beamIdx];
        SpFluenceGradT_ptr[i] = &SpFluenceGradT[beamIdx];
    }
    if (diagBlock(D_Eigen, SpFluenceGrad_ptr) ||
        diagBlock(DTrans_Eigen, SpFluenceGradT_ptr)) {
        std::cerr << "Error coalescing the beam-wise fluence map gradient operators "
            "into a single matrix" << std::endl;
        return 1;
    }
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0);
        std::cout << "Fluence polish coaleasing beam-wise matrices time elapsed: "
            << duration.count() * 1e-3f << " [s]" << std::endl;
    #endif

    // Loading CPU matrices to GPU
    MatCSR64 A, ATrans, D, DTrans;
    if (Eigen2Cusparse(A_Eigen, A) || Eigen2Cusparse(ATrans_Eigen, ATrans) ||
        Eigen2Cusparse(D_Eigen, D) || Eigen2Cusparse(DTrans_Eigen, DTrans)) {
        std::cerr << "Error loading CPU matrices to GPU" << std::endl;
        return 1;
    }
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
        std::cout << "Fluence polish loading CPU matrices to GPU time elapsed: "
            << duration.count() * 1e-3f << " [s]" << std::endl;
    #endif

    // prepare other weights
    array_1d<float> maxDose, minDoseTarget,
        minDoseTargetWeights, maxWeightsLong, OARWeightsLong;
    std::vector<array_1d<float>*> weight_targets{&maxDose, &minDoseTarget,
        &minDoseTargetWeights, &maxWeightsLong, &OARWeightsLong};
    std::vector<const std::vector<float>*> weight_sources{&weights_h.maxDose,
        &weights_h.minDoseTarget, &weights_h.minDoseTargetWeights,
        &weights_h.maxWeightsLong, &weights_h.OARWeightsLong};
    for (int i=0; i<weight_targets.size(); i++) {
        array_1d<float>& target = *weight_targets[i];
        const std::vector<float>& source = *weight_sources[i];
        size_t local_size = source.size();
        arrayInit(target, local_size);
        checkCudaErrors(cudaMemcpy(target.data, source.data(),
            local_size*sizeof(float), cudaMemcpyHostToDevice));
    }
    #if slicingTiming
        auto time3 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2);
        std::cout << "Weights initialization time elapsed: "
            << duration.count() * 1e-3f << " [s]" << std::endl;
    #endif

    // optimizing parameters
    costs.resize(params_h.maxIter);
    size_t numBeamlets = A_Eigen.getCols();
    int k_global = 0;
    float theta_km1, tkm1 = params_h.stepSize;
    array_1d<float> xkm1, vkm1;
    arrayInit(xkm1, numBeamlets);
    arrayInit(vkm1, numBeamlets);
    arrayRand01(xkm1);
    vkm1.copy(xkm1);

    // begin optimization
    std::cout << std::scientific << "Fluence map polishing starts." << std::endl;
    if (polish_BOO_IMRT_gpu(
        A, ATrans, D, DTrans,
        maxDose, minDoseTarget,
        minDoseTargetWeights, maxWeightsLong, OARWeightsLong,
        params_h.gamma, params_h.eta, k_global, params_h.maxIter,
        params_h.maxIter, theta_km1, tkm1, xkm1, vkm1, costs)) {
        std::cerr << "polish_BOO_IMRT_cpu error." << std::endl;
        return 1;
    }
    std::cout << "Fluence map polish finisehd." << std::endl;
    #if slicingTiming
        auto time4 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3);
        std::cout << "Optimization iterations: " << k_global << ", time elapsed: "
            << duration.count() * 1e-3f << " [s]" << std::endl;
    #endif
    std::cout << "\n\n";

    xFull.resize(numBeamlets);
    checkCudaErrors(cudaMemcpy(xFull.data(), xkm1.data,
        numBeamlets*sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}


bool IMRT::resultDoseCalc(
    const std::vector<MatCSR_Eigen>& MatricesT_full,
    const std::vector<int>& activeBeams,
    const Eigen::VectorXf& xFull,
    Eigen::VectorXf& finalDose) {
    size_t numVoxels = MatricesT_full[0].getCols();
    finalDose.resize(numVoxels);
    for (size_t i=0; i<numVoxels; i++)
        finalDose(i) = 0.0f;
    
    size_t numBeams = activeBeams.size();
    size_t starting_idx = 0;
    for (size_t i=0; i<numBeams; i++) {
        size_t idx = activeBeams[i];
        const MatCSR_Eigen& currentMat = MatricesT_full[idx];
        size_t currentCols = currentMat.getRows();
        Eigen::VectorXf xLocal = xFull.segment(starting_idx, currentCols);
        starting_idx += currentCols;
        finalDose += currentMat.transpose() * xLocal;
    }
    return 0;
}


bool IMRT::writeResults(const std::vector<int>& activeBeams,
    const std::vector<MatCSR_Eigen>& MatricesT_full,
    const Eigen::VectorXf& xFull, const Eigen::VectorXf& finalDose) {
    const std::string& planFolder = getarg<std::string>("planFolder");
    fs::path outputFolder;
    if (planFolder == "") {
        if (getarg<std::vector<std::string>>("outputFolder").size() != 1) {
            std::cerr << "Only one entry is expected in the argument outputFolder." << std::endl;
            return 1;
        }
        outputFolder = getarg<std::vector<std::string>>("outputFolder")[0];
    } else {
        outputFolder = planFolder;
    }
    const std::vector<int>& phantomDim = getarg<std::vector<int>>("phantomDim");
    std::stringstream metadata__;
    metadata__ << "Number of candidate beams: " << MatricesT_full.size()
        << "\nPhantom dimension: (" << phantomDim[0] << ", "
        << phantomDim[1] << ", " << phantomDim[2] << ")";
    metadata__ << "\nBeams selected:\n";
    for (size_t i=0; i<activeBeams.size(); i++)
        metadata__ << activeBeams[i] << "  ";
    metadata__ << "\nNumber of beamlets each:\n";
    for (size_t i=0; i<activeBeams.size(); i++) {
        size_t beamIdx = activeBeams[i];
        const MatCSR_Eigen& matT = MatricesT_full[beamIdx];
        metadata__ << matT.getRows() << "  ";
    }
    metadata__ << std::endl;

    std::string metadata = metadata__.str();
    fs::path metadataPath = outputFolder / std::string("metadata.txt");
    std::ofstream f(metadataPath.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file " << metadataPath << std::endl;
        return 1;
    }
    f << metadata;
    f.close();

    fs::path beamletWeightsPath = outputFolder / std::string("beamletWeights.bin");
    f.open(beamletWeightsPath.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file " << beamletWeightsPath << std::endl;
        return 1;
    }
    f.write((char*)xFull.data(), xFull.size()*sizeof(float));
    f.close();

    fs::path dosePath = outputFolder / std::string("dose.bin");
    f.open(dosePath.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file " << dosePath << std::endl;
        return 1;
    }
    f.write((char*)finalDose.data(), finalDose.size()*sizeof(float));
    f.close();

    return 0;
}