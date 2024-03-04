#include <string>
#include <tuple>
#include <iomanip>
#include <algorithm>
#include <chrono>
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
    const array_1d<float>& beamWeights, const array_1d<float>& maxDose,
    const array_1d<float>& minDoseTarget, const array_1d<float>& minDoseTargetWeights,
    const array_1d<float>& maxWeightsLong, const array_1d<float>& OARWeightsLong,
    size_t numBeamletsPerBeam, float gamma, float eta, int showTrigger,
    // variable
    int& k_global, int iters_global, int iters_local, float& theta_km1, float& tkm1,
    array_1d<float>& xkm1, array_1d<float>& vkm1, MatCSR64& x2d, MatCSR64& x2dprox,
    // for result logging
    std::vector<float>& loss_cpu, std::vector<float>& nrm_cpu
) {    
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

    array_1d<float> nrm, nrm_sqrt, t_times_beamWeights;
    arrayInit(nrm, numBeams);
    arrayInit(nrm_sqrt, numBeams);
    arrayInit(t_times_beamWeights, numBeams);

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
        
        if ((k_global + 1) % showTrigger == 0) {
            std::cout << "Iteration: " << k_global << ", cost: " << cost
                << ", t: " << t << std::endl;
        }
    }
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
    const Weights_h& weights_h, const Params& params_h, std::vector<uint8_t> fluenceArray, 
    std::vector<float>& xFull, std::vector<float>& costs, std::vector<int>& activeBeams,
    std::vector<float>& activeNorms, std::vector<std::pair<int, std::vector<int>>>& topN
) {
    // firstly, we ignore the dimension reduction
    size_t numBeams = VOIMatrices.size();
    if (VOIMatricesT.size() != numBeams ||
        SpFluenceGrad.size() != numBeams ||
        SpFluenceGradT.size() != numBeams) {
        std::cerr << "The sizes of VOIMatrices, VOIMatricesT, SpFluenceGrad, "
            "SpFluenceGradT should be equal" << std::endl;
        return 1;
    }

    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    // Coaleasing beam-wise matrices
    MatCSR_Eigen A_Eigen, ATrans_Eigen, D_Eigen, DTrans_Eigen;
    if (parallelMatCoalease(A_Eigen, ATrans_Eigen, VOIMatrices, VOIMatricesT)) {
        std::cerr << "Error coaleasing the beam-wise dose loading matrices "
            "into a single matrix." << std::endl;
        return 1;
    }
    std::vector<MatCSR_Eigen*> SpFluenceGrad_ptr(numBeams, nullptr);
    std::vector<MatCSR_Eigen*> SpFluenceGradT_ptr(numBeams, nullptr);
    for (size_t i=0; i<numBeams; i++) {
        SpFluenceGrad_ptr[i] = (MatCSR_Eigen*)&SpFluenceGrad[i];
        SpFluenceGradT_ptr[i] = (MatCSR_Eigen*)&SpFluenceGradT[i];
    }
    if (diagBlock(D_Eigen, SpFluenceGrad_ptr) ||
        diagBlock(DTrans_Eigen, SpFluenceGradT_ptr)) {
        std::cerr << "Error coaleasing the beam-wise fluence map gradient operators "
            "into a single matrix" << std::endl;
        return 1;
    }
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0);
        std::cout << "Coaleasing beam-wise matrices time elapsed: "
            << duration.count() * 1e-6f << " [s]" << std::endl;
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
        duration = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1);
        std::cout << "Loading CPU matrices to GPU time elapsed: "
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif

    // prepare beam weights
    std::vector<float> beamWeightsInit(numBeams, 0.0f);
    beamWeightsInit_func(VOIMatrices, beamWeightsInit,
        weights_h.voxels_PTV, weights_h.voxels_OAR);
    for (size_t i=0; i<numBeams; i++)
        beamWeightsInit[i] *= params_h.beamWeight;
    array_1d<float> beamWeights;
    arrayInit(beamWeights, numBeams);
    checkCudaErrors(cudaMemcpy(beamWeights.data, beamWeightsInit.data(),
        numBeams*sizeof(float), cudaMemcpyHostToDevice));

    #if false
    // for debug purposes
        std::cout << "Beam weights:\n";
        for (size_t i=0; i<numBeams; i++)
            std::cout << beamWeightsInit[i] << "  ";
        std::cout << std::endl;
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
        duration = std::chrono::duration_cast<std::chrono::microseconds>(time3-time2);
        std::cout << "Weights initialization time elapsed: "
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif

    // optimization parameters
    std::vector<float> loss_cpu(params_h.maxIter);
    std::vector<float> nrm_cpu(numBeams);
    size_t numBeamlets = A_Eigen.getCols();
    int k_global = 0;
    float theta_km1, tkm1=params_h.stepSize;
    array_1d<float> xkm1, vkm1;
    arrayInit(xkm1, numBeamlets);
    arrayInit(vkm1, numBeamlets);
    arrayRand01(xkm1);
    vkm1.copy(xkm1);

    // x2d and x2dprox
    if (fluenceArray.size() % numBeams != 0) {
        std::cerr << "fluenceArray size should be a multiple of numBeams." << std::endl;
        return 1;
    }
    size_t numBeamletsPerBeam = fluenceArray.size() / numBeams;
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor>
        x2d_dense_host(numBeams, numBeamletsPerBeam);
    for (size_t i=0; i<numBeams; i++) {
        size_t offset_j = i * numBeamletsPerBeam;
        for (size_t j=0; j<numBeamletsPerBeam; j++) {
            x2d_dense_host(i, j) = fluenceArray[offset_j + j] > 0;
        }
    }
    Eigen::SparseMatrix<float, Eigen::RowMajor, EigenIdxType>
        x2d_Eigen = x2d_dense_host.sparseView().pruned();
    MatCSR_Eigen* x2d_sparse = (MatCSR_Eigen*)&x2d_Eigen;
    #if false
    // for debug purposes
        std::cout << "x2d_sparse (rows, cols, nnz) = (" << x2d_sparse->getRows()
            << ", " << x2d_sparse->getCols() << ", " << x2d_sparse->getNnz() << ")" << std::endl;
    #endif
    MatCSR64 x2d, x2dprox;
    Eigen2Cusparse(*x2d_sparse, x2d);
    Eigen2Cusparse(*x2d_sparse, x2dprox);
    #if slicingTiming
        auto time4 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3);
        std::cout << "Variables initialization time elapsed: "\
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif

    // begin optimization
    std::cout << std::scientific << "\n\nOptimization starts." << std::endl;
    if (BOO_IMRT_L2OneHalf_gpu_QL(
        A, ATrans, D, DTrans,
        beamWeights, maxDose, minDoseTarget, minDoseTargetWeights,
        maxWeightsLong, OARWeightsLong, numBeamletsPerBeam,
        params_h.gamma, params_h.eta, 1,
        k_global, params_h.maxIter, params_h.maxIter, theta_km1, tkm1,
        xkm1, vkm1, x2d, x2dprox,
        loss_cpu, nrm_cpu)) {
        std::cerr << "BOO_IMRT_L2OneHalf_gpu_QL error." << std::endl;
        return 1;
    }
    std::cout << "\n\nOptimization finished." << std::endl;
    #if slicingTiming
        auto time5 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(time5 - time4);
        std::cout << "Optimization iterations: " << params_h.maxIter << ", time elapsed: "
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif
    return 0;
}