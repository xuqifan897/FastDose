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

#if false
bool IMRT::BOO_IMRT_L2OneHalf_gpu_QL(
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
    std::vector<std::pair<int, std::vector<int>>>& topN
) {
    int nBeams = VOIMatrices.size();
    // sanity check
    if (nBeams != VOIMatricesT.size() ||
        nBeams != SpFluenceGrad.size() ||
        nBeams != SpFluenceGradT.size()) {
        std::cerr << "The number of beams should be consistent among "
            "the input Eigen matrices." << std::endl;
        return 1;
    }

    cusparseHandle_t handle;
    checkCusparse(cusparseCreate(&handle));
    cublasHandle_t cublas_handle;
    checkCublas(cublasCreate(&cublas_handle));

    MatReservior VOIRes, VOIResT, FGRes, FGResT;
    VOIRes.load(VOIMatrices);
    VOIResT.load(VOIMatricesT);
    FGRes.load(SpFluenceGrad);
    FGResT.load(SpFluenceGradT);

    array_1d<float> beamWeights;
    arrayInit(beamWeights, nBeams);
    beamWeightsInit_func(VOIRes, beamWeights, weights_h.voxels_PTV,
        weights_h.voxels_OAR, handle, cublas_handle);
    elementWiseScale(beamWeights.data, params.beamWeight, beamWeights.size);

    // get some basic parameters
    size_t D_rows_max = 0, numBeamlets_max = 0;
    for (int i=0; i<nBeams; i++) {
        D_rows_max += FGRes.reservior[i].numRows;
        numBeamlets_max += VOIRes.reservior[i].numCols;
    }
    std::cout << std::scientific << "The maximum number of rows of matrix D: " << D_rows_max
        << "\nThe maximum number of beamlets: " << numBeamlets_max << "\n" << std::endl;

    MatCSR64 *A=nullptr, *ATrans=nullptr, *D=nullptr, *DTrans=nullptr;
    // of shape numBeamlets, which is changing
    array_1d<float> xkm1, vkm1, y, gradAty, in, x, v;
    // of shape nBeams, which is constant
    array_1d<float> nrm, beamWeights_active,
        beamWeights_activeStrict;
    // of shape (numBeamletsPerBeam, numBeams), constant shape
    MatCSR64 x2d, x2dprox;
    // weights
    Weights_d weights_d; weights_d.fromHost(weights_h);
    // operators
    eval_g operator_eval_g((size_t)weights_d.voxels_PTV, (size_t)weights_d.voxels_OAR, D_rows_max);
    eval_grad operator_eval_grad((size_t)weights_d.voxels_PTV, (size_t)weights_d.voxels_OAR,
        numBeamlets_max, D_rows_max);
    
    // create a list for easier management
    int fluenceDim = getarg<int>("fluenceDim");
    std::vector<array_1d<float>*> array_group1{&xkm1, &vkm1, &y, &gradAty, &in, &x, &v};
    std::vector<MatCSR64*> array_group2{&x2d, &x2dprox};
    arrayInit_group1(array_group1, numBeamlets_max);
    arrayInit_group2(array_group2, fluenceArray, nBeams, fluenceDim);
    arrayInit(nrm, nBeams);
    arrayInit(beamWeights_active, nBeams); arrayInit(beamWeights_activeStrict, nBeams);

    resize_group2 operator_resize_group2;

    // a vector containing the active beams
    std::vector<uint8_t> active_beams(nBeams, 1);
    // a vector containing the top beams of each iteration
    // in the format <iteration, topN at that iteration>
    topN.clear();
    
    if (DimensionReduction(active_beams,
        VOIRes, VOIMatrices,
        VOIResT, FGRes, FGResT,
        &A, &ATrans, &D, &DTrans,
        operator_eval_g, operator_eval_grad,
        array_group1, array_group2, operator_resize_group2,
        fluenceArray, fluenceDim, handle)) {
        std::cerr << "Dimension reduction error." << std::endl;
        return 1;
    }

    #if false
        // for debug purposes. To check whether the A and ATrans,
        // D and DTrans are consistent with each other.
        if (transposeConsistency(*A, *ATrans, handle)) {
            std::cerr << "A and ATrans inconsistent with each other." << std::endl;
            return 1;
        } else {
            std::cout << "A and ATrans consistent with each other!" << std::endl;
        }

        if (transposeConsistency(*D, *DTrans, handle)) {
            std::cerr << "D and DTrans inconsistent with each other." << std::endl;
            return 1;
        } else {
            std::cout << "D and DTrans consistent with each other!" << std::endl;
        }
        return 0;
    #endif
    
    float all_zero_cost = operator_eval_g.evaluate(
        *A, *D, vkm1, params.gamma, handle,
        weights_d.maxDose.data, weights_d.minDoseTarget.data,
        weights_d.minDoseTargetWeights.data, weights_d.maxWeightsLong.data,
        weights_d.OARWeightsLong.data, params.eta);
    std::cout << "all zero cost is: " << all_zero_cost << std::endl;
    copy_array_1d(vkm1, xkm1);  // vkm1 = xkm1;

    proxL2Onehalf_QL_gpu proxL2Onehalf_operator;

    calc_rhs operator_calc_rhs;
    operator_calc_rhs.customInit(numBeamlets_max);

    calc_loss operator_calc_loss;
    operator_calc_loss.customInit(nBeams);

    float t = params.stepSize;
    float reductionFactor = 0.5f;
    float a, b, c, theta, tkm1, theta_km1, gx;
    int pruneTrigger = 100;
    int currentActiveBeams = nBeams;
    // indicate weather the variable beamNorms_h is ready
    bool beamNorms_h_ready = false;
    
    // to store the loss values
    costs.clear(); costs.reserve(params.maxIter);
    std::vector<uint8_t> activeBeamsStrict(nBeams, 0);
    std::vector<float> beamNorms_h(nBeams);
    std::vector<std::pair<int, float>> beamNorms_pair(nBeams);

    // for timing
    auto time0 = std::chrono::high_resolution_clock::now();
    auto time1(time0);
    std::chrono::milliseconds duration;

    for (int k=0; k<params.maxIter; k++) {
        if (k < 50 || k % 5 == 0)
            t = t / reductionFactor; // attempt to increase t
        bool accept_t = false;
        while (! accept_t) {
            if (k>0) {
                a = tkm1;
                b = t * theta_km1 * theta_km1;
                c = - t * theta_km1 * theta_km1;

                theta = (-b + sqrtf(b * b - 4 * a * c)) / (2 * a);
                linearComb_array_1d(1.0f - theta, xkm1, theta, vkm1, y);
            } else {
                theta = 1.0f;
                copy_array_1d(y, xkm1);
            }
            float gy = operator_eval_grad.evaluate(
                *A, *ATrans, *D, *DTrans, y, gradAty, params.gamma, handle,
                weights_d.maxDose.data, weights_d.minDoseTarget.data,
                weights_d.minDoseTargetWeights.data, weights_d.maxWeightsLong.data,
                weights_d.OARWeightsLong.data, params.eta);

            linearComb_array_1d(1.0f, y, -t, gradAty, in);

            arrayToMatScatter(in, x2d);

            elementWiseMax(x2d, 0.0f);

            if (! proxL2Onehalf_operator.initFlag)
                proxL2Onehalf_operator.customInit(x2d);
            if (proxL2Onehalf_operator.evaluate(x2d, beamWeights, t,
                x2dprox, nrm, cublas_handle))
                return 1;

            checkCudaErrors(cudaMemcpy(x.data, x2dprox.d_csr_values,
                x2d.nnz*sizeof(float), cudaMemcpyDeviceToDevice));

            gx = operator_eval_g.evaluate(
                *A, *D, x, params.gamma, handle,
                weights_d.maxDose.data, weights_d.minDoseTarget.data,
                weights_d.minDoseTargetWeights.data, weights_d.maxWeightsLong.data,
                weights_d.OARWeightsLong.data, params.eta);
            
            float rhs;
            operator_calc_rhs.evaluate(rhs, gy, t, gradAty, x, y, cublas_handle);
            if (gx <= rhs)
                accept_t = true;
            else
                t *= reductionFactor;
        }
        
        float one_over_theta = 1.0f / theta;
        linearComb_array_1d(one_over_theta, x, 1.0f - one_over_theta, xkm1, v);

        theta_km1 = theta;
        tkm1 = t;
        copy_array_1d(xkm1, x);
        copy_array_1d(vkm1, v);

        /*  Now compute objective function value. */
        float cost;
        operator_calc_loss.evaluate(cost, gx, beamWeights, nrm, cublas_handle);
        costs.push_back(cost);
        #if BOO_IMRT_DEBUG
            std::cout << "iteration: " << k << ", cost: " << cost << ", theta: "
                << theta << std::endl;
        #endif
        elementWiseGreater(beamWeights.data, beamWeights_active.data, 1e-2f, nBeams);
        elementWiseGreater(beamWeights.data, beamWeights_activeStrict.data, 1e-6f, nBeams);
        float numActiveBeams = 0, numActiveBeamsStrict = 0;
        checkCublas(cublasSasum(cublas_handle, nBeams,
            beamWeights_active.data, 1, &numActiveBeams));
        checkCublas(cublasSasum(cublas_handle, nBeams,
            beamWeights_activeStrict.data, 1, &numActiveBeamsStrict));

        /*  Finished computing the objective function value.  */
        if ((k + 1) % params.changeWeightsTrigger == 0) {
            float factor = 1.0f;
            if (numActiveBeamsStrict >= 2 * params.numBeamsWeWant) {
                factor = 2.0f;
            } else if (numActiveBeamsStrict >= 1.5f * params.numBeamsWeWant) {
                factor = 1.5f;
            } else if (numActiveBeamsStrict >= 1.05f * params.numBeamsWeWant) {
                factor = 1.2f;
            } else if (numActiveBeamsStrict < params.numBeamsWeWant) {
                factor = 0.3333f;
            }
            elementWiseScale(beamWeights.data, factor, nBeams);
        }

        /*  Now throw out inactive beams */
        beamNorms_h_ready = false;
        if ((k+1 % pruneTrigger == 0) && (currentActiveBeams >
            numActiveBeamsStrict + 20)) {
            checkCudaErrors(cudaMemcpy(beamNorms_h.data(), nrm.data,
                nBeams*sizeof(float), cudaMemcpyDeviceToHost));
            beamNorms_h_ready = true;
            for (int i=0; i<nBeams; i++)
                activeBeamsStrict[i] = beamNorms_h[i] > 1e-6f;
            
            if (DimensionReduction(activeBeamsStrict,
                VOIRes, VOIMatrices,
                VOIResT, FGRes, FGResT,
                &A, &ATrans, &D, &DTrans,
                operator_eval_g, operator_eval_grad,
                array_group1, array_group2, operator_resize_group2,
                fluenceArray, fluenceDim, handle)) {
                std::cerr << "Dimension reduction error." << std::endl;
                return 1;
            }

            currentActiveBeams = numActiveBeamsStrict;
        }
        /*  Finished throwing out inactive beams  */
        if ((k+1) % params.showTrigger == 0) {
            // update top n beams
            if (! beamNorms_h_ready) {
                checkCudaErrors(cudaMemcpy(beamNorms_h.data(), nrm.data,
                    nBeams*sizeof(float), cudaMemcpyDeviceToHost));
                beamNorms_h_ready = true;
            }
            beamSort(beamNorms_h, beamNorms_pair);
            int numElements = min(20, (int)roundf(numActiveBeams));
            topN.push_back(std::pair<int, std::vector<int>>(k,
                std::vector<int>(numElements, 0)));
            for (int i=0; i<numElements; i++) {
                topN.back().second[i] = beamNorms_pair[i].first;
            }

            time1 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0);
            std::cout << "FISTA iteration is: " << k << ", cost: " << costs[k]
                << ", t: " << t << ", numActiveBeams: " << numActiveBeams
                << ", numActiveBeamsStrict: " << numActiveBeamsStrict
                << ", time: " << duration.count() * 0.001f << " [s], top beams:\n";
            const std::vector<int>& current_topN = topN.back().second;
            for (int i=0; i<current_topN.size(); i++)
                std::cout << i << "  ";
            std::cout << "\n" << std::endl;
        }

        if (numActiveBeamsStrict <= params.numBeamsWeWant) {
            break;
        } else if (numActiveBeamsStrict <= params.numBeamsWeWant + 1) {
            if (k >= 1 && abs(costs[k] - costs[k-1]) < 1e-5f * costs[k])
                break;
        } else if (numActiveBeamsStrict <= params.numBeamsWeWant * 1.05f) {
            if (k >= 1 && abs(costs[k] - costs[k-1]) < 1e-7f * costs[k])
                break;
        }
    }

    time1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0);
    std::cout << "\n\nOptimization finished. Time elapsed: "
        << duration.count()*0.001f << " [s]" << std::endl;

    // output results
    std::vector<float> xkm1_h(xkm1.size, 0.0f);
    checkCudaErrors(cudaMemcpy(xkm1_h.data(), xkm1.data,
        xkm1.size*sizeof(float), cudaMemcpyDeviceToHost));
    xFull.resize(numBeamlets_max);
    size_t xFull_idx = 0;
    size_t xkm1_h_idx = 0;
    for (size_t i=0; i<nBeams; i++) {
        size_t fluence_begin = i * fluenceDim * fluenceDim;
        size_t fluence_end = (i + 1) * fluenceDim * fluenceDim;
        if (activeBeamsStrict[i] > 0) {
            // beam i is an active beam
            for (size_t j=fluence_begin; j<fluence_end; j++) {
                xFull[xFull_idx] = xkm1_h[xkm1_h_idx];
                xFull_idx ++;
                xkm1_h_idx ++;
            }
        } else {
            // beam i in an inactive beam
            for (size_t j=fluence_begin; j<fluence_end; j++) {
                xFull_idx ++;
            }
        }
    }

    // update active beams as a result
    activeBeams = topN.back().second;
    activeNorms.resize(activeBeams.size());
    for (int i=0; i<activeBeams.size(); i++)
        activeNorms[i] = beamNorms_pair[i].second;

    // clean up
    checkCusparse(cusparseDestroy(handle));
    checkCublas(cublasDestroy(cublas_handle));
    return 0;
}
#endif


bool IMRT::BOO_IMRT_L2OneHalf_gpu_QL (
    const MatCSR64& A, const MatCSR64& ATrans,
    const MatCSR64& D, const MatCSR64& DTrans,
    // parameters
    const array_1d<float>& beamWeights, const array_1d<float>& maxDose,
    const array_1d<float>& minDoseTarget, const array_1d<float>& minDoseTargetWeights,
    const array_1d<float>& maxWeightsLong, const array_1d<float>& OARWeightsLong,
    size_t numBeamletsPerBeam, float gamma, float eta,
    // variable
    int& k_global, int iters_global, int iters_local, float& theta_km1, float& tkm1,
    array_1d<float>& xkm1, array_1d<float>& vkm1, MatCSR64& x2d, MatCSR64& x2dprox
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
            
            float gx = operator_eval_g.evaluate(
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

            #if false
            // for debug purposes
                std::cout << "Iteration: " << k_global << "\ny:\n" << y
                    << "\ngradAty:\n" << gradAty << "\nin:" << in
                    << "\nx:\n" << x << "\n\n" << std::endl;
            #endif
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

        #if true
        // for debug purposes
            std::cout << "Iteration: " << k_global << ", t: " << std::scientific << t
                << ", loss: " << cost << std::endl;
        #endif
    }
    #if true
    // for debug purposes
        std::cout << "final x:\n";
        std::vector<float> x_host(numBeamlets);
        checkCudaErrors(cudaMemcpy(x_host.data(), x.data,
            numBeamlets*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i=0; i<numBeams; i++) {
            size_t j_offset = i * numBeamletsPerBeam;
            for (size_t j=0; j<numBeamletsPerBeam; j++) {
                std::cout << x_host[j_offset + j] << "  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n\nfinal norm:\n";
        std::cout << nrm << std::endl;
    #endif
    return 0;
}