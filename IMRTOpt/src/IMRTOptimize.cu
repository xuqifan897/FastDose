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


bool IMRT::BOO_IMRT_L2OneHalf_gpu_QL(
    const std::vector<MatCSR_Eigen>& VOIMatrices,
    const std::vector<MatCSR_Eigen>& VOIMatricesT,
    const std::vector<MatCSR_Eigen>& SpFluenceGrad,
    const std::vector<MatCSR_Eigen>& SpFluenceGradT,
    const Weights_h& weights_h,
    const Params& params,
    const std::vector<uint8_t>& fluenceArray
) {
    int nBeams = VOIMatrices.size();
    // sanity check
    if (nBeams != VOIMatricesT.size() ||
        nBeams != SpFluenceGrad.size() ||
        nBeams != SpFluenceGradT.size()) {
        std::cerr << "The number of beams should be consistent through "
            "the input Eigen matrices." << std::endl;
        return 1;
    }

    cusparseHandle_t handle;
    checkCusparse(cusparseCreate(&handle));

    MatReservior VOIRes, VOIResT, FGRes, FGResT;
    VOIRes.load(VOIMatrices);
    VOIResT.load(VOIMatricesT);
    FGRes.load(SpFluenceGrad);
    FGResT.load(SpFluenceGradT);

    array_1d<float> beamWeights;
    arrayInit(beamWeights, nBeams);
    beamWeightsInit_func(VOIRes, beamWeights, weights_h.voxels_PTV,
        weights_h.voxels_OAR, handle);
    elementWiseScale(beamWeights.data, params.beamWeight, beamWeights.size);

    // get some basic parameters
    size_t D_rows_max = 0, numBeamlets_max = 0;
    for (int i=0; i<nBeams; i++) {
        D_rows_max += FGRes.reservior[i].numRows;
        numBeamlets_max += VOIRes.reservior[i].numCols;
    }
    std::cout << "The maximum number of rows of matrix D: " << D_rows_max
        << "\nThe maximum number of beamlets: " << numBeamlets_max << "\n" << std::endl;

    MatCSR64 *A=nullptr, *ATrans=nullptr, *D=nullptr, *DTrans=nullptr;
    // of shape numBeamlets, which is changing
    array_1d<float> xkm1, vkm1, y, gradAty, in, x, v;
    // of shape nBeams, which is constant
    array_1d<float> nrm;
    // of shape (numBeamletsPerBeam, numBeams), constant shape
    MatCSR64 x2d, x2dprox;
    // weights
    Weights_d weights_d; weights_d.fromHost(weights_h);
    // operators
    eval_g operator_eval_g((size_t)weights_d.voxels_PTV, (size_t)weights_h.voxels_OAR, D_rows_max);
    eval_grad operator_eval_grad((size_t)weights_d.voxels_PTV, (size_t)weights_d.voxels_OAR,
        numBeamlets_max, D_rows_max);
    
    // create a list for easier management
    int fluenceDim = getarg<int>("fluenceDim");
    std::vector<array_1d<float>*> array_group1{&xkm1, &vkm1, &y, &gradAty, &in, &x, &v};
    std::vector<MatCSR64*> array_group2{&x2d, &x2dprox};
    arrayInit_group1(array_group1, numBeamlets_max);
    arrayInit_group2(array_group2, fluenceArray, nBeams, fluenceDim);
    arrayInit(nrm, nBeams);

    resize_group2 operator_resize_group2;

    // a vector containing the active beams
    std::vector<uint8_t> active_beams(nBeams, 1);
    // a vector containing the top beams of each iteration
    std::vector<std::vector<int>> topN(nBeams);
    
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
    #endif
    return 0;
    
    float all_zero_cost = operator_eval_g.evaluate(
        *A, *D, vkm1, params.gamma, handle,
        weights_d.maxDose.data, weights_d.minDoseTarget.data,
        weights_d.minDoseTargetWeights.data, weights_d.maxWeightsLong.data,
        weights_d.OARWeightsLong.data, params.eta);
    std::cout << "all zero cost is: " << all_zero_cost << std::endl;

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

    array_1d<float> beamWeights;
    arrayInit(beamWeights, nBeams);
    
    // to store the loss values
    std::vector<float> costs(params.maxIter, 0.0f);
    std::vector<uint8_t> activeBeamsStrict(nBeams, 0);
    std::vector<float> beamNorms_h(nBeams);

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
            proxL2Onehalf_operator.evaluate(x2d, beamWeights, t, x2dprox, nrm);

            checkCudaErrors(cudaMemcpy(x.data, x2dprox.d_csr_values,
                x2d.nnz, cudaMemcpyDeviceToDevice));

            gx = operator_eval_g.evaluate(
                *A, *D, x, params.gamma, handle,
                weights_d.maxDose.data, weights_d.minDoseTarget.data,
                weights_d.minDoseTargetWeights.data, weights_d.maxWeightsLong.data,
                weights_d.OARWeightsLong.data, params.eta);
            
            float rhs;
            operator_calc_rhs.evaluate(rhs, gy, t, gradAty, x, y);
            if (gx <= rhs)
                accept_t = true;
            else
                t += reductionFactor;
        }
        
        float one_over_theta = 1.0f / theta;
        linearComb_array_1d(one_over_theta, x, 1.0f - one_over_theta, xkm1, v);

        theta_km1 = theta;
        tkm1 = t;
        copy_array_1d(xkm1, x);
        copy_array_1d(vkm1, v);

        /*  Now compute objective function value. */
        float cost;
        operator_calc_loss.evaluate(cost, gx, beamWeights, nrm);
        costs[k] = cost;
        checkCudaErrors(cudaMemcpy(beamNorms_h.data(), nrm.data,
            nBeams*sizeof(float), cudaMemcpyDeviceToHost));

        int numActiveBeams = 0;
        int numActiveBeamsStrict = 0;
        for (int i=0; i<nBeams; i++) {
            numActiveBeams += (beamNorms_h[i] > 1e-2f);
            numActiveBeamsStrict += (beamNorms_h[i] > 1e-6f);
        }

        std::vector<std::pair<int, float>> beamNorms_pair(nBeams);
        for (int i=0; i<nBeams; i++) {
            beamNorms_pair[i].first = i;
            beamNorms_pair[i].second = beamNorms_h[i];
        }

        auto customComparator = [](const std::pair<int, float>& a,
            const std::pair<int, float>& b) {
            return a.second > b.second; };
        std::sort(beamNorms_pair.begin(), beamNorms_pair.end(), customComparator);
        int numElements = min(20, numActiveBeams);
        topN[k].resize(numElements);
        for (int i=0; i<numElements; i++) {
            topN[k][i] = beamNorms_pair[i].first;
        }

        /*  Finished computing the objective function value.  */
        if ((k + 1) % params.changeWeightsTrigger == 0) {
            float factor = 0.0f;
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
        if ((k+1 % pruneTrigger == 0) && (currentActiveBeams >
            numActiveBeamsStrict + 20)) {
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
            std::cout << "FISTA iteration is: " << k << ", cost: " << costs[k]
                << ", t: " << t << ", numActiveBeams: " << numActiveBeams
                << ", numActiveBeamsStrict: " << numActiveBeamsStrict
                << ", top beams:\n";
            const std::vector<int>& current_topN = topN[k];
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

    // output results
    std::vector<float> xkm1_h(xkm1.size, 0.0f);
    checkCudaErrors(cudaMemcpy(xkm1_h.data(), xkm1.data,
        xkm1.size*sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> xFull(numBeamlets_max, 0.0f);
    size_t xFull_idx = 0;
    size_t xkm1_h_idx = 0;
    for (size_t i=0; i<nBeams; i++) {
        size_t current_beam_pixels = 0;
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
    return 0;
}