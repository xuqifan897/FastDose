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

        #if true
        // for debug purposes
            std::cout << "Iteration: " << k_global << ", t: " << std::scientific << t
                << ", loss: " << cost << std::endl;
        #endif
    }
    #if true
    // for debug purposes
        std::cout << "End of optimization GPU\ngx:" << gx << "\nnrm:\n\n" << nrm << "\n\n"
            << "x:\n" << x << std::endl;
    #endif
    return 0;
}