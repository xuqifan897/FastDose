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

    MatReservior VOIRes, VOIResT, FGRes, FGResT;
    VOIRes.load(VOIMatrices);
    VOIResT.load(VOIMatricesT);
    FGRes.load(SpFluenceGrad);
    FGResT.load(SpFluenceGradT);

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
    array_1d<float> xkm1, vkm1, a, y, gradAty, in, x;
    // of shape (numBeamletsPerBeam, numBeams), constant shape
    array_1d<float> x2d, x2dprox;
    // weights
    Weights_d weights_d; weights_d.fromHost(weights_h);
    // operators
    eval_g operator_eval_g((size_t)weights_d.voxels_PTV, (size_t)weights_h.voxels_OAR, D_rows_max);
    eval_grad operator_eval_grad((size_t)weights_d.voxels_PTV, (size_t)weights_d.voxels_OAR,
        numBeamlets_max, D_rows_max);
    
    // create a list for easier management
    std::vector<array_1d<float>*> array_group1{&xkm1, &vkm1, &a, &y, &gradAty, &in, &x};
    std::vector<array_1d<float>*> array_group2{&x2d, &x2dprox};
    arrayInit_group1(array_group1, numBeamlets_max);
    arrayInit_group2(array_group2, fluenceArray.size());

    // a vector indicating the active beams
    std::vector<uint8_t> active_beams(nBeams, 1);

    cusparseHandle_t handle;
    checkCusparse(cusparseCreate(&handle));
    
    if (DimensionReduction(active_beams,
        VOIRes, VOIMatrices,
        VOIResT, FGRes, FGResT,
        &A, &ATrans, &D, &DTrans,
        operator_eval_g, operator_eval_grad,
        array_group1, handle)) {
        std::cerr << "Dimension reduction error." << std::endl;
        return 1;
    }
    
    float all_zero_cost = operator_eval_g.evaluate(
        *A, *D, vkm1, params.gamma, handle,
        weights_d.maxDose.data, weights_d.minDoseTarget.data,
        weights_d.minDoseTargetWeights.data, weights_d.maxWeightsLong.data,
        weights_d.OARWeightsLong.data, params.eta);
    std::cout << "all zero cost is: " << all_zero_cost << std::endl;
    return 0;
}