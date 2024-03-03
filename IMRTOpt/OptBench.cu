#include <iostream>
#include <iomanip>
#include "IMRTOptBench.cuh"
#include "IMRTDoseMatEigen.cuh"
#include "IMRTOptimize_var.cuh"
#include "IMRTOptimize.cuh"
#include "IMRTDoseMat.cuh"

int main(int argc, char** argv) {
    size_t numBeams = 20;
    size_t numBeamletsPerBeam = 5;
    size_t numBeamlets = numBeams * numBeamletsPerBeam;
    size_t ptv_voxels = 50;
    size_t oar_voxels = 50;
    size_t d_rows_current = 50;
    IMRT::Params params;
    params.eta = 0.1f;
    params.gamma = 1.0f;
    params.maxIter = 1000;
    params.stepSize = 1e-3f;

    std::srand(10086);

    // initialize matrix
    IMRT::MatCSR_Eigen A, ATrans, D, DTrans;
    IMRT::randomize_MatCSR_Eigen(A, ptv_voxels+oar_voxels, numBeamlets);
    IMRT::randomize_MatCSR_Eigen(D, d_rows_current, numBeamlets);
    ATrans = A.transpose();
    DTrans = D.transpose();

    // initialize optimization variables
    Eigen::VectorXf xkm1(numBeamlets);
    IMRT::randomize_VectorXf(xkm1);

    // weight parameters
    Eigen::VectorXf beamWeights(numBeams),
        maxDose(ptv_voxels + oar_voxels),
        minDoseTarget(ptv_voxels),
        minDoseTargetWeights(ptv_voxels),
        maxWeightsLong(ptv_voxels + oar_voxels),
        OARWeightsLong(oar_voxels);
    IMRT::randomize_VectorXf(beamWeights); beamWeights *= 100.0f;
    IMRT::randomize_VectorXf(maxDose); maxDose *= 0.0f;
    IMRT::randomize_VectorXf(minDoseTarget); minDoseTarget *= 10.0f;
    maxDose.segment(0, ptv_voxels) = minDoseTarget;
    IMRT::randomize_VectorXf(minDoseTargetWeights); minDoseTargetWeights *= 1000.0f;
    IMRT::randomize_VectorXf(maxWeightsLong);
    IMRT::randomize_VectorXf(OARWeightsLong);

    if (false) {
        // optimize using CPU
        IMRT::Optimize_Eigen(numBeams, numBeamletsPerBeam,
            ptv_voxels, oar_voxels, d_rows_current,
            A, ATrans, D, DTrans, xkm1,
            beamWeights, maxDose, minDoseTarget, minDoseTargetWeights,
            maxWeightsLong, OARWeightsLong, params);
        return;
    }


    // convert the variables above into GPU
    IMRT::MatCSR64 A_cu, ATrans_cu, D_cu, DTrans_cu;
    IMRT::Eigen2Cusparse(A, A_cu);
    IMRT::Eigen2Cusparse(ATrans, ATrans_cu);
    IMRT::Eigen2Cusparse(D, D_cu);
    IMRT::Eigen2Cusparse(DTrans, DTrans_cu);

    IMRT::array_1d<float> beamWeights_cu, maxDose_cu, minDoseTarget_cu,
        minDoseTargetWeights_cu, maxWeightsLong_cu, OARWeightsLong_cu, xkm1_cu, vkm1_cu;
    IMRT::arrayInit(beamWeights_cu, beamWeights);
    IMRT::arrayInit(maxDose_cu, maxDose);
    IMRT::arrayInit(minDoseTarget_cu, minDoseTarget);
    IMRT::arrayInit(minDoseTargetWeights_cu, minDoseTargetWeights);
    IMRT::arrayInit(maxWeightsLong_cu, maxWeightsLong);
    IMRT::arrayInit(OARWeightsLong_cu, OARWeightsLong);
    IMRT::arrayInit(xkm1_cu, xkm1);
    IMRT::arrayInit(vkm1_cu, xkm1);

    IMRT::MatCSR64 x2d, x2dprox;
    // here, x2d and x2dprox are essentially dense matrices.
    IMRT::MatCSR_Eigen x2d_Eigen;
    EigenIdxType* x2d_offsets = (EigenIdxType*)malloc((numBeams+1)*sizeof(EigenIdxType));
    EigenIdxType* x2d_columns = new EigenIdxType[numBeamlets];
    float* x2d_values = new float[numBeamlets];
    for (size_t i=0; i<numBeams+1; i++)
        x2d_offsets[i] = i * numBeamletsPerBeam;
    for (size_t i=0; i<numBeams; i++) {
        size_t offset_j = i * numBeamletsPerBeam;
        for (size_t j=0; j<numBeamletsPerBeam; j++) {
            x2d_columns[offset_j + j] = j;
            x2d_values[offset_j + j] = 1.0f;
        }
    }
    x2d_Eigen.customInit(
        numBeams, numBeamletsPerBeam, numBeamlets,
        x2d_offsets, x2d_columns, x2d_values);
    IMRT::Eigen2Cusparse(x2d_Eigen, x2d);
    IMRT::Eigen2Cusparse(x2d_Eigen, x2dprox);

    int k_global = 0;
    float theta_km1 = 0.0f, tkm1 = params.stepSize;
    IMRT::BOO_IMRT_L2OneHalf_gpu_QL(A_cu, ATrans_cu, D_cu, DTrans_cu,
        beamWeights_cu, maxDose_cu, minDoseTarget_cu, minDoseTargetWeights_cu,
        maxWeightsLong_cu, OARWeightsLong_cu, numBeamletsPerBeam,
        params.gamma, params.eta,
        k_global, params.maxIter, params.maxIter, theta_km1, tkm1,
        xkm1_cu, vkm1_cu, x2d, x2dprox);
}