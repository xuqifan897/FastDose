#include <iostream>
#include <iomanip>
#include "IMRTOptBench.cuh"

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

    IMRT::Optimize_Eigen(numBeams, numBeamletsPerBeam,
        ptv_voxels, oar_voxels, d_rows_current,
        A, ATrans, D, DTrans, xkm1,
        beamWeights, maxDose, minDoseTarget, minDoseTargetWeights,
        maxWeightsLong, OARWeightsLong, params);
}