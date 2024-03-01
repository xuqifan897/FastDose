#ifndef __IMRTOPTBENCH_H__
#define __IMRTOPTBENCH_H__
#include <vector>
#include <Eigen/Dense>
#include "IMRTDoseMatEigen.cuh"
#include "IMRTInit.cuh"

#define BENCH_INF 10086.0f
#define BENCH_EPS 1e-4f

namespace IMRT {
    class eval_grad_Eigen {
    public:
        eval_grad_Eigen(size_t input_dim, size_t ptv_voxels,
            size_t oar_voxels, size_t d_rows_current);
        float evaluate(const MatCSR_Eigen& A, const MatCSR_Eigen& ATrans,
            const MatCSR_Eigen& D, const MatCSR_Eigen& DTrans,
            const Eigen::VectorXf& x, Eigen::VectorXf& grad, float gamma,
            
            const Eigen::VectorXf& maxDose,
            const Eigen::VectorXf& minDoseTarget,
            const Eigen::VectorXf& minDoseTargetWeights,
            const Eigen::VectorXf& maxWeightsLong,
            const Eigen::VectorXf& OARWeightsLong,
            float eta);
    
    private:
        size_t inputDim;
        size_t PTV_voxels;
        size_t OAR_voxels;
        size_t D_rows_current;

        Eigen::VectorXf Ax;
        Eigen::VectorXf prox1;
        Eigen::VectorXf prox2;
        Eigen::VectorXf term3;
        Eigen::VectorXf term4;
        Eigen::VectorXf prox4;

        Eigen::VectorXf sumProx1;
        Eigen::VectorXf sumProx2;
        Eigen::VectorXf sumTerm3;
        Eigen::VectorXf sumProx4;
        Eigen::VectorXf sumProx4Term4;

        Eigen::VectorXf grad_term1_input;
        Eigen::VectorXf grad_term1_output;
        Eigen::VectorXf grad_term2_input;
        Eigen::VectorXf grad_term2_output;
    };

    class eval_g_Eigen {
    public:
        eval_g_Eigen(size_t ptv_voxels, size_t oar_voxels, size_t d_rows_current);
        float evaluate(const MatCSR_Eigen& A, const MatCSR_Eigen& D,
            const Eigen::VectorXf& x, float gamma,
            
            const Eigen::VectorXf& maxDose,
            const Eigen::VectorXf& minDoseTarget,
            const Eigen::VectorXf& minDoseTargetWeights,
            const Eigen::VectorXf& maxWeightsLong,
            const Eigen::VectorXf& OARWeightsLong,
            float eta);
    
    private:
        size_t PTV_voxels;
        size_t OAR_voxels;
        size_t D_rows_current;

        Eigen::VectorXf Ax;
        Eigen::VectorXf prox1;
        Eigen::VectorXf prox2;
        Eigen::VectorXf term3;
        Eigen::VectorXf term4;
        Eigen::VectorXf prox4;

        Eigen::VectorXf sumProx1;
        Eigen::VectorXf sumProx2;
        Eigen::VectorXf sumTerm3;
        Eigen::VectorXf sumProx4;
        Eigen::VectorXf sumProx4Term4;
    };

    class proxL2Onehalf_QL_Sparse {
    public:
        proxL2Onehalf_QL_Sparse(const MatCSR_Eigen& g0_example);
        bool evaluate(const MatCSR_Eigen& g0, const Eigen::VectorXf& tau,
            MatCSR_Eigen& prox, Eigen::VectorXf& nrmnew);
    private:
        size_t numRows;
        size_t numCols;
        MatCSR_Eigen g02;
        Eigen::VectorXf nrm2;
        Eigen::VectorXf nrm234;
        Eigen::VectorXf alpha;
        Eigen::VectorXf sHat;
        Eigen::VectorXf tHat;
        MatCSR_Eigen prox_square;
        Eigen::VectorXf nrm2newbuff;
    };

    class proxL2Onehalf_QL_Dense {
    public:
        proxL2Onehalf_QL_Dense(const Eigen::MatrixXf& g0_example);
        bool evaluate (const Eigen::MatrixXf& g0, const Eigen::VectorXf& tau,
            Eigen::MatrixXf& prox, Eigen::VectorXf& nrmnew);
    private:
        size_t numRows;
        size_t numCols;
        Eigen::MatrixXf g02;
        Eigen::VectorXf sum_vec;
        Eigen::VectorXf nrm2;
        Eigen::VectorXf nrm234;
        Eigen::VectorXf alpha;
        Eigen::VectorXf sHat;
        Eigen::VectorXf tHat;
        Eigen::MatrixXf prox_square;
        Eigen::VectorXf nrm2newbuff;
    };

    bool randomize_MatCSR_Eigen(MatCSR_Eigen& target,
        size_t numRows, size_t numCols, float sparsity=0.3);
    bool randomize_VectorXf(Eigen::VectorXf& target);
    bool Optimize_Eigen(size_t numBeams, size_t numBeamletsPerBeam,
        size_t ptv_voxels, size_t oar_voxels, size_t d_rows_current,
        const MatCSR_Eigen& A, const MatCSR_Eigen& ATrans,
        const MatCSR_Eigen& D, const MatCSR_Eigen& DTrans, Eigen::VectorXf& xkm1,
        const Eigen::VectorXf& beamWeights, const Eigen::VectorXf& maxDose,
        const Eigen::VectorXf& minDoseTarget, const Eigen::VectorXf& minDoseTargetWeights,
        const Eigen::VectorXf& maxWeightsLong, const Eigen::VectorXf& OARWeightsLong,
        const Params& params);
}

#endif