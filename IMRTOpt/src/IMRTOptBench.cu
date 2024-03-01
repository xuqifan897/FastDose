#include <iostream>
#include "IMRTOptBench.cuh"

#define TWO_OVER_ROOT3 1.1547005383792517f
#define THREE_QUARTERS_ROOT3 1.299038105676658f
#define PI_OVER_TWO 1.5707963267948966f
#define TWO_TIMES_ROOT6_OVER_NINE 0.5443310539518174f

IMRT::eval_grad_Eigen::eval_grad_Eigen(
    size_t input_dim, size_t ptv_voxels,
    size_t oar_voxels, size_t d_rows_current) :
    inputDim(input_dim), PTV_voxels(ptv_voxels),
    OAR_voxels(oar_voxels), D_rows_current(d_rows_current),
    Ax(ptv_voxels + oar_voxels), prox1(ptv_voxels), prox2(ptv_voxels + oar_voxels),
    term3(oar_voxels), term4(d_rows_current), prox4(d_rows_current),
    
    sumProx1(ptv_voxels), sumProx2(ptv_voxels+oar_voxels), sumTerm3(oar_voxels),
    sumProx4(d_rows_current), sumProx4Term4(d_rows_current),
    
    grad_term1_input(ptv_voxels+oar_voxels), grad_term1_output(input_dim),
    grad_term2_input(d_rows_current), grad_term2_output(input_dim) {}

float IMRT::eval_grad_Eigen::evaluate(
    const MatCSR_Eigen& A, const MatCSR_Eigen& ATrans,
    const MatCSR_Eigen& D, const MatCSR_Eigen& DTrans,
    const Eigen::VectorXf& x, Eigen::VectorXf& grad, float gamma,
    
    const Eigen::VectorXf& maxDose,
    const Eigen::VectorXf& minDoseTarget,
    const Eigen::VectorXf& minDoseTargetWeights,
    const Eigen::VectorXf& maxWeightsLong,
    const Eigen::VectorXf& OARWeightsLong,
    float eta) {
    
    this->Ax = A * x;
    this->prox1 = this->Ax.segment(0l, this->PTV_voxels);
    this->prox1 -= minDoseTarget;
    this->prox1 = this->prox1.cwiseMin(0.0f);

    this->prox2 = this->Ax - maxDose;
    this->prox2 = this->prox2.cwiseMax(0.0f);

    this->term3 = this->Ax.segment(this->PTV_voxels, this->OAR_voxels);

    this->term4 = D * x;

    this->prox4 = this->term4.cwiseAbs();
    this->prox4 = this->prox4.array() - gamma;
    this->prox4 = this->prox4.cwiseMax(0.0f);
    this->prox4 = this->prox4.array() * this->term4.cwiseSign().array();

    this->grad_term1_input.segment(0l, this->PTV_voxels) =
        minDoseTargetWeights.array() * this->prox1.array();
    this->grad_term1_input.segment(this->PTV_voxels, this->OAR_voxels) =
        OARWeightsLong.array() * this->term3.array();
    this->grad_term1_input = this->grad_term1_input.array() + maxWeightsLong.array() * this->prox2.array();
    this->grad_term1_output = ATrans * this->grad_term1_input;

    this->grad_term2_input = (this->term4 - this->prox4) * (eta / gamma);
    this->grad_term2_output = DTrans * this->grad_term2_input;

    grad = this->grad_term1_output + this->grad_term2_output;

    this->sumProx1 = minDoseTargetWeights.array() * this->prox1.array() * this->prox1.array();
    this->sumProx2 = maxWeightsLong.array() * this->prox2.array() * this->prox2.array();
    this->sumTerm3 = OARWeightsLong.array() * this->term3.array() * this->term3.array();
    this->sumProx4 = this->prox4.cwiseAbs();
    this->sumProx4Term4 = this->prox4 - this->term4;
    this->sumProx4Term4 = this->sumProx4Term4.array().square();

    float result = 0.5f * (this->sumProx1.sum() + this->sumProx2.sum() + this->sumTerm3.sum())
        + eta * (this->sumProx4.sum() + (0.5f / gamma) * this->sumProx4Term4.sum());

    return result;
}


IMRT::eval_g_Eigen::eval_g_Eigen(
    size_t ptv_voxels, size_t oar_voxels, size_t d_rows_current):
    PTV_voxels(ptv_voxels), OAR_voxels(oar_voxels), D_rows_current(d_rows_current),

    Ax(ptv_voxels + oar_voxels), prox1(ptv_voxels), prox2(ptv_voxels + oar_voxels),
    term3(oar_voxels), term4(d_rows_current), prox4(d_rows_current),
    
    sumProx1(ptv_voxels), sumProx2(ptv_voxels+oar_voxels), sumTerm3(oar_voxels),
    sumProx4(d_rows_current), sumProx4Term4(d_rows_current)
{}


float IMRT::eval_g_Eigen::evaluate(
    const MatCSR_Eigen& A, const MatCSR_Eigen& D,
    const Eigen::VectorXf& x, float gamma,
    
    const Eigen::VectorXf& maxDose,
    const Eigen::VectorXf& minDoseTarget,
    const Eigen::VectorXf& minDoseTargetWeights,
    const Eigen::VectorXf& maxWeightsLong,
    const Eigen::VectorXf& OARWeightsLong,
    float eta
) {
    this->Ax = A * x;
    this->prox1 = this->Ax.segment(0l, this->PTV_voxels);
    this->prox1 -= minDoseTarget;
    this->prox1 = this->prox1.cwiseMin(0.0f);

    this->prox2 = this->Ax - maxDose;
    this->prox2 = this->prox2.cwiseMax(0.0f);

    this->term3 = this->Ax.segment(this->PTV_voxels, this->OAR_voxels);

    this->term4 = D * x;

    this->prox4 = this->term4.cwiseAbs();
    this->prox4 = this->prox4.array() - gamma;
    this->prox4 = this->prox4.cwiseMax(0.0f);
    this->prox4 = this->prox4.array() * this->term4.cwiseSign().array();

    this->sumProx1 = minDoseTargetWeights.array() * this->prox1.array() * this->prox1.array();
    this->sumProx2 = maxWeightsLong.array() * this->prox2.array() * this->prox2.array();
    this->sumTerm3 = OARWeightsLong.array() * this->term3.array() * this->term3.array();
    this->sumProx4 = this->prox4.cwiseAbs();
    this->sumProx4Term4 = this->prox4 - this->term4;
    this->sumProx4Term4 = this->sumProx4Term4.array().square();

    float result = 0.5f * (this->sumProx1.sum() + this->sumProx2.sum() + this->sumTerm3.sum())
        + eta * (this->sumProx4.sum() + (0.5f / gamma) * this->sumProx4Term4.sum());

    return result;
}


IMRT::proxL2Onehalf_QL_Sparse::proxL2Onehalf_QL_Sparse(
    const MatCSR_Eigen& g0_example):
    numRows(g0_example.getRows()), numCols(g0_example.getCols()),
    g02(g0_example), prox_square(g0_example),
    nrm2(numRows), nrm234(numRows), alpha(numRows),
    sHat(numRows), tHat(numRows), nrm2newbuff(numRows) {}


bool IMRT::proxL2Onehalf_QL_Sparse::evaluate(
    const MatCSR_Eigen& g0, const Eigen::VectorXf& tau,
    MatCSR_Eigen& prox, Eigen::VectorXf& nrmnew) {

    // element wise square
    size_t nnz = g0.getNnz();
    const float* source_values = g0.getValues();
    float* target_values = (float*)(this->g02.getValues());
    for (size_t i=0; i<nnz; i++) {
        target_values[i] = source_values[i] * source_values[i];
    }

    // sum along rows
    const EigenIdxType* target_offsets = *(this->g02.getOffset());
    for (size_t i=0; i<this->numRows; i++) {
        size_t idx_begin = target_offsets[i];
        size_t idx_end = target_offsets[i+1];
        float row_sum = 0.0f;
        for (size_t j=idx_begin; j<idx_end; j++)
            row_sum += target_values[j];
        this->nrm2(i) = row_sum;
    }

    // element-wise power
    for (size_t i=0; i<this->numRows; i++) {
        float source_value = this->nrm2(i);
        float target_value = (source_value < BENCH_EPS) ?
            BENCH_INF : std::pow(source_value, -0.75f);
        this->nrm234(i) = target_value;
    }

    this->alpha = tau.array() * this->nrm234.array();

    // calculate sHat
    for (size_t i=0; i<this->numRows; i++) {
        float source_value = this->alpha(i);
        if (source_value > TWO_TIMES_ROOT6_OVER_NINE) {
            this->sHat(i) = 0.0f;
            continue;
        }
        source_value *= THREE_QUARTERS_ROOT3;
        source_value = std::acos(source_value);
        source_value += PI_OVER_TWO;
        source_value /= 3.0f;
        source_value = std::sin(source_value);
        this->sHat(i) = TWO_OVER_ROOT3 * source_value;
    }

    // calculate tHat
    this->tHat = this->sHat.array().square();

    // calculate prox
    const EigenIdxType* prox_offsets = *prox.getOffset();
    float* prox_values = (float*)prox.getValues();
    for (size_t i=0; i<this->numRows; i++) {
        float scale = this->tHat(i);
        size_t idx_begin = prox_offsets[i];
        size_t idx_end = prox_offsets[i+1];
        for (size_t j=idx_begin; j<idx_end; j++) {
            prox_values[j] *= scale;
        }
    }

    // calculate prox_square
    float* prox_square_values = (float*)this->prox_square.getValues();
    for (size_t i=0; i<prox.getNnz(); i++) {
        prox_square_values[i] = prox_values[i] * prox_values[i];
    }

    // calculate nrm2newbuff
    EigenIdxType* prox_square_offsets = *(this->prox_square.getOffset());
    for (size_t i=0; i<this->numRows; i++) {
        size_t idx_begin = prox_square_offsets[i];
        size_t idx_end = prox_square_offsets[i];
        float row_sum = 0.0f;
        for (size_t j=idx_begin; j<idx_end; j++)
            row_sum += prox_square_values[i];
        this->nrm2newbuff(i) = row_sum;
    }

    nrmnew = this->nrm2newbuff.array().sqrt();

    return 0;
}


bool IMRT::randomize_MatCSR_Eigen(MatCSR_Eigen& target,
    size_t numRows, size_t numCols, float sparsity) {
    // firstly, construct a dense matrix
    Eigen::MatrixXf dense(numRows, numCols);
    float* dense_data = dense.data();
    size_t numElements = numRows * numCols;
    for (size_t i=0; i<numElements; i++) {
        float luck = (float)std::rand() / RAND_MAX;
        float value = (float)std::rand() / RAND_MAX;
        dense_data[i] = (luck < sparsity) ? value : 0.0f;
    }
    
    Eigen::SparseMatrix<float, Eigen::RowMajor, EigenIdxType>* target_ptr = &target;
    *target_ptr = dense.sparseView().pruned();

    return 0;
}


bool IMRT::randomize_VectorXf(Eigen::VectorXf& target) {
    for (size_t i=0; i<target.size(); i++) {
        target(i) = (float)std::rand() / RAND_MAX;
    }
    return 0;
}


IMRT::proxL2Onehalf_QL_Dense::proxL2Onehalf_QL_Dense(
    const Eigen::MatrixXf& g0_example):
    numRows(g0_example.rows()), numCols(g0_example.cols()),
    g02(numRows, numCols), prox_square(numRows, numCols), sum_vec(numCols),
    nrm2(numRows), nrm234(numRows), alpha(numRows),
    sHat(numRows), tHat(numRows), nrm2newbuff(numRows) {
    
    for (size_t i=0; i<this->numCols; i++)
        this->sum_vec(i) = 1.0f;
}


bool IMRT::proxL2Onehalf_QL_Dense::evaluate(
    const Eigen::MatrixXf& g0, const Eigen::VectorXf& tau,
    Eigen::MatrixXf& prox, Eigen::VectorXf& nrmnew) {
    
    // element-wise square
    this->g02 = g0.array().square();

    // row-wise sum
    this->nrm2 = this->g02 * this->sum_vec;

    // element-wise power
    for (size_t i=0; i<this->numRows; i++) {
        float source_value = this->nrm2(i);
        this->nrm234(i) = (source_value < BENCH_EPS) ?
            BENCH_INF : std::pow(source_value, -0.75f);
    }

    this->alpha = tau.array() * this->nrm234.array();

    // calculate sHat
    for (size_t i=0; i<this->numRows; i++) {
        float source_value = this->alpha(i);
        if (source_value > TWO_TIMES_ROOT6_OVER_NINE) {
            this->sHat(i) = 0.0f;
            continue;
        }
        source_value *= THREE_QUARTERS_ROOT3;
        source_value = std::acos(source_value);
        source_value += PI_OVER_TWO;
        source_value /= 3.0f;
        source_value = std::sin(source_value);
        this->sHat(i) = source_value * TWO_OVER_ROOT3;
    }

    // calculate tHat
    this->tHat = this->sHat.array().square();

    // calculate prox
    for (size_t i=0; i<this->numRows; i++) {
        float factor = this->tHat(i);
        for (size_t j=0; j<this->numCols; j++)
            prox(i, j) = factor * g0(i, j);
    }

    this->prox_square = prox.array().square();

    this->nrm2newbuff = this->prox_square * this->sum_vec;
    nrmnew = this->nrm2newbuff.array().sqrt();
    return 0;
}


bool IMRT::Optimize_Eigen(size_t numBeams, size_t numBeamletsPerBeam,
    size_t ptv_voxels, size_t oar_voxels, size_t d_rows_current,
    const MatCSR_Eigen& A, const MatCSR_Eigen& ATrans,
    const MatCSR_Eigen& D, const MatCSR_Eigen& DTrans, Eigen::VectorXf& xkm1,
    const Eigen::VectorXf& beamWeights, const Eigen::VectorXf& maxDose,
    const Eigen::VectorXf& minDoseTarget, const Eigen::VectorXf& minDoseTargetWeights,
    const Eigen::VectorXf& maxWeightsLong, const Eigen::VectorXf& OARWeightsLong,
    const Params& params
) {
    // sanity check
    size_t numVoxels = ptv_voxels + oar_voxels;
    size_t numBeamlets = numBeams * numBeamletsPerBeam;
    if (A.getRows() != numVoxels
        || A.getCols() != numBeamlets
        || D.getRows() != d_rows_current
        || D.getCols() != numBeamlets
        || xkm1.size() != numBeamlets
        || beamWeights.size() != numBeams
        || maxDose.size() != numVoxels
        || minDoseTarget.size() != ptv_voxels
        || minDoseTargetWeights.size() != ptv_voxels
        || maxWeightsLong.size() != numVoxels
        || OARWeightsLong.size() != oar_voxels) {
        std::cerr << "Size mismatch in function Optimize_Eigen" << std::endl;
    }

    Eigen::MatrixXf x2d(numBeams, numBeamletsPerBeam),
        x2dprox(numBeams, numBeamletsPerBeam);
    // initialize optimization variables
    Eigen::VectorXf vkm1(numBeamlets),
        x(numBeamlets), y(numBeamlets), x_minus_y(numBeamlets),
        v(numBeamlets), in(numBeamlets), gradAty(numBeamlets), nrm(numBeams);

    // initialize operators
    IMRT::eval_grad_Eigen operator_eval_grad_Eigen(
        numBeamlets, ptv_voxels, oar_voxels, d_rows_current);
    IMRT::eval_g_Eigen operator_eval_g_Eigen(
        ptv_voxels, oar_voxels, d_rows_current);
    IMRT::proxL2Onehalf_QL_Dense operator_proxL2One(x2d);

    // scalar parameters
    float gx, gy, rhs, tkm1, theta, theta_km1, a, b, c,
        reductionFactor = 0.5f, t = params.stepSize;

    vkm1 = xkm1 * 0.0f;
    float all_zero_cost = operator_eval_g_Eigen.evaluate(
        A, D, vkm1, params.gamma, maxDose, minDoseTarget,
        minDoseTargetWeights, maxWeightsLong, OARWeightsLong, params.eta);
    std::cout << "All zero cost is: " << all_zero_cost << std::endl;
    vkm1 = xkm1;

    // start optimization
    for (int k=0; k<params.maxIter; k++) {
        if (k <= 50 || k % 5 == 0)
            t /= reductionFactor;
        bool accept_t = false;
        while (! accept_t) {
            if (k > 1) {
                a = tkm1;
                b = t * theta_km1 * theta_km1;
                c = - b;

                theta = (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
                y = (1-theta) * xkm1 + theta * vkm1;
            } else {
                theta = 1.0f;
                y = xkm1;
            }
            gy = operator_eval_grad_Eigen.evaluate(
                A, ATrans, D, DTrans, y, gradAty, params.gamma,
                maxDose, minDoseTarget, minDoseTargetWeights,
                maxWeightsLong, OARWeightsLong, params.eta);
            
            in = y - t * gradAty;
            in = in.cwiseMax(0.0f);
            for (size_t i=0; i<numBeams; i++) {
                size_t offset = i * numBeamletsPerBeam;
                for (size_t j=0; j<numBeamletsPerBeam; j++) {
                    x2d(i, j) = in(offset + j);
                }
            }

            operator_proxL2One.evaluate(x2d, t*beamWeights, x2dprox, nrm);

            for (size_t i=0; i<numBeams; i++) {
                size_t offset = i * numBeamletsPerBeam;
                for (size_t j=0; j<numBeamletsPerBeam; j++) {
                    x(offset + j) = x2dprox(i, j);
                }
            }

            gx = operator_eval_g_Eigen.evaluate(
                A, D, x, params.gamma, maxDose, minDoseTarget,
                minDoseTargetWeights, maxWeightsLong, OARWeightsLong, params.eta);

            x_minus_y = x - y;
            rhs = gy + gradAty.dot(x_minus_y) + (0.5f / t) * x_minus_y.squaredNorm();
            if (gx <= rhs)
                accept_t = true;
            else
                t *= reductionFactor;
        }

        float One_over_theta = 1.0f / theta;
        v = One_over_theta * x + (1 - One_over_theta) * xkm1;

        theta_km1 = theta;
        tkm1 = t;
        xkm1 = x;
        vkm1 = v;

        float loss = gx + (nrm.array() * beamWeights.array()).sum();
        #if true
            std::cout << "Iteration: " << k << ", t:" << std::scientific << t
                << ", loss: " << loss << std::endl;
        #endif
    }
    std::cout << "Result:\n" << x2d << std::endl;
    return 0;
}