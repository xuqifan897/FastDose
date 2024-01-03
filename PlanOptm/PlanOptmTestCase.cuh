#ifndef __PLANOPTMTESTCASE_CUH__
#define __PLANOPTMTESTCASE_CUH__
#include "fastdose.cuh"
#include "PlanOptmBeamBundle.cuh"
#include "cuda_runtime.h"

#include <vector>

#define pitchModule 64

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

namespace PlanOptm {
    bool testCase(
        fastdose::DENSITY_h& density_h,
        fastdose::DENSITY_d& density_d,
        fastdose::SPECTRUM_h& spectrum_h,
        fastdose::KERNEL_h& kernel_h,
        cudaStream_t stream=0
    );

    bool beamBundleTestCase(
        std::vector<BeamBundle> beam_bundles,
        fastdose::DENSITY_h& density_h,
        fastdose::DENSITY_d& density_d,
        fastdose::SPECTRUM_h& spectrum_h,
        fastdose::KERNEL_h& kernel_h,
        cudaStream_t stream=0
    );

    bool beamBundleTestCaseSparse(
        std::vector<BeamBundle> beam_bundles,
        fastdose::DENSITY_h& density_h,
        fastdose::DENSITY_d& density_d,
        fastdose::SPECTRUM_h& spectrum_h,
        fastdose::KERNEL_h& kernel_h,
        cudaStream_t stream=0
    );

    bool textureMemTest();
}

#endif