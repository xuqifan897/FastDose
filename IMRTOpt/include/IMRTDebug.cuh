#ifndef __IMRTDEBUG_CUH__
#define __IMRTDEBUG_CUH__
#include <vector>
#include "IMRTBeamBundle.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTDoseMatEns.cuh"
#include "fastdose.cuh"

namespace IMRT {
    bool doseCalcDebug (
        std::vector<BeamBundle>& beam_bundles,
        fastdose::DENSITY_d& density_d,
        fastdose::SPECTRUM_h& spectrum_h,
        fastdose::KERNEL_h& kernel_h,
        cudaStream_t stream = 0
    );

    bool sparseValidation(const MatCSREnsemble* matEns);

    // to test if the sparse matrix is correct
    class MatCSR_Eigen;
    class MatCSR64;
    class MatCSR32;
    bool conversionValidation(const MatCSR64& mat, const MatCSREnsemble& matEns);
    bool test_MatCSR_host();
    bool test_MatCSR_load(const MatCSR_Eigen& input, const std::string& doseMatFolder);
    bool test_MatFilter(const MatCSR32& matFilter, const MatCSR32& matFilterT);
}

#endif