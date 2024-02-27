#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <vector>
#include <string>

#define MAX_THETA_ANGLES 24

namespace fastdose{
    class KERNEL_h {
    public:
        std::vector<float> thetaBegin;
        std::vector<float> thetaEnd;
        std::vector<float> thetaMiddle;
        std::vector<float> paramA;
        std::vector<float> parama;
        std::vector<float> paramB;
        std::vector<float> paramb;
        std::vector<float> phiAngles;
        int nTheta;
        int nPhi;

        bool read_kernel_file(const std::string& kernel_file, int nPhi, bool verbose=false);
        bool bind_kernel();
    };

    void test_kernel(const KERNEL_h& kernel_h);
    __global__ void
    d_test_kernel(float* output, int width, int idx);
}

#endif