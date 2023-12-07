#ifndef __SPECTRUM_CUH__
#define __SPECTRUM_CUH__

#include <vector>
#include <string>

#define MAX_KERNEL_NUM 24

namespace fastdose {
    class SPECTRUM_h {
    public:
        std::vector<float> fluence;
        std::vector<float> energy;
        std::vector<float> mu_en;
        std::vector<float> mu;
        int nkernels;

        bool read_spectrum_file(const std::string& spectrum_file, bool verbose=true);
        bool bind_spectrum();
    };

    void test_spectrum(const SPECTRUM_h& spectrum_h);
}

#endif