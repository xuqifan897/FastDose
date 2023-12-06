#ifndef __SPECTRUM_CUH__
#define __SPECTRUM_CUH__

#include <vector>

namespace fastdose {
    class SPECTRUM_h {
    public:
        std::vector<float> fluence;
        std::vector<float> energy;
        std::vector<float> mu_en;
        std::vector<float> mu;
        int nkernels;

        bool read_spectrum_file(const std::string& spectrum_file);
    };

    class SPECTRUM_d {

    };
}

#endif