#ifndef __PREPROCESSRINGSTRUCTURE_H__
#define __PREPROCESSRINGSTRUCTURE_H__

#include "rtstruct.h"
#include "PreProcessROI.h"
#include "PreProcessArgs.h"
#include <vector>
#include "cuda_runtime.h"
typedef unsigned char uint8_t;

namespace PreProcess {
    class cudaVolume {
    public:
        cudaVolume(): size(int3{0, 0, 0}), _vect(nullptr) {}
        cudaVolume(int3 _size_);
        ~cudaVolume();
        int3 size;  // volume shape (units of integer voxels)
        uint8_t* _vect = nullptr;  // linear array flattened in C-Major order (Depth->Row->Column)
    };
    bool imdilate(cudaVolume& result, const cudaVolume& op1, const cudaVolume& op2);
    __global__ void d_imdilate(
        const int3 targetDim,
        const int3 op2Dim,
        uint8_t* target_vec,
        const uint8_t* op1_vec,
        const uint8_t* op2_vec
    );
    bool CreateRingStructure(ROIMaskList& roi_list, RTStruct& rtstruct,
        const FloatVolume& ctdata, const FloatVolume& density, bool verbose);

    bool test_imdilate();
}

#endif