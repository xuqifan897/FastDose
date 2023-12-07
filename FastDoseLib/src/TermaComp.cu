#include "fastdose.cuh"
namespace fd = fastdose;

#define DIM_X 8
#define DIM_Y 8

bool fd::TermaCompute(
    BEAM_d& beam_d, DENSITY_d& density_d, cudaStream_t stream
) {
    dim3 blockSize{DIM_X, DIM_Y};
    dim3 gridSize;
    gridSize.x = (beam_d.fmap_size.x + blockSize.x - 1) / blockSize.x;
    gridSize.y = (beam_d.fmap_size.y + blockSize.y - 1) / blockSize.y;

}

__global__ void
fd::d_TermaCompute(BEAM_d beam_d, DENSITY_d density_d) {
    
}