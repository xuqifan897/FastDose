#include "fastdose.cuh"
#include "argparse.h"
#include "init.cuh"

namespace fd = fastdose;
using namespace example;


__global__ void d_testKernel(int* output, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    output[idx] = idx;
}


void testKernel(){
    int size = 1024;
    int* output;
    checkCudaErrors(cudaMalloc((void**)&output, size*sizeof(int)));
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(16, 1, 1);
    d_testKernel<<<gridSize, blockSize>>>(output, size);

    cudaDeviceSynchronize();
    std::vector<int> output_h(size);
    checkCudaErrors(cudaMemcpy(output_h.data(), output, size*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i=0; i<size; i++)
        std::cout << output_h[i] << " ";
    std::cout << std::endl;
}


int main(int argc, char** argv) {
    testKernel();
    return 0;

    if(argparse(argc, argv))
        return 0;
    
    int deviceIdx = getarg<int>("deviceIdx");
    cudaSetDevice(deviceIdx);

    if (fd::showDeviceProperties(deviceIdx)) {
        std::cerr << "Cannot show device properties." << std::endl;
        return 1;
    }

    fd::DENSITY_h density_h;
    fd::DENSITY_d density_d;
    if(densityInit(density_h, density_d)) {
        std::cerr << "density initialization failure." << std::endl;
        return 1;
    }
#if false
    densityTest(density_h, density_d);
#endif

    std::vector<fd::BEAM_h> beams_h;
    std::vector<fd::BEAM_d> beams_d;
    if (beamsInit(beams_h, beams_d, density_h)) {
        std::cerr << "beam initialization failure." << std::endl;
        return 1;
    }

    fd::SPECTRUM_h spectrum_h;
    if (specInit(spectrum_h)) {
        std::cerr << "spectrum initialization failure." << std::endl;
        return 1;
    }

    fd::KERNEL_h kernel_h;
    if (kernelInit(kernel_h)) {
        std::cerr << "kernel initialization failure." << std::endl;
        return 1;
    }

#if true
    std::string outputFolder = getarg<std::string>("outputFolder");
    if (fd::test_TermaComputeCollective(beams_d, density_d, spectrum_h, outputFolder))
        return 1;
#endif

#if true
    if (fd::test_DoseComputeCollective(
        beams_d, density_d, outputFolder,
        example::getarg<int>("FmapOn"), kernel_h))
        return 1;
#endif
}