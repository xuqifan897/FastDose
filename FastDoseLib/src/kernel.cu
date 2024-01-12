#include "fastdose.cuh"
#include "kernel.cuh"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "math_constants.h"

namespace fd = fastdose;

bool fd::KERNEL_h::read_kernel_file(
    const std::string& kernel_file, int nPhi, bool verbose
) {
    std::ifstream f(kernel_file);
    if (! f.is_open()) {
        std::cerr << "Could not open kernel file: " << kernel_file << std::endl;
        return 1;
    }

    this->nTheta = 0;
    std::string firstLine;
    std::string tableRow;
    std::string buff;

    std::getline(f, firstLine);
    while (std::getline(f, tableRow)) {
        std::istringstream iss(tableRow);
        this->thetaBegin.push_back(0.);
        this->thetaEnd.push_back(0.);
        this->thetaMiddle.push_back(0.);
        this->paramA.push_back(0.);
        this->parama.push_back(0.);
        this->paramB.push_back(0.);
        this->paramb.push_back(0.);
        iss >> this->thetaBegin.back() >> this->thetaEnd.back()
            >> this->paramA.back() >> this->parama.back()
            >> this->paramB.back() >> this->paramb.back();
        this->thetaMiddle.back() = (this->thetaBegin.back() + this->thetaEnd.back()) / 2;
        this->nTheta ++;
    }

    //initialize phi angles
    this->nPhi = nPhi;
    float phi_interval = 2 * CUDART_PI_F / this->nPhi;
    this->phiAngles.resize(this->nPhi);
    for (int i=0; i<this->nPhi; i++) {
        this->phiAngles[i] = (i + 0.5) * phi_interval;
    }

    if (verbose) {
        int width = 12;
        std::cout << firstLine << std::endl;
        for (int i=0; i<this->nTheta; i++) {
            std::cout << std::left << std::setw(width) << this->thetaBegin[i]
            << std::left << std::setw(width) << this->thetaEnd[i]
            << std::left << std::setw(width) << this->paramA[i]
            << std::left << std::setw(width) << this->parama[i]
            << std::left << std::setw(width) << this->paramB[i]
            << std::left << std::setw(width) << this->paramb[i] << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Phi angles:" << std::endl;
        for (int i=0; i<this->nPhi; i++) {
            std::cout << std::left << std::setw(width) << this->phiAngles[i];
        }
        std::cout << std::endl << std::endl;
    }
    return 0;
}


void fd::test_kernel(const KERNEL_h& kernel_h) {
    int nTheta = kernel_h.nTheta;

    std::vector<float> paramA_sample(nTheta);
    std::vector<float> parama_sample(nTheta);
    std::vector<float> paramB_sample(nTheta);
    std::vector<float> paramb_sample(nTheta);
    std::vector<float> theta_sample(nTheta);
    std::vector<float> phi_sample(kernel_h.nPhi);

    float* paramA_sample_d;
    float* parama_sample_d;
    float* paramB_sample_d;
    float* paramb_sample_d;
    float* theta_sample_d;
    float* phi_sample_d;

    checkCudaErrors(cudaMalloc((void**)&paramA_sample_d, nTheta*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&parama_sample_d, nTheta*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&paramB_sample_d, nTheta*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&paramb_sample_d, nTheta*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&theta_sample_d, nTheta*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&phi_sample_d, kernel_h.nPhi*sizeof(float)));

    dim3 gridSize(1, 1, 1);
    dim3 blockSize(1, 1, 1);
    int nAngles = std::max(kernel_h.nTheta, kernel_h.nPhi);
    blockSize.x = ((nAngles + WARPSIZE-1) / WARPSIZE) * WARPSIZE;
    d_test_kernel<<<gridSize, blockSize>>>(paramA_sample_d, kernel_h.nTheta, 0);
    d_test_kernel<<<gridSize, blockSize>>>(parama_sample_d, kernel_h.nTheta, 1);
    d_test_kernel<<<gridSize, blockSize>>>(paramB_sample_d, kernel_h.nTheta, 2);
    d_test_kernel<<<gridSize, blockSize>>>(paramb_sample_d, kernel_h.nTheta, 3);
    d_test_kernel<<<gridSize, blockSize>>>(theta_sample_d, kernel_h.nTheta, 4);
    d_test_kernel<<<gridSize, blockSize>>>(phi_sample_d, kernel_h.nPhi, 5);

    checkCudaErrors(cudaMemcpy(paramA_sample.data(), paramA_sample_d,
        nTheta*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(parama_sample.data(), parama_sample_d,
        nTheta*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(paramB_sample.data(), paramB_sample_d,
        nTheta*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(paramb_sample.data(), paramb_sample_d,
        nTheta*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(theta_sample.data(), theta_sample_d,
        nTheta*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(phi_sample.data(), phi_sample_d,
        kernel_h.nPhi*sizeof(float), cudaMemcpyDeviceToHost));
    
    int width = 12;
    std::cout << std::left << std::setw(width) << "kernelAngle"
        << std::left << std::setw(width) << "A"
        << std::left << std::setw(width) << "a"
        << std::left << std::setw(width) << "B"
        << std::left << std::setw(width) << "b" << std::endl;
    for (int i=0; i<kernel_h.nTheta; i++) {
        std::cout << std::left << std::setw(width) << std::setprecision(4) << theta_sample[i]
        << std::left << std::setw(width) << paramA_sample[i]
        << std::left << std::setw(width) << parama_sample[i]
        << std::left << std::setw(width) << paramB_sample[i]
        << std::left << std::setw(width) << paramb_sample[i] << std::endl;
    }
    std::cout << std::endl;
    std::cout << "phi angles:" << std::endl;
    for (int i=0; i<kernel_h.nPhi; i++)
        std::cout << std::left << std::setw(width) << phi_sample[i];
    std::cout << std::endl;

    // clean up
    checkCudaErrors(cudaFree(paramA_sample_d));
    checkCudaErrors(cudaFree(parama_sample_d));
    checkCudaErrors(cudaFree(paramB_sample_d));
    checkCudaErrors(cudaFree(paramb_sample_d));
    checkCudaErrors(cudaFree(theta_sample_d));
    checkCudaErrors(cudaFree(phi_sample_d));
}