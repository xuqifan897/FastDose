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
        std::cout << std::endl;
    }
    return 0;
}