#include "spectrum.cuh"
#include "helper_cuda.h"

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <boost/filesystem.hpp>

namespace fd = fastdose;
namespace fs = boost::filesystem;

bool fd::SPECTRUM_h::read_spectrum_file(const std::string& spectrum_file, bool verbose) {
    std::ifstream f(spectrum_file);
    if (! f) {
        std::cout << "Cannot open spectrum file " << spectrum_file << std::endl;
        return true;
    }

    this->nkernels = 0;
    float sum_fluence = 0.;
    std::string tableRow;
    std::string buff;
    while (std::getline(f, tableRow)) {
        if (tableRow == std::string("\n"))
            break;
        std::istringstream iss(tableRow);
        this->fluence.push_back(0.);
        this->energy.push_back(0.);
        this->mu_en.push_back(0.);
        this->mu.push_back(0.);

        iss >> this->energy.back() >> this->fluence.back() 
            >> this->mu.back() >> this->mu_en.back() >> buff;
        
        sum_fluence += this->fluence.back();
        this->nkernels ++;
    }
    for (int i=0; i<this->nkernels; i++)
        this->fluence[i] /= sum_fluence;

    if (verbose) {
        int width = 20;
        std::cout << "SPECTRUM_DATA" << std::endl;
        std::cout << "    Spectrum file: " << spectrum_file << std::endl;
        std::cout << std::left << std::setw(width) << "energy [MeV]" << std::left << std::setw(width) << "fluence"
            << std::left << std::setw(width) << "mu[cm^2/g]" << std::left << std::setw(width) << "mu_en[cm^2/g]" << std::endl;
        std::cout << std::scientific;
        std::cout << std::setprecision(4);
        for (int i=0; i<this->nkernels; i++) {
            std::cout << std::left << std::setw(width) << this->energy[i]
                << std::left << std::setw(width) << this->fluence[i]
                << std::left << std::setw(width) << this->mu[i]
                <<std::left << std::setw(width) << this->mu_en[i] << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}

void fd::test_spectrum(const SPECTRUM_h& spectrum_h) {
    int nkernels = spectrum_h.nkernels;

    std::vector<float> energy_sample(nkernels);
    std::vector<float> fluence_sample(nkernels);
    std::vector<float> mu_sample(nkernels);
    std::vector<float> mu_en_sample(nkernels);

    float* energy_sample_d;
    float* fluence_sample_d;
    float* mu_sample_d;
    float* mu_en_sample_d;

    checkCudaErrors(cudaMalloc((void**)&energy_sample_d, nkernels*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&fluence_sample_d, nkernels*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&mu_sample_d, nkernels*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&mu_en_sample_d, nkernels*sizeof(float)));

    dim3 blockSize{MAX_KERNEL_NUM};
    dim3 gridSize{1};
    d_test_spectrum<<<gridSize, blockSize>>>(energy_sample_d, nkernels, 0);
    d_test_spectrum<<<gridSize, blockSize>>>(fluence_sample_d, nkernels, 1);
    d_test_spectrum<<<gridSize, blockSize>>>(mu_sample_d, nkernels, 2);
    d_test_spectrum<<<gridSize, blockSize>>>(mu_en_sample_d, nkernels, 3);

    checkCudaErrors(cudaMemcpy(energy_sample.data(), energy_sample_d,
        nkernels*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(fluence_sample.data(), fluence_sample_d,
        nkernels*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mu_sample.data(), mu_sample_d,
        nkernels*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mu_en_sample.data(), mu_en_sample_d,
        nkernels*sizeof(float), cudaMemcpyDeviceToHost));
    
    double absolute_diff = 0.;
    for (int i=0; i<nkernels; i++) {
        absolute_diff += abs(spectrum_h.energy[i] - energy_sample[i]);
        absolute_diff += abs(spectrum_h.fluence[i] - fluence_sample[i]);
        absolute_diff += abs(spectrum_h.mu[i] - mu_sample[i]);
        absolute_diff += abs(spectrum_h.mu_en[i] - mu_en_sample[i]);
    }

    std::cout << "Absolute difference: " << absolute_diff << std::endl;

    // clean up
    checkCudaErrors(cudaFree(energy_sample_d));
    checkCudaErrors(cudaFree(fluence_sample_d));
    checkCudaErrors(cudaFree(mu_sample_d));
    checkCudaErrors(cudaFree(mu_en_sample_d));
}