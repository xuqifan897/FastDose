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

namespace fastdose{
    __constant__ float d_energy[MAX_KERNEL_NUM];
    __constant__ float d_fluence[MAX_KERNEL_NUM];
    __constant__ float d_mu[MAX_KERNEL_NUM];
    __constant__ float d_mu_en[MAX_KERNEL_NUM];
}

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
    }
    return 0;
}

bool fd::SPECTRUM_h::bind_spectrum() {
    if (this->nkernels > MAX_KERNEL_NUM) {
        std::cerr << "The number of kernels included is more than MAX_KERNEL_NUM (" 
            << MAX_KERNEL_NUM << ")" << std::endl;
        return 1;
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_energy, this->energy.data(), this->nkernels*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_fluence, this->fluence.data(), this->nkernels*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_mu, this->mu.data(), this->nkernels*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_mu_en, this->mu_en.data(), this->nkernels*sizeof(float)));
    return 0;
}

void fd::test_spectrum(const SPECTRUM_h& spectrum_h) {
    std::vector<float> energy_sample(spectrum_h.nkernels);
    std::vector<float> fluence_sample(spectrum_h.nkernels);
    std::vector<float> mu_sample(spectrum_h.nkernels);
    std::vector<float> mu_en_sample(spectrum_h.nkernels);

    checkCudaErrors(cudaMemcpy(energy_sample.data(), d_energy, spectrum_h.nkernels*sizeof(float), cudaMemcpyDeviceToHost));

}