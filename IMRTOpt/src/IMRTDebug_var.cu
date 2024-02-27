#include "IMRTDebug.cuh"

bool IMRT::array_1d_diagnosis(const float* obj, size_t size) {
    std::vector<float> obj_h(size, 0.0f);
    checkCudaErrors(cudaMemcpy(obj_h.data(), obj,
        size*sizeof(float), cudaMemcpyDeviceToHost));
    int64_t spot = 0;
    for (; spot<obj_h.size(); spot++) {
        if (std::isnan(obj_h[spot])) {
            return 1;
        }
    }
    return 0;
}

bool IMRT::viewArray(const float* obj, const std::string& var,
    const std::string& file, int line, size_t size) {
    std::vector<float> obj_host(size);
    checkCudaErrors(cudaMemcpy(obj_host.data(), obj,
        size*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << var << " at " << file << ":" << line << ":\n";
    for (size_t i=0; i<size; i++)
        std::cout << obj_host[i] << " ";
    std::cout << "\n" << std::endl;
    return 0;
}