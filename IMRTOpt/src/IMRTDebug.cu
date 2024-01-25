#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <omp.h>
#include "IMRTDebug.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTArgs.h"
#include "IMRTDoseMatEigen.cuh"

namespace fs = boost::filesystem;
namespace fd = fastdose;

bool IMRT::doseCalcDebug(
    std::vector<BeamBundle>& beam_bundles,
    fastdose::DENSITY_d& density_d,
    fastdose::SPECTRUM_h& spectrum_h,
    fastdose::KERNEL_h& kernel_h,
    cudaStream_t stream
) {
    int beamIdx = getarg<int>("beamIdxDebug");
    BeamBundle& beam_bundle = beam_bundles[beamIdx];
    int nBeamlets = beam_bundle.beams_h.size();
    std::vector<fd::BEAM_d> beamlets(nBeamlets);
    for (int i=0; i<nBeamlets; i++)
        fd::beam_h2d(beam_bundle.beams_h[i], beamlets[i]);

    #if false
        std::vector<fd::BEAM_h>& beamlets_h = beam_bundle.beams_h;
        for (int i=0; i<beamlets_h.size(); i++) {
            std::cout << "Beamlet " << i << std::endl;
            std::cout << beamlets_h[i] << std::endl << std::endl;
        }
        return 0;
    #endif

    // preparation
    std::vector<fd::d_BEAM_d> h_beams;
    h_beams.reserve(nBeamlets);
    for (int i=0; i<nBeamlets; i++)
        h_beams.emplace_back(fd::d_BEAM_d(beamlets[i]));
    fd::d_BEAM_d* d_beams = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beams, nBeamlets*sizeof(fd::d_BEAM_d)));
    checkCudaErrors(cudaMemcpy(d_beams, h_beams.data(),
        nBeamlets*sizeof(fd::d_BEAM_d), cudaMemcpyHostToDevice));

    std::vector<float*> h_fluence_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_fluence_array[i] = beamlets[i].fluence;
    float** d_fluence_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_fluence_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_fluence_array, h_fluence_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    std::vector<float*> h_TermaBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_TermaBEV_array[i] = beamlets[i].TermaBEV;
    float** d_TermaBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_TermaBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_TermaBEV_array, h_TermaBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    std::vector<float*> h_DensityBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_DensityBEV_array[i] = beamlets[i].DensityBEV;
    float** d_DensityBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_DensityBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_DensityBEV_array, h_DensityBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    std::vector<float*> h_DoseBEV_array(nBeamlets, nullptr);
    for (int i=0; i<nBeamlets; i++)
        h_DoseBEV_array[i] = beamlets[i].DoseBEV;
    float** d_DoseBEV_array = nullptr;
    checkCudaErrors(cudaMalloc((void***)&d_DoseBEV_array, nBeamlets*sizeof(float*)));
    checkCudaErrors(cudaMemcpy(d_DoseBEV_array, h_DoseBEV_array.data(),
        nBeamlets*sizeof(float*), cudaMemcpyHostToDevice));

    size_t fmap_npixels = beamlets[0].fmap_size.x * beamlets[0].fmap_size.y;

    // calculate Terma collectively
    fd::TermaComputeCollective(
        fmap_npixels,
        nBeamlets,
        d_beams,
        d_fluence_array,
        d_TermaBEV_array,
        d_DensityBEV_array,
        density_d,
        spectrum_h,
        stream
    );
    cudaDeviceSynchronize();
    std::cout << "Collective Terma calculation finished." << std::endl << std::endl;

    // log out data
    fs::path resultFolder(getarg<std::string>("outputFolder"));
    resultFolder /= std::string("doseCompDebug");
    if (! fs::is_directory(resultFolder))
        fs::create_directory(resultFolder);

    #if false
        // log out Terma data
        for (int i=0; i<nBeamlets; i++) {
            const fd::BEAM_d& beamlet = beamlets[i];
            size_t DoseBEVSize = beamlet.DensityBEV_pitch / sizeof(float) * beamlet.long_dim;
            std::vector<float> h_TermaBEV(DoseBEVSize, 0.0f);
            checkCudaErrors(cudaMemcpy(h_TermaBEV.data(), beamlet.TermaBEV,
                DoseBEVSize*sizeof(float), cudaMemcpyDeviceToHost));
            
            fs::path file = resultFolder / (std::string("BEVTerma")
                + std::to_string(i) + ".bin");
            std::ofstream f(file.string());
            if (! f.is_open()) {
                std::cerr << "Could not open file: " << file.string();
                return 1;
            }
            f.write((char*)(h_TermaBEV.data()), DoseBEVSize*sizeof(float));
            f.close();
        }

        // log out Density data
        for (int i=0; i<nBeamlets; i++) {
            const fd::BEAM_d& beamlet = beamlets[i];
            size_t DoseBEVSize = beamlet.DensityBEV_pitch / sizeof(float) * beamlet.long_dim;
            std::vector<float> h_DensityBEV(DoseBEVSize, 0.0f);
            checkCudaErrors(cudaMemcpy(h_DensityBEV.data(), beamlet.DensityBEV,
                DoseBEVSize*sizeof(float), cudaMemcpyDeviceToHost));
            
            fs::path file = resultFolder / (std::string("BEVDensity")
                + std::to_string(i) + ".bin");
            std::ofstream f(file.string());
            if (! f.is_open()) {
                std::cerr << "Could not open file: " << file.string();
                return 1;
            }
            f.write((char*)(h_DensityBEV.data()), DoseBEVSize*sizeof(float));
            f.close();
        }
        return 0;
    #endif

    // print the longitudinal dimensions of beamlets
    for (int i=0; i<nBeamlets; i++)
        std::cout << "Beamlet " << i << ", long_dim: "
        << beamlets[i].long_dim << std::endl;
    std::cout << std::endl;

    #if true
        // calculate Dose collectively
        fd::DoseComputeCollective(
            fmap_npixels,
            nBeamlets,
            d_beams,
            d_TermaBEV_array,
            d_DensityBEV_array,
            d_DoseBEV_array,
            kernel_h.nTheta,
            kernel_h.nPhi,
            stream
        );
        cudaDeviceSynchronize();
        std::cout << "Collective dose calculation finished." << std::endl;
    #else
        for (int i=0; i<nBeamlets; i++) {
            fd::DoseComputeCollective(
                fmap_npixels,
                1,
                d_beams + i,
                d_TermaBEV_array + i,
                d_DensityBEV_array + i,
                d_DoseBEV_array + i,
                kernel_h.nTheta,
                kernel_h.nPhi,
                stream
            );
            cudaDeviceSynchronize();
            std::cout << "Dose calculation. Beamlet: " << i
                << " / " << nBeamlets << std::endl;
        }
        
        for (int i=0; i<nBeamlets; i++) {
            const fd::BEAM_d& beamlet = beamlets[i];
            size_t DoseBEVSize = beamlet.DensityBEV_pitch / sizeof(float) * beamlet.long_dim;
            std::vector<float> h_DoseBEV(DoseBEVSize, 0.0f);
            checkCudaErrors(cudaMemcpy(h_DoseBEV.data(), beamlet.DoseBEV,
                DoseBEVSize*sizeof(float), cudaMemcpyDeviceToHost));
            
            fs::path file = resultFolder / (std::string("BEVDose")
                + std::to_string(i) + ".bin");
            std::ofstream f(file.string());
            if (! f.is_open()) {
                std::cerr << "Could not open file: " << file.string();
                return 1;
            }
            f.write((char*)(h_DoseBEV.data()), DoseBEVSize*sizeof(float));
            f.close();
        }
    #endif

    // clean-up
    checkCudaErrors(cudaFree(d_DoseBEV_array));
    checkCudaErrors(cudaFree(d_DensityBEV_array));
    checkCudaErrors(cudaFree(d_TermaBEV_array));
    checkCudaErrors(cudaFree(d_fluence_array));
    checkCudaErrors(cudaFree(d_beams));

    return 0;
}


bool IMRT::sparseValidation(const MatCSREnsemble* matEns) {
    // do beam-wise dose calculation, the sum of dose constributions from all beamlets
    // get the maximum number of rows (beamlets) of all beams
    int maxNumBeamletsPerBatch = 0;
    for (int i=0; i<matEns->numMatrices; i++)
        maxNumBeamletsPerBatch = max(maxNumBeamletsPerBatch, (int)(matEns->numRowsPerMat[i]));

    // allocate the column vector, representing the weights of all beamlets
    float* d_beamletWeights = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beamletWeights,
        maxNumBeamletsPerBatch*sizeof(float)));
    std::vector<float> h_beamletWeights(maxNumBeamletsPerBatch, 1.0f);
    checkCudaErrors(cudaMemcpy(d_beamletWeights, h_beamletWeights.data(),
        maxNumBeamletsPerBatch*sizeof(float), cudaMemcpyHostToDevice));
    
    // allocate the output vector
    cusparseDnVecDescr_t vecOutput = nullptr;
    float* d_vecOutput = nullptr;
    size_t numColsPerMat = matEns->numColsPerMat;
    checkCudaErrors(cudaMalloc((void**)&d_vecOutput, numColsPerMat*sizeof(float)));
    checkCusparse(cusparseCreateDnVec(&vecOutput, numColsPerMat, d_vecOutput, CUDA_R_32F));
    std::vector<float> h_vecOutput(numColsPerMat, 0.0f);

    size_t bufferSize = 0;
    void* dBuffer = nullptr;

    fs::path resultFolder = fs::path(getarg<std::string>("outputFolder"));
    resultFolder /= std::string("BeamDoseMat");
    if (! fs::is_directory(resultFolder))
        fs::create_directory(resultFolder);
    
    cusparseHandle_t handle = nullptr;
    checkCusparse(cusparseCreate(&handle));
    for (int i=0; i<matEns->numMatrices; i++) {
        int numRows = matEns->numRowsPerMat[i];
        int numNonZero = matEns->NonZeroElements[i];
        size_t* d_csr_offsets = matEns->d_offsetsBuffer + matEns->OffsetBufferIdx[i];
        size_t* d_csr_columns = matEns->d_columnsBuffer;
        float* d_csr_values = matEns->d_valuesBuffer;
        if (i > 0) {
            d_csr_columns += matEns->CumuNonZeroElements[i-1];
            d_csr_values += matEns->CumuNonZeroElements[i-1];
        }

        cusparseSpMatDescr_t matSparse;
        checkCusparse(cusparseCreateCsr(
            &matSparse, numRows, numColsPerMat, numNonZero,
            d_csr_offsets, d_csr_columns, d_csr_values,
            CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        
        // construct input vector
        cusparseDnVecDescr_t vecInput = nullptr;
        checkCusparse(cusparseCreateDnVec(
            &vecInput, numRows, d_beamletWeights, CUDA_R_32F));

        // determine the size of an external buffer
        size_t bufferSizeLocal = 0;
        float alpha = 1;
        float beta = 0;
        checkCusparse(cusparseSpMV_bufferSize (
            handle, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, matSparse, vecInput, &beta, vecOutput, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeLocal));
        
        if (bufferSizeLocal > bufferSize) {
            std::cout << "Enlarge buffer size, from " << bufferSize
                << " to " << bufferSizeLocal << std::endl;
            bufferSize = bufferSizeLocal;
            if (dBuffer != nullptr) {
                checkCudaErrors(cudaFree(dBuffer));
            }
            checkCudaErrors(cudaMalloc((void**)&dBuffer, bufferSize));
        }

        checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, matSparse, vecInput, &beta, vecOutput, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        checkCusparse(cusparseDestroyDnVec(vecInput));
        checkCusparse(cusparseDestroySpMat(matSparse));

        // log out
        checkCudaErrors(cudaMemcpyAsync(h_vecOutput.data(), d_vecOutput,
            numColsPerMat*sizeof(float), cudaMemcpyDeviceToHost));
        fs::path file = resultFolder / (std::string("beam")
            + std::to_string(i) + std::string(".bin"));
        std::ofstream f_handle(file.string());
        if (! f_handle.is_open()) {
            std::cerr << "Cannot open file: " << file << std::endl;
            return 1;
        }
        f_handle.write((char*)(h_vecOutput.data()), numColsPerMat*sizeof(float));
        f_handle.close();
        std::cout << file << std::endl;
    }

    if (bufferSize > 0)
        checkCudaErrors(cudaFree(dBuffer));
    checkCusparse(cusparseDestroyDnVec(vecOutput));
    checkCudaErrors(cudaFree(d_vecOutput));
    if (maxNumBeamletsPerBatch > 0)
        checkCudaErrors(cudaFree(d_beamletWeights));
    return 0;
}


bool IMRT::conversionValidation(
    const MatCSR64& mat, const MatCSREnsemble& matEns
) {
    // allocate input
    const std::vector<size_t>& numRowsPerMat = matEns.numRowsPerMat;
    size_t numRowsTotal = matEns.CumuNumRowsPerMat.back();
    std::vector<float> h_beamletWeights(numRowsTotal, 0);
    float* d_beamletWeights = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_beamletWeights, numRowsTotal*sizeof(float)));
    cusparseDnVecDescr_t vec_beamletWeights = nullptr;
    checkCusparse(cusparseCreateDnVec(&vec_beamletWeights, numRowsTotal,
        d_beamletWeights, CUDA_R_32F));

    // allocate result buffer
    size_t numCols = matEns.numColsPerMat;
    std::vector<float> h_result(numCols, 0);
    float* d_result = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_result, numCols*sizeof(float)));
    cusparseDnVecDescr_t vec_result = nullptr;
    checkCusparse(cusparseCreateDnVec(
        &vec_result, numCols, d_result, CUDA_R_32F));

    // allocate additional buffer
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    cusparseHandle_t handle = nullptr;
    checkCusparse(cusparseCreate(&handle));
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, mat.matA, vec_beamletWeights, &beta, vec_result, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    std::cout << "Buffer size: " << bufferSize << " [bytes]" << std::endl;
    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

    fs::path resultFolder(getarg<std::string>("outputFolder"));
    resultFolder /= std::string("BeamDoseMatNew");
    if (! fs::is_directory(resultFolder))
        fs::create_directory(resultFolder);

    for (int beamIdx=0; beamIdx<numRowsPerMat.size(); beamIdx++) {
        // construct input vector, the fluence map of beamletIdx is one
        size_t beamWeightIdx = 0;
        for (int ii=0; ii<numRowsPerMat.size(); ii++) {
            size_t currentRows = numRowsPerMat[ii];
            float fluenceValue = 0.0f;
            if (ii == beamIdx)
                fluenceValue = 1.0f;
            for (size_t jj=0; jj<currentRows; jj++) {
                h_beamletWeights[beamWeightIdx] = fluenceValue;
                beamWeightIdx++;
            }
        }
        checkCudaErrors(cudaMemcpy(d_beamletWeights, h_beamletWeights.data(),
            numRowsTotal*sizeof(float), cudaMemcpyHostToDevice));

        checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, mat.matA, vec_beamletWeights, &beta, vec_result, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        checkCudaErrors(cudaMemcpy(
            h_result.data(), d_result, numCols*sizeof(float), cudaMemcpyDeviceToHost));

        fs::path file = resultFolder / (std::string("beam")
            + std::to_string(beamIdx) + std::string(".bin"));
        std::ofstream f(file.string());
        if (! f.is_open()) {
            std::cerr << "Cannot open file: " << file << std::endl;
            return 1;
        }
        f.write((char*)h_result.data(), numCols*sizeof(float));
        f.close();
        std::cout << file << std::endl;
    }

    // clean-up
    checkCudaErrors(cudaFree(dBuffer));
    checkCusparse(cusparseDestroy(handle));
    checkCudaErrors(cudaFree(d_result));
    checkCudaErrors(cudaFree(d_beamletWeights));
    return 0;
}


bool IMRT::test_MatCSR_host() {
    // test a sparse matrix
    float matDense[] = {0, 0, 1, 0, 2, 0,
                        2, 0, 0, 1, 0, 0,
                        0, 0, 1, 2, 0, 1,
                        1, 0, 1, 0, 0, 0,
                        1, 1, 0, 0, 0, 1};

    EigenIdxType nRows = 5;
    EigenIdxType nCols = 6;
    EigenIdxType nnz = 0;
    for (int i=0; i<nRows; i++) {
        for (int j=0; j<nCols; j++) {
            EigenIdxType idx = i * nCols + j;
            nnz += (matDense[idx] > eps_fastdose);
        }
    }

    EigenIdxType** offsets = new EigenIdxType*;
    EigenIdxType** columns = new EigenIdxType*;
    float** values = new float*;

    *offsets = (EigenIdxType*)malloc(((nCols+1)*sizeof(EigenIdxType)));
    *columns = (EigenIdxType*)malloc(nnz*sizeof(EigenIdxType));
    *values = (float*)malloc(nnz*sizeof(float));

    EigenIdxType cumu_nnz = 0;
    (*offsets)[0] = cumu_nnz;
    for (int i=0; i<nRows; i++) {
        for (int j=0; j<nCols; j++) {
            EigenIdxType idx = i * nCols + j;
            if (matDense[idx] > eps_fastdose) {
                (*columns)[cumu_nnz] = j;
                (*values)[cumu_nnz] = matDense[idx];
                cumu_nnz ++;
            }
        }
        (*offsets)[i+1] = cumu_nnz;
    }

    #if false
        std::cout << "Offsets: ";
        for (int i=0; i<nRows+1; i++)
            std::cout << (*offsets)[i] << " ";
        std::cout << std::endl;

        std::cout << "Columns: ";
        for (int i=0; i<nnz; i++) {
            std::cout << (*columns)[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Values: ";
        for (int i=0; i<nnz; i++) {
            std::cout << (*values)[i] << " ";
        }
        std::cout << std::endl;
    #endif

    IMRT::MatCSR_Eigen mat;
    mat.customInit(nRows, nCols, nnz,
        *offsets, *columns, *values);
    free(offsets);
    free(columns);
    free(values);

    // then, print out the sparse matrix
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> denseMat = mat.toDense();
    std::cout << "The constructed matrix:\n" << denseMat << std::endl;

    return 0;
}


bool IMRT::test_MatCSR_load(const MatCSR_Eigen& input, const std::string& doseMatFolder) {
    fs::path numRowsPerMatFile = fs::path(doseMatFolder) / fs::path("numRowsPerMat.bin");
    std::ifstream f(numRowsPerMatFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << numRowsPerMatFile << std::endl;
        return 1;
    }
    f.seekg(0, std::ios::end);
    size_t numMatrices = f.tellg() / sizeof(size_t);
    f.seekg(0, std::ios::beg);
    std::vector<size_t> numRowsPerMat(numMatrices, 0);
    f.read((char*)numRowsPerMat.data(), numMatrices*sizeof(size_t));
    f.close();

    fs::path resultFolder(getarg<std::string>("outputFolder"));
    resultFolder /= std::string("BeamDoseMatEigen");
    if (! fs::is_directory(resultFolder))
        fs::create_directory(resultFolder);
    
    // calculate the number of rows
    size_t totalNumRows = 0;
    for (size_t i=0; i<numMatrices; i++)
        totalNumRows += numRowsPerMat[i];
    
    // do calculation
    size_t nThreads = 128;
    Eigen::setNbThreads(nThreads);
    std::cout << Eigen::nbThreads() << " threads are used in Eigen." << std::endl;
    size_t offset = 0;
    for (size_t i=0; i<numMatrices; i++) {
        // construct the sparse vector
        size_t local_nnz = numRowsPerMat[i];
        EigenIdxType* local_offsets = (EigenIdxType*)malloc(2*sizeof(EigenIdxType));
        local_offsets[0] = 0;
        local_offsets[1] = local_nnz;
        EigenIdxType* local_columns = (EigenIdxType*)malloc(local_nnz*sizeof(EigenIdxType));
        float* local_values = (float*)malloc(local_nnz*sizeof(float));
        for (size_t j=0; j<local_nnz; j++) {
            local_columns[j] = offset;
            offset ++;
            local_values[j] = 1.0f;
        }

        MatCSR_Eigen beamletWeights;
        beamletWeights.customInit(1, totalNumRows, local_nnz,
            local_offsets, local_columns, local_values);
        
        MatCSR_Eigen result = beamletWeights * input;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> result_dense = result.toDense();
        // std::cout << "Dense matrix rows: " << result_dense.rows() << ", columns: " << result_dense.cols() << std::endl;
        size_t size = result_dense.rows() * result_dense.cols();

        // save result
        fs::path file = resultFolder / (std::string("beam")
            + std::to_string(i) + std::string(".bin"));
        std::ofstream ofs(file.string());
        if (! ofs.is_open()) {
            std::cerr << "Cannot open file: " << file << std::endl;
            return 1;
        }
        ofs.write((char*)result_dense.data(), size*sizeof(float));
        ofs.close();
        std::cout << file << std::endl;
    }
    return 0;
}