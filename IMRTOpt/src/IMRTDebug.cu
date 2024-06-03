#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <omp.h>
#include "IMRTDebug.cuh"
#include "IMRTDoseMat.cuh"
#include "IMRTArgs.h"
#include "IMRTDoseMatEigen.cuh"
#include "IMRTOptimize.cuh"

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
    fs::path resultFolder(getarg<std::vector<std::string>>("outputFolder")[0]);
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

    fs::path resultFolder = fs::path(getarg<std::vector<std::string>>("outputFolder")[0]);
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

    fs::path resultFolder(getarg<std::vector<std::string>>("outputFolder")[0]);
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

    fs::path resultFolder(getarg<std::vector<std::string>>("outputFolder")[0]);
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


bool IMRT::test_MatFilter(const MatCSR32& matFilter, const MatCSR32& matFilterT) {
    // firstly, test the correctness of matFilterT
    int numRows = matFilterT.numRows;
    int numCols = matFilterT.numCols;
    cusparseDnVecDescr_t vecX, vecY;
    float* X = nullptr, *Y = nullptr;
    checkCudaErrors(cudaMalloc((void**)&X, numCols*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&Y, numRows*sizeof(float)));
    std::vector<float> h_X(numCols, 1.0f);
    checkCudaErrors(cudaMemcpy(X, h_X.data(),
        numCols*sizeof(float), cudaMemcpyHostToDevice));
    checkCusparse(cusparseCreateDnVec(&vecX, numCols, X, CUDA_R_32F));
    checkCusparse(cusparseCreateDnVec(&vecY, numRows, Y, CUDA_R_32F));

    cusparseHandle_t handle = nullptr;
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCusparse(cusparseCreate(&handle));
    // allocate an external buffer if needed
    size_t bufferSizeT = 0;
    void* dBuffer = nullptr;
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matFilterT.matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeT));
    checkCudaErrors(cudaMalloc(&dBuffer, bufferSizeT));
    checkCusparse(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matFilterT.matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // check results
    std::vector<float> h_Y(numRows, 0.0f);
    std::vector<float> h_Y_reference(numRows, 1.0f);
    checkCudaErrors(cudaMemcpy(h_Y.data(), Y, numRows*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i=0; i<numRows; i++) {
        if (abs(h_Y[i] - h_Y_reference[i]) > eps_fastdose) {
            std::cerr << "The result of matFilterT is not as expected." << std::endl;
            return 1;
        }
    }


    // secondly, test the correctness of matFilterT
    size_t bufferSize = 0;
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matFilter.matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > bufferSizeT) {
        checkCudaErrors(cudaFree(dBuffer));
        checkCudaErrors(cudaMalloc((void**)&dBuffer, bufferSize));
    }
    checkCusparse(cusparseSpMV(
        handle, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matFilter.matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // checkResults
    checkCudaErrors(cudaMemcpy(h_Y.data(), Y, numRows*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i=0; i<numRows; i++) {
        if (abs(h_Y[i] - h_Y_reference[i]) > eps_fastdose) {
            std::cerr << "The result of matFilter is not as expected." << std::endl;
            return 1;
        }
    }

    std::cout << "OAR filtering matrix test passed!" << std::endl;

    // clean up
    checkCudaErrors(cudaFree(dBuffer));
    checkCusparse(cusparseDestroy(handle))
    checkCusparse(cusparseDestroyDnVec(vecX));
    checkCusparse(cusparseDestroyDnVec(vecY))
    checkCudaErrors(cudaFree(X));
    checkCudaErrors(cudaFree(Y));
    return 0;
}


bool IMRT::test_SpMatOAR(const MatCSR64& SpOARmat, const MatCSR64& SpOARmatT,
    const MatCSR_Eigen& filter, const std::vector<MatCSR_Eigen>& OARMatrices) {
    // calculate the total number of beamlets
    size_t totalNumBeamlets = 0;
    for (size_t i=0; i<OARMatrices.size(); i++)
        totalNumBeamlets += OARMatrices[i].getCols();
    if (totalNumBeamlets != SpOARmat.numCols) {
        std::cerr << "The number of columns is inconsistent with other "
            "parts of the program." << std::endl;
    }


    // construct SpFilter
    MatCSR64 SpFilter;
    checkCudaErrors(cudaMalloc((void**)&SpFilter.d_csr_offsets, (filter.getRows()+1)*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&SpFilter.d_csr_columns, filter.getNnz()*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&SpFilter.d_csr_values, filter.getNnz()*sizeof(float)));
    checkCudaErrors(cudaMemcpy(SpFilter.d_csr_offsets, filter.getOffset(),
        (filter.getRows()+1)*sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(SpFilter.d_csr_columns, filter.getIndices(),
        filter.getNnz()*sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(SpFilter.d_csr_values, filter.getValues(),
        filter.getNnz()*sizeof(float), cudaMemcpyHostToDevice));
    checkCusparse(cusparseCreateCsr(
        &SpFilter.matA, filter.getRows(), filter.getCols(), filter.getNnz(),
        SpFilter.d_csr_offsets, SpFilter.d_csr_columns, SpFilter.d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    SpFilter.numRows = filter.getRows();
    SpFilter.numCols = filter.getCols();
    SpFilter.nnz = filter.getNnz();

    // construct weight matrix
    float* d_weight_data = nullptr;
    std::vector<float> h_weight_data(totalNumBeamlets, 1.0f);
    checkCudaErrors(cudaMalloc((void**)&d_weight_data, totalNumBeamlets*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_weight_data, h_weight_data.data(),
        totalNumBeamlets*sizeof(float), cudaMemcpyHostToDevice));
    cusparseDnVecDescr_t d_weight_vec;
    checkCusparse(cusparseCreateDnVec(&d_weight_vec, totalNumBeamlets, d_weight_data, CUDA_R_32F));

    float* d_result_data = nullptr;
    size_t nVoxels = SpOARmat.numRows;
    std::vector<float> h_result_data(nVoxels, 1.0f);
    checkCudaErrors(cudaMalloc((void**)&d_result_data, nVoxels*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_result_data, h_result_data.data(),
        nVoxels*sizeof(float), cudaMemcpyHostToDevice));
    cusparseDnVecDescr_t d_result_vec;
    checkCusparse(cusparseCreateDnVec(&d_result_vec, nVoxels, d_result_data, CUDA_R_32F));

    float* d_additional_data = nullptr;
    size_t totalVoxels = SpFilter.numRows;
    checkCudaErrors(cudaMalloc((void**)&d_additional_data, totalVoxels*sizeof(float)));
    std::vector<float> h_additional_data(totalVoxels, 0.0f);
    cusparseDnVecDescr_t d_additional_vec;
    checkCusparse(cusparseCreateDnVec(&d_additional_vec, totalVoxels,
        d_additional_data, CUDA_R_32F));

    cusparseHandle_t handle = nullptr;
    checkCusparse(cusparseCreate(&handle));
    // prepare buffer
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpOARmat.matA, d_weight_vec, &beta, d_result_vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

    // prepare additional buffer
    size_t additionalBufferSize = 0;
    void* dAdditionalBuffer = nullptr;
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpFilter.matA, d_result_vec, &beta, d_additional_vec, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &additionalBufferSize));
    checkCudaErrors(cudaMalloc((void**)&dAdditionalBuffer, additionalBufferSize));

    #if slicingTiming
        cudaEvent_t start, stop;
        float milliseconds = 0.0f;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
    #endif

    fs::path resultFolder(getarg<std::vector<std::string>>("outputFolder")[0]);
    resultFolder /= std::string("BeamDoseMat");
    if (! fs::is_directory(resultFolder))
        fs::create_directory(resultFolder);
    
    size_t localBegin = 0;
    for (size_t i=0; i<OARMatrices.size(); i++) {
        size_t localNumBeamlets = OARMatrices[i].getCols();
        std::fill(h_weight_data.begin(), h_weight_data.end(), 0.0f);
        std::fill(h_weight_data.begin() + localBegin, h_weight_data.begin()
            + localBegin + localNumBeamlets, 1.0f);
        localBegin += localNumBeamlets;
        checkCudaErrors(cudaMemcpy(d_weight_data, h_weight_data.data(),
            totalNumBeamlets*sizeof(float), cudaMemcpyHostToDevice));
        
        #if slicingTiming
            cudaEventRecord(start);
        #endif
        checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, SpOARmat.matA, d_weight_vec, &beta, d_result_vec, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        #if slicingTiming
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "Beamlet " << i << " multiplication, time elapsed: "
                << milliseconds * 0.001f << " [s]" << std::endl;
        #endif

        checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, SpFilter.matA, d_result_vec, &beta, d_additional_vec, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, dAdditionalBuffer));
        checkCudaErrors(cudaMemcpy(h_additional_data.data(), d_additional_data,
            totalVoxels*sizeof(float), cudaMemcpyDeviceToHost));
        fs::path file = resultFolder / (std::string("beam")
            + std::to_string(i) + std::string(".bin"));
        std::ofstream f(file.string());
        if (! f.is_open()) {
            std::cerr << "Cannot open file: " << file << std::endl;
            return 1;
        }
        f.write((char*)(h_additional_data.data()), totalVoxels*sizeof(float));
        f.close();
        std::cout << file << std::endl << std::endl;
    }

    // clean up
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaFree(dAdditionalBuffer));
    checkCudaErrors(cudaFree(dBuffer));
    checkCusparse(cusparseDestroy(handle));
    checkCusparse(cusparseDestroyDnVec(d_additional_vec));
    checkCudaErrors(cudaFree(d_additional_data));
    checkCusparse(cusparseDestroyDnVec(d_result_vec));
    checkCudaErrors(cudaFree(d_result_data));
    checkCusparse(cusparseDestroyDnVec(d_weight_vec));
    checkCudaErrors(cudaFree(d_weight_data));
    return 0;
}


bool IMRT::test_cusparseSlicing() {
    // firstly, construct a sparse matrix
    // 0  0  1  2
    // 1  0  0  1
    // 0  0  1  0
    // 2  0  0  0

    std::vector<size_t> offsets_h {0, 2, 4, 5, 6};
    std::vector<size_t> columns_h {2, 3, 0, 3, 2, 0};
    std::vector<float> values_h {1.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f};

    // construct an Eigen matrix for sanity check
    MatCSR_Eigen mat_h;
    EigenIdxType* mat_h_offsets = (EigenIdxType*)malloc(offsets_h.size()*sizeof(EigenIdxType));
    EigenIdxType* mat_h_columns = new EigenIdxType[columns_h.size()];
    float* mat_h_values = new float[values_h.size()];
    std::copy(offsets_h.begin(), offsets_h.end(), mat_h_offsets);
    std::copy(columns_h.begin(), columns_h.end(), mat_h_columns);
    std::copy(values_h.begin(), values_h.end(), mat_h_values);
    mat_h.customInit(4, 4, 6, mat_h_offsets, mat_h_columns, mat_h_values);
    std::cout << mat_h << std::endl;

    size_t* offsets_d = nullptr;
    size_t* columns_d = nullptr;
    float* values_d = nullptr;
    
    // construct a slicing
    MatCSR64 mat_slicing;
    checkCudaErrors(cudaMalloc((void**)&offsets_d, 3*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&columns_d, 6*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&values_d, 6*sizeof(float)));
    checkCudaErrors(cudaMemcpy(offsets_d, &offsets_h[2],
        3*sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(columns_d, columns_h.data(),
        columns_h.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(values_d, values_h.data(),
        values_h.size()*sizeof(float), cudaMemcpyHostToDevice));
    mat_slicing.numRows = 2;
    mat_slicing.numCols = 4;
    mat_slicing.nnz = 2;
    mat_slicing.d_csr_offsets = offsets_d;
    mat_slicing.d_csr_columns = columns_d;
    mat_slicing.d_csr_values = values_d;
    checkCusparse(cusparseCreateCsr(
        &mat_slicing.matA, mat_slicing.numRows, mat_slicing.numCols, mat_slicing.nnz,
        mat_slicing.d_csr_offsets, mat_slicing.d_csr_columns, mat_slicing.d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // construct dense vector
    cusparseDnVecDescr_t input;
    float* input_data = nullptr;
    std::vector<float> input_data_h(mat_slicing.numCols, 1.0f);
    checkCudaErrors(cudaMalloc((void**)&input_data, mat_slicing.numCols*sizeof(float)));
    checkCudaErrors(cudaMemcpy(input_data, input_data_h.data(),
        input_data_h.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCusparse(cusparseCreateDnVec(&input, input_data_h.size(), input_data, CUDA_R_32F));

    cusparseDnVecDescr_t output;
    float* output_data = nullptr;
    std::vector<float> output_data_h(mat_slicing.numRows, 1.0f);
    std::vector<float> output_data_ref(mat_slicing.numRows, 1.0f);
    checkCudaErrors(cudaMalloc((void**)&output_data, mat_slicing.numRows*sizeof(float)));
    checkCusparse(cusparseCreateDnVec(&output, mat_slicing.numRows, output_data, CUDA_R_32F));

    float alpha = 1.0f;
    float beta = 0.0f;
    size_t bufferSize = 0;
    cusparseHandle_t handle = nullptr;
    checkCusparse(cusparseCreate(&handle));
    checkCusparse(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_slicing.matA, input, &beta, output,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErrors(cudaMalloc((void**)&mat_slicing.d_buffer_spmv, bufferSize));

    int numCases = 100;
    for (int i=0; i<numCases; i++) {
        for (int j=0; j<input_data_h.size(); j++) {
            input_data_h[j] = std::rand() / RAND_MAX;
        }
        checkCudaErrors(cudaMemcpy(input_data, input_data_h.data(),
            input_data_h.size()*sizeof(float), cudaMemcpyHostToDevice));
        checkCusparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_slicing.matA, input, &beta, output, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, mat_slicing.d_buffer_spmv));
        checkCudaErrors(cudaMemcpy(output_data_h.data(), output_data,
            output_data_h.size()*sizeof(float), cudaMemcpyDeviceToHost));

        output_data_ref[0] = input_data_h[2];
        output_data_ref[1] = input_data_h[0] * 2;
        if (abs(output_data_h[0] - output_data_ref[0]) > eps_fastdose
            || abs(output_data_h[1] - output_data_ref[1]) > eps_fastdose) {
            std::cerr << "result unexpected." << std::endl;
            return 1;
        }
    }
    std::cout << "Cusparse slicing test passed!" << std::endl;
    return 0;
}