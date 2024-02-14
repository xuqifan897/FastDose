#include "IMRTDoseMat.cuh"
#include "IMRTDoseMatEigen.cuh"
#include "IMRTOptimize.cuh"
#include "IMRTOptimize_var.cuh"
#include "IMRTOptimize_var.h"
#include <iomanip>
#include <chrono>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

bool IMRT::OARFiltering(
    const std::string& resultFolder, const std::vector<StructInfo>& structs,
    std::vector<MatCSR_Eigen>& VOIMatrices, std::vector<MatCSR_Eigen>& VOIMatricesT,
    Weights_h& weights
) {
    MatCSR_Eigen filter, filterT;
    if (getStructFilter(filter, filterT, structs, weights)) {
        std::cerr << "OAR filter and its transpose construction error." << std::endl;
        return 1;
    }
    if (parallelSpGEMM(resultFolder, filter, filterT, VOIMatrices, VOIMatricesT)) {
        std::cerr << "CPU VOI dose loading matrices and their transpose "
            "construction error." << std::endl;
        return 1;
    }
    return 0;
}


size_t IMRT::sizeEstimate(
    const std::vector<MatCSR_Eigen>& VOIMatrices,
    const std::vector<MatCSR_Eigen>& VOIMatricesT,
    const std::vector<MatCSR_Eigen>& SpFluenceGrad,
    const std::vector<MatCSR_Eigen>& SpFluenceGradT) {
    
    std::vector<const std::vector<MatCSR_Eigen>*> iterator
        {&VOIMatrices, &VOIMatricesT, &SpFluenceGrad, &SpFluenceGradT};
    
    size_t size = 0;
    for (const std::vector<MatCSR_Eigen>* ptr : iterator) {
        int numMatrices = ptr->size();
        for (int i=0; i<numMatrices; i++) {
            const MatCSR_Eigen& localMat = (*ptr)[i];
            size_t localRows = localMat.getRows();
            size_t localNnz = localMat.getNnz();
            size_t localSize = (localRows + 1 + localNnz) * sizeof(size_t)
                + localNnz * sizeof(float);
            size += localSize;
        }
    }
    return size;
}


bool IMRT::MatReservior::load(const std::vector<MatCSR_Eigen>& source) {
    this->reservior.resize(source.size());
    for (int i=0; i<source.size(); i++) {
        MatCSR64& current = this->reservior[i];
        current.numRows = 0;
        current.numCols = 0;
        current.nnz = 0;
        current.matA = nullptr;
        current.d_csr_offsets = nullptr;
        current.d_csr_columns = nullptr;
        current.d_csr_values = nullptr;

        if(Eigen2Cusparse(source[i], current))
            return 1;
    }
    return 0;
}


bool IMRT::MatReservior::assemble_row_block(MatCSR64& target,
    const std::vector<uint8_t>& flags ) const {
    // firstly, check if the target is empty
    if (target.matA != nullptr || target.d_csr_offsets != nullptr ||
        target.d_csr_columns != nullptr || target.d_csr_values != nullptr ||
        target.d_buffer_spmv != nullptr) {
        std::cerr << "The target is not an empty matrix." << std::endl;
        return 1;
    }
    if (flags.size() != this->reservior.size()) {
        std::cerr << "The size of the input vector flags should be the same "
            "as this->reservior." << std::endl;
        return 1;
    }

    size_t total_nnz = 0;
    size_t total_rows = 0;
    size_t columns = 0;
    for (int i=0; i<this->reservior.size(); i++) {
        if (flags[i] == 0)
            continue;
        const MatCSR64& res = this->reservior[i];
        total_nnz += res.nnz;
        total_rows += res.numRows;
        if (columns == 0)
            columns = res.numCols;
        else if (columns != res.numCols) {
            std::cerr << "The number of columns inconsistent amoung matrices." << std::endl;
            return 1;
        }
    }

    target.numRows = total_rows;
    target.numCols = columns;
    target.nnz = total_nnz;
    checkCudaErrors(cudaMalloc((void**)&target.d_csr_offsets, (total_rows+1)*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&target.d_csr_columns, total_nnz*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&target.d_csr_values, total_nnz*sizeof(float)));

    checkCudaErrors(cudaMemset(target.d_csr_offsets, 0, sizeof(size_t)));
    size_t nnz_offset = 0;
    size_t row_offset = 0;
    std::vector<size_t> cumu_row;
    std::vector<size_t> cumu_nnz;
    cumu_row.reserve(this->reservior.size()+1);
    cumu_nnz.reserve(this->reservior.size()+1);
    cumu_row.push_back(0);
    cumu_nnz.push_back(0);
    for (int i=0; i<this->reservior.size(); i++) {
        if (flags[i] == 0)
            continue;
        const MatCSR64& res = this->reservior[i];
        size_t local_nnz = res.nnz;
        size_t local_rows = res.numRows;
        const size_t* local_offsets = res.d_csr_offsets;
        const size_t* local_columns = res.d_csr_columns;
        const float* local_values = res.d_csr_values;

        checkCudaErrors(cudaMemcpyAsync(target.d_csr_columns + nnz_offset, local_columns,
            local_nnz*sizeof(size_t), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyAsync(target.d_csr_values + nnz_offset, local_values,
            local_nnz*sizeof(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyAsync(target.d_csr_offsets + row_offset + 1, local_offsets + 1,
            local_rows*sizeof(size_t), cudaMemcpyDeviceToDevice));
        
        nnz_offset += local_nnz;
        row_offset += local_rows;

        cumu_row.push_back(row_offset);
        cumu_nnz.push_back(nnz_offset);
    }

    size_t* cumu_row_d = nullptr;
    size_t* cumu_nnz_d = nullptr;
    checkCudaErrors(cudaMalloc((void**)&cumu_row_d, cumu_row.size()*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&cumu_nnz_d, cumu_nnz.size()*sizeof(size_t)));
    checkCudaErrors(cudaMemcpy(cumu_row_d, cumu_row.data(),
        cumu_row.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(cumu_nnz_d, cumu_nnz.data(),
        cumu_nnz.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (cumu_row.size() - 1 + blockSize.x - 1) / blockSize.x;
    d_assembly_row_block<<<gridSize, blockSize>>>(
        target.d_csr_offsets, cumu_row_d, cumu_nnz_d, cumu_row.size()-1);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCusparse(cusparseCreateCsr(
        &target.matA, target.numRows, target.numCols, target.nnz,
        target.d_csr_offsets, target.d_csr_columns, target.d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // clean up
    checkCudaErrors(cudaFree(cumu_row_d));
    checkCudaErrors(cudaFree(cumu_nnz_d));
    return 0;
}


__global__ void
IMRT::d_assembly_row_block(size_t* d_csr_offsets,
    size_t* cumu_row, size_t* cumu_nnz, size_t numMatrices) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numMatrices)
        return;

    size_t row_start = cumu_row[idx];
    size_t row_end = cumu_row[idx + 1];
    size_t nnz_offset = cumu_nnz[idx];

    for (size_t i=row_start; i<row_end; i++) {
        size_t local = d_csr_offsets[i+1] + nnz_offset;
        d_csr_offsets[i+1] = local;
    }
}


bool IMRT::MatReservior_dev(
    const std::vector<MatCSR_Eigen>& VOIMatrices,
    const std::vector<MatCSR_Eigen>& VOIMatricesT,
    const std::vector<MatCSR_Eigen>& SpFluenceGrad,
    const std::vector<MatCSR_Eigen>& SpFluenceGradT
) {
    IMRT::MatReservior VOIReservior, VOIReserviorT, FGReservior, FGReserviorT;
    #if true
    // estimate size
        size_t totalSize = IMRT::sizeEstimate(VOIMatrices, VOIMatricesT,
            SpFluenceGrad, SpFluenceGradT);
        std::cout << "Total size: " << (float)totalSize / (1<<30) << " G" << std::endl;
    #endif
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    if (VOIReservior.load(VOIMatrices) ||
        VOIReserviorT.load(VOIMatricesT) ||
        FGReservior.load(SpFluenceGrad) ||
        FGReservior.load(SpFluenceGradT)) {
        std::cerr << "Loading data from CPU to GPU error." << std::endl;
    }
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0);
        std::cout << std::setprecision(4) << "Loading data from CPU to GPU time elapsed: " 
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif

    std::vector<uint8_t> flags(VOIReserviorT.reservior.size(), 1);
    MatCSR64 VOIMatT;
    if (VOIReserviorT.assemble_row_block(VOIMatT, flags))
        return 1;
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1);
        std::cout << "Assembly_row_block time elapsed: "
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif

    #if false
        // verify against MatCSR_Eigen
        std::cout << "\nATrans Benchmarking starts..." << std::endl;
        // for comparision
        MatCSR_Eigen VOIMat_Eigen;
        MatCSR_Eigen VOIMatT_Eigen;
        MatCSR_Eigen D_Eigen;
        MatCSR_Eigen DTrans_Eigen;
        std::vector<MatCSR_Eigen*> VOIMatrice_ptr(VOIMatricesT.size(), nullptr);
        std::vector<MatCSR_Eigen*> VOIMatriceT_ptr(VOIMatricesT.size(), nullptr);
        std::vector<MatCSR_Eigen*> SpFluenceGrad_ptr(VOIMatricesT.size(), nullptr);
        std::vector<MatCSR_Eigen*> SpFluenceGradT_ptr(VOIMatricesT.size(), nullptr);
        for (int i=0; i<VOIMatricesT.size(); i++) {
            VOIMatrice_ptr[i] = (MatCSR_Eigen*)&VOIMatrices[i];
            VOIMatriceT_ptr[i] = (MatCSR_Eigen*)&VOIMatricesT[i];
            SpFluenceGrad_ptr[i] = (MatCSR_Eigen*)&SpFluenceGrad[i];
            SpFluenceGradT_ptr[i] = (MatCSR_Eigen*)&SpFluenceGradT[i];
        }
        matFuseFunc(VOIMatrice_ptr, VOIMatriceT_ptr, SpFluenceGrad_ptr, SpFluenceGradT_ptr,
            VOIMat_Eigen, VOIMatT_Eigen, D_Eigen, DTrans_Eigen);

        std::vector<size_t> VOIMatT_offsets(VOIMatT.numRows + 1);
        std::vector<size_t> VOIMatT_columns(VOIMatT.nnz);
        std::vector<float> VOIMatT_values(VOIMatT.nnz);
        checkCudaErrors(cudaMemcpy(VOIMatT_offsets.data(), VOIMatT.d_csr_offsets,
            VOIMatT_offsets.size()*sizeof(size_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(VOIMatT_columns.data(), VOIMatT.d_csr_columns,
            VOIMatT_columns.size()*sizeof(size_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(VOIMatT_values.data(), VOIMatT.d_csr_values,
            VOIMatT_values.size()*sizeof(float), cudaMemcpyDeviceToHost));

        EigenIdxType* ref_offsets = *VOIMatT_Eigen.getOffset();
        const EigenIdxType* ref_columns = VOIMatT_Eigen.getIndices();
        const float* ref_values = VOIMatT_Eigen.getValues();

        for (size_t i=0; i<VOIMatT_offsets.size(); i++) {
            if (VOIMatT_offsets[i] != ref_offsets[i]) {
                std::cerr << "Offsets unmatch at i=" << i << ", VOIMatT_offsets[i]=="
                    << VOIMatT_offsets[i] << ", ref_offsets[i]==" << ref_offsets[i] << std::endl;
                return 1;
            }
        }
        for (size_t i=0; i<VOIMatT_columns.size(); i++) {
            if (VOIMatT_columns[i] != ref_columns[i] ||
                std::abs(VOIMatT_values[i] - ref_values[i]) > 1e-4f) {
                std::cerr << "Element unmatch at i=" << i << " test: (" << VOIMatT_columns[i]
                    << ", " << VOIMatT_values[i] << "), reference: (" << ref_columns[i]
                    << ", " << ref_values[i] << ")" << std::endl;
                return 1;
            }
        }
    #endif

    #if true
        // VOIMatT verification
        array_1d<float> input, output_full, output_slice;
        size_t sliceIdx = 63;
        const MatCSR64& slice = VOIReserviorT.reservior[sliceIdx];
        arrayInit(input, VOIMatT.numCols);
        arrayInit(output_full, VOIMatT.numRows);
        arrayInit(output_slice, slice.numRows);
        arrayRand01(input);

        cusparseHandle_t handle;
        checkCusparse(cusparseCreate(&handle));
        float alpha = 1.0f;
        float beta = 0.0f;

        void* buffer;
        size_t bufferSize;
        checkCusparse(cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, VOIMatT.matA, input.vec, &beta, output_full.vec,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        checkCudaErrors(cudaMalloc((void**)&buffer, bufferSize));

        checkCusparse(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, VOIMatT.matA, input.vec, &beta, output_full.vec,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
        checkCusparse(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, slice.matA, input.vec, &beta, output_slice.vec,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        // compare results
        size_t resultOffset = 0;
        for (int i=0; i<sliceIdx; i++)
            resultOffset += VOIReserviorT.reservior[i].numRows;
        std::vector<float> result_full(output_slice.size);
        std::vector<float> result_slice(output_slice.size);
        checkCudaErrors(cudaMemcpy(result_full.data(), output_full.data + resultOffset,
            result_full.size()*sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(result_slice.data(), output_slice.data,
            result_slice.size()*sizeof(float), cudaMemcpyDeviceToHost));
        
        for (size_t i=0; i<result_full.size(); i++) {
            if (abs(result_full[i] - result_slice[i]) > 1e-4f * result_slice[i]) {
                std::cerr << "Result unmatch at index: " << i << ", result_full[i] == "
                    << result_full[i] << ", result_slice[i] == " << result_slice[i] << std::endl;
                return 1;
            }
        }
        checkCudaErrors(cudaFree(buffer));
        checkCusparse(cusparseDestroy(handle));

        std::cout << "assemble_row_block passed the test!" << std::endl;
    #endif

    return 0;
}


bool IMRT::MatReservior::assemble_col_block(MatCSR64& target,
    const std::vector<MatCSR_Eigen>& reservior_h,
    const std::vector<uint8_t>& flags) const {
    // firstly, check if the target is empty
    if (target.matA != nullptr || target.d_csr_offsets != nullptr ||
        target.d_csr_columns != nullptr || target.d_csr_values != nullptr ||
        target.d_buffer_spmv != nullptr) {
        std::cerr << "The target is not an empty matrix." << std::endl;
        return 1;
    }
    if (flags.size() != this->reservior.size()) {
        std::cerr << "The size of the input vector flags should be the same "
            "as this->reservior." << std::endl;
        return 1;
    }
    size_t numRows = 0;
    size_t numCols = 0;
    size_t total_nnz = 0;
    size_t numMatrices = 0;
    std::vector<size_t> cumu_nnz;
    #if false
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    assemble_col_block_meta(numRows, numCols, total_nnz, numMatrices,
        cumu_nnz, flags, reservior_h);
    #if false
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0);
        std::cout << "Function assemble_col_block_meta time elapsed: "
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif

    target.numRows = numRows;
    target.numCols = numCols;
    target.nnz = total_nnz;
    checkCudaErrors(cudaMalloc((void**)&target.d_csr_offsets, (target.numRows+1)*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&target.d_csr_columns, target.nnz*sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)&target.d_csr_values, target.nnz*sizeof(float)));

    if (this->reservior.size() != reservior_h.size()) {
        std::cerr << "this->reservior.size() != reservior_h.size()" << std::endl;
        return 1;
    }

    checkCudaErrors(cudaMemcpy(target.d_csr_offsets, cumu_nnz.data(),
        (target.numRows+1)*sizeof(size_t), cudaMemcpyHostToDevice));
    
    size_t** source_offsets = nullptr;
    size_t** source_columns = nullptr;
    float** source_values = nullptr;
    size_t* source_columns_offset = nullptr;
    checkCudaErrors(cudaMalloc((void**)&source_offsets, numMatrices*sizeof(size_t*)));
    checkCudaErrors(cudaMalloc((void**)&source_columns, numMatrices*sizeof(size_t*)));
    checkCudaErrors(cudaMalloc((void**)&source_values, numMatrices*sizeof(float*)));
    checkCudaErrors(cudaMalloc((void**)&source_columns_offset, numMatrices*sizeof(size_t)));
    std::vector<size_t*> h_source_offsets(numMatrices, nullptr);
    std::vector<size_t*> h_source_columns(numMatrices, nullptr);
    std::vector<float*> h_source_values(numMatrices, nullptr);
    std::vector<size_t> h_source_columns_offset(numMatrices + 1, 0);
    int idx = 0;
    for (int i=0; i<this->reservior.size(); i++) {
        if (flags[i] == 0)
            continue;
        const MatCSR64& res = this->reservior[i];
        h_source_offsets[idx] = res.d_csr_offsets;
        h_source_columns[idx] = res.d_csr_columns;
        h_source_values[idx] = res.d_csr_values;
        h_source_columns_offset[idx + 1] = h_source_columns_offset[idx] + res.numCols;
        idx ++;
    }
    checkCudaErrors(cudaMemcpy(source_offsets, h_source_offsets.data(),
        numMatrices*sizeof(size_t*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(source_columns, h_source_columns.data(),
        numMatrices*sizeof(size_t*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(source_values, h_source_values.data(),
        numMatrices*sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(source_columns_offset, h_source_columns_offset.data(),
        numMatrices*sizeof(size_t), cudaMemcpyHostToDevice));

    dim3 blockSize(64, 1, 1);
    dim3 gridSize(1, 1, 1);
    gridSize.x = (target.numRows + blockSize.x - 1) / blockSize.x;
    d_assembly_col_block<<<gridSize, blockSize>>>(
        target.d_csr_offsets, target.d_csr_columns, target.d_csr_values,
        source_offsets, source_columns, source_values,
        source_columns_offset, numRows, numMatrices);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCusparse(cusparseCreateCsr(
        &target.matA, target.numRows, target.numCols, target.nnz,
        target.d_csr_offsets, target.d_csr_columns, target.d_csr_values,
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // clean up
    checkCudaErrors(cudaFree(source_offsets));
    checkCudaErrors(cudaFree(source_columns));
    checkCudaErrors(cudaFree(source_values));
    checkCudaErrors(cudaFree(source_columns_offset));
    return 0;
}


__global__ void
IMRT::d_assembly_col_block(size_t* d_csr_offsets, size_t* d_csr_columns, float* d_csr_values,
    size_t** source_offsets, size_t** source_columns, float** source_values,
    size_t* source_columns_offset, size_t numRows, size_t numMatrices) {
    size_t row_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (row_idx >= numRows)
        return;
    
    size_t idx_start = d_csr_offsets[row_idx];
    for (size_t i=0; i<numMatrices; i++) {
        size_t* current_source_offsets = source_offsets[i];
        size_t* current_source_columns = source_columns[i];
        float* current_source_values = source_values[i];
        size_t current_columns_offset = source_columns_offset[i];

        size_t current_idx_start = current_source_offsets[row_idx];
        size_t current_idx_end = current_source_offsets[row_idx+1];
        for (size_t element_idx=current_idx_start; element_idx<current_idx_end; element_idx++) {
            d_csr_columns[idx_start] = current_source_columns[element_idx] + current_columns_offset;
            d_csr_values[idx_start] = current_source_values[element_idx];
            idx_start ++;
        }
    }
}


bool IMRT::MatReservior_dev_col(
    const std::vector<MatCSR_Eigen>& VOIMatrices,
    const std::vector<MatCSR_Eigen>& VOIMatricesT,
    const std::vector<MatCSR_Eigen>& SpFluenceGrad,
    const std::vector<MatCSR_Eigen>& SpFluenceGradT
) {
    IMRT::MatReservior VOIReservior, VOIReserviorT, FGReservior, FGReserviorT;
    #if true
    // estimate size
        size_t totalSize = IMRT::sizeEstimate(VOIMatrices, VOIMatricesT,
            SpFluenceGrad, SpFluenceGradT);
        std::cout << "Total size: " << (float)totalSize / (1<<30) << " G" << std::endl;
    #endif
    #if slicingTiming
        auto time0 = std::chrono::high_resolution_clock::now();
    #endif
    if (VOIReservior.load(VOIMatrices) ||
        VOIReserviorT.load(VOIMatricesT) ||
        FGReservior.load(SpFluenceGrad) ||
        FGReservior.load(SpFluenceGradT)) {
        std::cerr << "Loading data from CPU to GPU error." << std::endl;
    }
    #if slicingTiming
        auto time1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0);
        std::cout << std::setprecision(4) << "Loading data from CPU to GPU time elapsed: " 
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif
    std::vector<uint8_t> flags(VOIReservior.reservior.size(), 1);
    MatCSR64 VOIMat;
    if (VOIReservior.assemble_col_block(VOIMat, VOIMatrices, flags))
        return 1;
    #if slicingTiming
        auto time2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1);
        std::cout << "Assembly_col_block time elapsed: "
            << duration.count() * 1e-6f << " [s]" << std::endl;
    #endif

    #if false
        // verify against MatCSR_Eigen
        std::cout << "\nATrans Benchmarking starts..." << std::endl;
        // for comparision
        MatCSR_Eigen VOIMat_Eigen;
        MatCSR_Eigen VOIMatT_Eigen;
        MatCSR_Eigen D_Eigen;
        MatCSR_Eigen DTrans_Eigen;
        std::vector<MatCSR_Eigen*> VOIMatrice_ptr(VOIMatricesT.size(), nullptr);
        std::vector<MatCSR_Eigen*> VOIMatriceT_ptr(VOIMatricesT.size(), nullptr);
        std::vector<MatCSR_Eigen*> SpFluenceGrad_ptr(VOIMatricesT.size(), nullptr);
        std::vector<MatCSR_Eigen*> SpFluenceGradT_ptr(VOIMatricesT.size(), nullptr);
        for (int i=0; i<VOIMatricesT.size(); i++) {
            VOIMatrice_ptr[i] = (MatCSR_Eigen*)&VOIMatrices[i];
            VOIMatriceT_ptr[i] = (MatCSR_Eigen*)&VOIMatricesT[i];
            SpFluenceGrad_ptr[i] = (MatCSR_Eigen*)&SpFluenceGrad[i];
            SpFluenceGradT_ptr[i] = (MatCSR_Eigen*)&SpFluenceGradT[i];
        }
        matFuseFunc(VOIMatrice_ptr, VOIMatriceT_ptr, SpFluenceGrad_ptr, SpFluenceGradT_ptr,
            VOIMat_Eigen, VOIMatT_Eigen, D_Eigen, DTrans_Eigen);

        std::vector<size_t> VOIMat_offsets(VOIMat.numRows + 1);
        std::vector<size_t> VOIMat_columns(VOIMat.nnz);
        std::vector<float> VOIMat_values(VOIMat.nnz);
        checkCudaErrors(cudaMemcpy(VOIMat_offsets.data(), VOIMat.d_csr_offsets,
            VOIMat_offsets.size()*sizeof(size_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(VOIMat_columns.data(), VOIMat.d_csr_columns,
            VOIMat_columns.size()*sizeof(size_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(VOIMat_values.data(), VOIMat.d_csr_values,
            VOIMat_values.size()*sizeof(float), cudaMemcpyDeviceToHost));

        EigenIdxType* ref_offsets = *VOIMat_Eigen.getOffset();
        const EigenIdxType* ref_columns = VOIMat_Eigen.getIndices();
        const float* ref_values = VOIMat_Eigen.getValues();

        for(size_t i=0; i<VOIMat_offsets.size(); i++) {
            if (VOIMat_offsets[i] != ref_offsets[i]) {
                std::cerr << "Offsets unmatch at i=" << i << ", VOIMat_offsets[i]=="
                    << VOIMat_offsets[i] << ", ref_offsets[i]==" << ref_offsets[i] << std::endl;
                return 1;
            }
        }

        for (size_t i=0; i<VOIMat_columns.size(); i++) {
            if (VOIMat_columns[i] != ref_columns[i] ||
                std::abs(VOIMat_values[i] - ref_values[i]) > 1e-4f) {
                std::cerr << "Element unmatch at i=" << i << ", test: (" << VOIMat_columns[i]
                    << ", " << VOIMat_values[i] << "), reference: (" << ref_columns[i]
                    << ", " << ref_values[i] << ")" << std::endl;
                return 1;
            }
        }
    std::cout << "assemble_col_block passed the test!" << std::endl;
    #endif

    return 0;
}