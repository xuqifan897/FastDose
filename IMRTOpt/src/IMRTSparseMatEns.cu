#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "IMRTDoseMat.cuh"
#include "IMRTDoseMatEns.cuh"


IMRT::MatCSREnsemble::MatCSREnsemble(
    const std::vector<size_t> numRowsPerMat_,
    size_t numColsPerMat_,
    size_t estBufferSize
) : numMatrices(numRowsPerMat_.size()), 
    numRowsPerMat(numRowsPerMat_),
    numColsPerMat(numColsPerMat_)
{
    this->bufferSize = estBufferSize;
    this->matA_array.reserve(this->numMatrices);
    this->NonZeroElements.reserve(this->numMatrices);
    this->CumuNonZeroElements.reserve(this->numMatrices);

    this->CumuNumRowsPerMat.resize(numMatrices);
    this->OffsetBufferIdx.resize(numMatrices);
    size_t prevNumRowsPerMat = 0;
    size_t prevOffsetBuffer = 0;
    for (int i=0; i<numMatrices; i++) {
        this->CumuNumRowsPerMat[i] = prevNumRowsPerMat + this->numRowsPerMat[i];
        prevNumRowsPerMat = this->CumuNumRowsPerMat[i];
        this->OffsetBufferIdx[i] = prevOffsetBuffer;
        prevOffsetBuffer += this->numRowsPerMat[i] + 1;
    }

    size_t offsetsBufferSize = 0;
    for (int i=0; i<this->numMatrices; i++) {
        offsetsBufferSize += this->numRowsPerMat[i] + 1;
    }
    checkCudaErrors(cudaMalloc((void**)(&this->d_offsetsBuffer),
        offsetsBufferSize * sizeof(size_t)));

    checkCudaErrors(cudaMalloc((void**)(&this->d_columnsBuffer),
        this->bufferSize * sizeof(size_t)));
    checkCudaErrors(cudaMalloc((void**)(&this->d_valuesBuffer),
        this->bufferSize * sizeof(float)));

    this->constructBufferSize = 0;
    this->d_constructBuffer = nullptr;
}


IMRT::MatCSREnsemble::~MatCSREnsemble() {
    if (this->numMatrices > 0 && this->d_offsetsBuffer != nullptr) {
        checkCudaErrors(cudaFree(this->d_offsetsBuffer));
    }
    if (this->bufferSize > 0) {
        if (this->d_columnsBuffer != nullptr)
            checkCudaErrors(cudaFree(this->d_columnsBuffer));
        if (this->d_valuesBuffer != nullptr)
            checkCudaErrors(cudaFree(this->d_valuesBuffer));
    }
    if (this->constructBufferSize > 0 && this->d_constructBuffer != nullptr)
        checkCudaErrors(cudaFree(d_constructBuffer));
}


bool IMRT::MatCSREnsemble::addMat(float* d_dense, size_t numRows, size_t numCols) {
    cusparseHandle_t handle = nullptr;
    checkCusparse(cusparseCreate(&handle));

    if (numCols != this->numColsPerMat) {
        std::cerr << "The number of columns of this dense matrix is " << numCols
            << ", which does not equal to that of the sparse matrix ensemble" << std::endl;
            return 1;
    }
    
    // dense matrix descriptor
    cusparseDnMatDescr_t matDense;
    checkCusparse(cusparseCreateDnMat(&matDense, numRows,
        numCols, numCols, d_dense,
        CUDA_R_32F, CUSPARSE_ORDER_ROW));
    

    // sparse matrix descriptor
    int matrixIdx = this->matA_array.size();
    this->matA_array.emplace_back(cusparseSpMatDescr_t());
    cusparseSpMatDescr_t& matSparse = this->matA_array.back();

    size_t* d_csr_offsets = this->d_offsetsBuffer + this->OffsetBufferIdx[matrixIdx];
    checkCusparse(cusparseCreateCsr(&matSparse, numRows, numCols, 0,
        d_csr_offsets, nullptr, nullptr, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    

    // allocate an external buffer if needed.
    size_t bufferSize = 0;
    checkCusparse(cusparseDenseToSparse_bufferSize(
        handle, matDense, matSparse,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &bufferSize));
    // re-allocate buffer if the size is not big enough
    if (bufferSize > this->constructBufferSize) {
        this->constructBufferSize = bufferSize;
        checkCudaErrors(cudaFree(this->d_constructBuffer));
        checkCudaErrors(cudaMalloc((void**)(&this->d_constructBuffer), bufferSize));
    }

    // execute dense to sparse conversion
    checkCusparse(cusparseDenseToSparse_analysis(handle, matDense, matSparse,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, this->d_constructBuffer));
    
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    checkCusparse(cusparseSpMatGetSize(matSparse, &num_rows_tmp, &num_cols_tmp, &nnz));

    // allocate CSR column indices and values
    this->NonZeroElements.push_back(nnz);
    size_t valueOffset = 0;
    if (this->CumuNonZeroElements.size() == 0)
        this->CumuNonZeroElements.push_back(nnz);
    else {
        valueOffset = this->CumuNonZeroElements.back();
        size_t cumuValue = nnz + valueOffset;
        this->CumuNonZeroElements.push_back(cumuValue);
    }
    if (this->CumuNonZeroElements.back() > this->bufferSize) {
        size_t bufferSizeOld = this->bufferSize;
        while (this->bufferSize < this->CumuNonZeroElements.back())
            this->bufferSize *= 2;
        std::cout << "Pre-allocated buffer is full. To re-allocate the buffer, "
            "from " << bufferSizeOld << " to " << this->bufferSize << " elements." << std::endl;

        size_t* d_columnsBuffer_new = nullptr;
        checkCudaErrors(cudaMalloc(
            (void**)(&d_columnsBuffer_new), this->bufferSize*sizeof(size_t)));
        checkCudaErrors(cudaMemcpy(d_columnsBuffer_new,
            this->d_columnsBuffer, bufferSizeOld*sizeof(size_t),
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaFree(this->d_columnsBuffer));
        this->d_columnsBuffer = d_columnsBuffer_new;

        float* d_valuesBuffer_new = nullptr;
        checkCudaErrors(cudaMalloc(
            (void**)(&d_valuesBuffer_new), this->bufferSize*sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_valuesBuffer_new,
            this->d_valuesBuffer, bufferSizeOld*sizeof(float),
            cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaFree(this->d_valuesBuffer));
        this->d_valuesBuffer = d_valuesBuffer_new;
    }

    size_t* d_csr_columns = this->d_columnsBuffer + valueOffset;
    float* d_csr_values = this->d_valuesBuffer + valueOffset;
    // reset offsets, column indices, and value pointers
    checkCusparse(cusparseCsrSetPointers(
        matSparse, d_csr_offsets, d_csr_columns, d_csr_values));
    
    // execute dense to sparse conversion
    checkCusparse(cusparseDenseToSparse_convert(
        handle, matDense, matSparse,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        this->d_constructBuffer));

    checkCusparse(cusparseDestroy(handle));
    return 0;
}


bool IMRT::MatCSREnsemble::tofile(const std::string& resultFolder) {
    if (! fs::is_directory(resultFolder))
        fs::create_directory(resultFolder);

    // firstly, output the arrays of size numMatrices
    std::vector<std::pair<std::vector<size_t>&, std::string>> outputQueue {
        {this->NonZeroElements, std::string("NonZeroElements")},
        {this->CumuNonZeroElements, std::string("CumuNonZeroElements")},
        {this->numRowsPerMat, std::string("numRowsPerMat")},
        {this->CumuNumRowsPerMat, std::string("CumuNumRowsPerMat")},
        {this->OffsetBufferIdx, std::string("OffsetBufferIdx")}
    };
    for (const auto a : outputQueue) {
        const auto & array = a.first;
        const std::string & name = a.second;
        fs::path fullFile = fs::path(resultFolder) / (name + std::string(".bin")); 
        std::ofstream f(fullFile.string());
        if (! f.is_open()) {
            std::cerr << "Cannot open file: " << fullFile << std::endl;
            return 1;
        }
        f.write((char*)(array.data()), this->numMatrices*sizeof(size_t));
        f.close();
        std::cout << "Saving \"" << fullFile << "\" completes." << std::endl;
    }

    // secondly, log out d_offsetsBuffer, which contains the row information
    size_t nRows = this->OffsetBufferIdx.back() + numRowsPerMat.back() + 1;
    std::vector<size_t> h_offsetsBuffer(nRows, 0);
    checkCudaErrors(cudaMemcpy(h_offsetsBuffer.data(), this->d_offsetsBuffer,
        nRows * sizeof(size_t), cudaMemcpyDeviceToHost));
    fs::path fullFile = fs::path(resultFolder) / std::string("offsetsBuffer.bin");
    std::ofstream f(fullFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << fullFile << std::endl;
        return 1;
    }
    f.write((char*)(h_offsetsBuffer.data()), nRows*sizeof(size_t));
    f.close();
    std::cout << "Saving \"" << fullFile << "\" completes." << std::endl;

    // thirdly, log out d_columnsBuffer and d_valuesBuffer, both of
    // the size of the total number of non-zero elements
    size_t totalNnz = this->CumuNonZeroElements.back();
    std::vector<size_t> h_columnsBuffer(totalNnz, 0);
    checkCudaErrors(cudaMemcpy(h_columnsBuffer.data(), this->d_columnsBuffer,
        totalNnz * sizeof(size_t), cudaMemcpyDeviceToHost));
    fullFile = fs::path(resultFolder) / std::string("columnsBuffer.bin");
    f.open(fullFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << fullFile << std::endl;
        return 1;
    }
    f.write((char*)(h_columnsBuffer.data()), totalNnz*sizeof(size_t));
    f.close();
    std::cout << "Saving \"" << fullFile << "\" completes." << std::endl;

    std::vector<float> h_valuesBuffer(totalNnz, 0.0f);
    checkCudaErrors(cudaMemcpy(h_valuesBuffer.data(), this->d_valuesBuffer,
        totalNnz*sizeof(float), cudaMemcpyDeviceToHost));
    fullFile = fs::path(resultFolder) / std::string("valuesBuffer.bin");
    f.open(fullFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << fullFile << std::endl;
        return 1;
    }
    f.write((char*)(h_valuesBuffer.data()), totalNnz*sizeof(float));
    f.close();
    std::cout << "Saving \"" << fullFile << "\" completes." << std::endl;

    return 0;
}


bool IMRT::MatCSREnsemble::fromfile(const std::string& resultFolder) {
    cudaEvent_t start, stop;
    float milliseconds;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    cudaEventRecord(start);

    // like above, we first read the arrays of size numMatrices
    std::vector<std::pair<std::vector<size_t>&, std::string>> inputQueue {
        {this->NonZeroElements, std::string("NonZeroElements")},
        {this->CumuNonZeroElements, std::string("CumuNonZeroElements")},
        {this->numRowsPerMat, std::string("numRowsPerMat")},
        {this->CumuNumRowsPerMat, std::string("CumuNumRowsPerMat")},
        {this->OffsetBufferIdx, std::string("OffsetBufferIdx")}
    };

    for (auto& a : inputQueue) {
        std::vector<size_t>& array = a.first;
        const std::string& name = a.second;
        fs::path fullFile = fs::path(resultFolder) / (name + std::string(".bin"));
        std::ifstream f(fullFile.string());
        if (! f.is_open()) {
            std::cerr << "Cannot open file: " << fullFile << std::endl;
            return 1;
        }
        f.seekg(0, std::ios::end);
        array.resize(f.tellg()/sizeof(size_t));
        f.seekg(0, std::ios::beg);
        f.read((char*)(array.data()), array.size()*sizeof(size_t));
        f.close();
    }
    this->numMatrices = this->NonZeroElements.size();
    for (const auto& a : inputQueue) {
        if (a.first.size() != this->numMatrices) {
            std::cerr << "The size of \"" << a.second << "\" does not equal to "
                << this->numMatrices << std::endl;
            return 1;
        }
    }

    // fill offsetsBuffer
    fs::path offsetsBufferFile = fs::path(resultFolder) / std::string("offsetsBuffer.bin");
    std::ifstream f(offsetsBufferFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << offsetsBufferFile << std::endl;
        return 1;
    }
    f.seekg(0, std::ios::end);
    size_t nRows = f.tellg() / sizeof(size_t);
    std::vector<size_t> h_offsetsBuffer(nRows, 0);
    f.seekg(0, std::ios::beg);
    f.read((char*)(h_offsetsBuffer.data()), nRows*sizeof(size_t));
    f.close();
    checkCudaErrors(cudaMalloc((void**)&this->d_offsetsBuffer, nRows*sizeof(size_t)));
    checkCudaErrors(cudaMemcpyAsync(this->d_offsetsBuffer, h_offsetsBuffer.data(),
        nRows*sizeof(size_t), cudaMemcpyHostToDevice));

    // fill d_columnsBuffer and d_valuesBuffer
    fs::path columnsBufferFile = fs::path(resultFolder) / std::string("columnsBuffer.bin");
    f.open(columnsBufferFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << columnsBufferFile << std::endl;
        return 1;
    }
    f.seekg(0, std::ios::end);
    size_t nnz = f.tellg() / sizeof(size_t);
    std::vector<size_t> h_columnsBuffer(nnz, 0);
    f.seekg(0, std::ios::beg);
    f.read((char*)(h_columnsBuffer.data()), nnz*sizeof(size_t));
    f.close();
    checkCudaErrors(cudaMalloc((void**)&this->d_columnsBuffer, nnz*sizeof(size_t)));
    checkCudaErrors(cudaMemcpyAsync(this->d_columnsBuffer, h_columnsBuffer.data(),
        nnz*sizeof(size_t), cudaMemcpyHostToDevice));
    
    fs::path valuesBufferFile = fs::path(resultFolder) / std::string("valuesBuffer.bin");
    f.open(valuesBufferFile.string());
    if (! f.is_open()) {
        std::cerr << "Cannot open file: " << valuesBufferFile << std::endl;
        return 1;
    }
    std::vector<float> h_valuesBuffer(nnz, 0);
    f.read((char*)(h_valuesBuffer.data()), nnz*sizeof(float));
    f.close();
    checkCudaErrors(cudaMalloc((void**)&this->d_valuesBuffer, nnz*sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(this->d_valuesBuffer, h_valuesBuffer.data(),
        nnz*sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    std::cout << "Sparse matrix initialiation completed. Time elapsed: "
        << std::fixed << milliseconds * 0.001f << " [s]" << std::endl << std::endl;
    return 0;
}