#include "density.cuh"
#include "utils.cuh"
#include <iomanip>
namespace fd = fastdose;

std::ostream& fd::operator<<(std::ostream& os, const DENSITY_h& obj) {
    os << "voxel size: " << obj.VoxelSize << std::endl;
    os << "volume dimension: " << obj.VolumeDim << std::endl;
    os << "bounding box start coords: " << obj.BBoxStart << std::endl;
    os << "bounding box dimension: " << obj.BBoxDim << std::endl;
    bool dataMatch = obj.density.size() == obj.VolumeDim.x * obj.VolumeDim.y * obj.VolumeDim.z;
    os << (dataMatch ? "The data volume is correct" : "The data volume isn't correct") << std::endl;
    return os;
}

fd::DENSITY_d::~DENSITY_d() {
    if (this->densityArray != nullptr) {
        checkCudaErrors(cudaDestroyTextureObject(this->densityTex));
        checkCudaErrors(cudaFreeArray(this->densityArray));
    }
}

void fd::density_h2d(DENSITY_h& _density_h_, DENSITY_d& _density_d_) {
    _density_d_.VoxelSize = _density_h_.VoxelSize;
    _density_d_.VolumeDim = _density_h_.VolumeDim;
    _density_d_.BBoxStart = _density_h_.BBoxStart;
    _density_d_.BBoxDim = _density_h_.BBoxDim;

    cudaExtent volumeSize = make_cudaExtent(_density_d_.VolumeDim.x,
        _density_d_.VolumeDim.y, _density_d_.VolumeDim.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMalloc3DArray(&(_density_d_.densityArray), &channelDesc, volumeSize));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = 
        make_cudaPitchedPtr(_density_h_.density.data(),
            volumeSize.width * sizeof(float),
            volumeSize.width, volumeSize.height);
    copyParams.dstArray = _density_d_.densityArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = _density_d_.densityArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;
    checkCudaErrors(cudaCreateTextureObject(&(_density_d_.densityTex), &texRes, &texDescr, NULL));
}

void fd::test_density() {
    DENSITY_h _density_h_;
    DENSITY_d _density_d_;
    _density_h_.VoxelSize = float3{0.25, 0.25, 0.25};
    _density_h_.VolumeDim = uint3{103, 103, 103};
    _density_h_.BBoxStart = uint3{1, 1, 1};
    _density_h_.BBoxDim = uint3{101, 101, 101};

    size_t vol = _density_h_.VolumeDim.x * _density_h_.VolumeDim.y * _density_h_.VolumeDim.z;
    _density_h_.density.resize(vol);
    // Initialize density array
    for (size_t i=0; i<vol; i++) {
        _density_h_.density[i] = rand01();
    }

    std::cout << "Host density volume: " << _density_h_;

    density_h2d(_density_h_, _density_d_);

    // read texture result
    float* result;
    checkCudaErrors(cudaMalloc(&result, vol*sizeof(float)));

    h_readTexture3D(result, _density_d_.densityTex, _density_d_.VolumeDim.x,
        _density_d_.VolumeDim.y, _density_d_.VolumeDim.z);

    std::vector<float> result_h(vol);
    checkCudaErrors(cudaMemcpy(result_h.data(), result, vol*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(result));

    double absoluteDiff = 0.;
    for (size_t i=0; i<vol; i++)
        absoluteDiff += abs(_density_h_.density[i] - result_h[i]);
    std::cout << "After copying the host density into the device density, "
        "and reading it back to host, the difference is :" << absoluteDiff << std::endl;
}