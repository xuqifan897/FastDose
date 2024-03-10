#include "PreProcessRingStruct.cuh"
#include "helper_cuda.h"
#include "helper_math.h"
#include <iostream>

PreProcess::cudaVolume::cudaVolume(int3 _size_): size(_size_) {
    size_t num_elements = this->size.x * this->size.y * this->size.z;
    checkCudaErrors(cudaMalloc((void**)&this->_vect, num_elements*sizeof(uint8_t)));
}

PreProcess::cudaVolume::~cudaVolume() {
    if (this->_vect)
        checkCudaErrors(cudaFree(this->_vect));
}

bool PreProcess::imdilate(cudaVolume& result, const cudaVolume& op1, const cudaVolume& op2) {
    if (op2.size.x % 2 == 0 || op2.size.y % 2 == 0 || op2.size.z % 2 == 0) {
        std::cerr << "The dimension of op2 is expected to be an odd number" << std::endl;
        return 1;
    }
    result.size = op1.size;
    size_t num_elements = result.size.x * result.size.y * result.size.z;
    checkCudaErrors(cudaMalloc((void**)&result._vect, num_elements*sizeof(uint8_t)));
    dim3 blockSize(8, 8, 8);
    dim3 gridSize (
        (result.size.x + blockSize.x - 1) / blockSize.x,
        (result.size.y + blockSize.y - 1) / blockSize.y,
        (result.size.z + blockSize.z - 1) / blockSize.z);
    d_imdilate<<<gridSize, blockSize>>>(
        result.size,
        op2.size,
        result._vect,
        op1._vect,
        op2._vect );
    return 0;
}


__global__ void PreProcess::d_imdilate(
    const int3 targetDim,
    const int3 op2Dim,
    uint8_t* target_vec,
    const uint8_t* op1_vec,
    const uint8_t* op2_vec
) {
    uint3 _Idx_ = threadIdx + blockDim * blockIdx;
    int3 Idx{(int)_Idx_.x, (int)_Idx_.y, (int)_Idx_.z};
    if (Idx.x >= targetDim.x || Idx.y >= targetDim.y || Idx.z >= targetDim.z)
        return;
    int3 offset{(op2Dim.x - 1) / 2, (op2Dim.y - 1) / 2, (op2Dim.z - 1) / 2};
    
    bool find = false;
    for (int k=0; k<op2Dim.z; k++) {
        for (int j=0; j<op2Dim.y; j++) {
            for (int i=0; i<op2Dim.x; i++) {
                size_t op2_vec_idx = i + op2Dim.x * (j + op2Dim.y * k);
                uint8_t op2_value = op2_vec[op2_vec_idx];
                if (! op2_value)
                    continue;
                
                int3 op1_coords = Idx - offset + int3{i, j, k};
                if (op1_coords.x < 0 || op1_coords.x >= targetDim.x ||
                    op1_coords.y < 0 || op1_coords.y >= targetDim.y ||
                    op1_coords.z < 0 || op1_coords.z >= targetDim.z)
                    continue;

                size_t op1_vec_idx = op1_coords.x + targetDim.x *
                    (op1_coords.y + targetDim.y * op1_coords.z);
                find = op1_vec[op1_vec_idx];
                if (find)
                    break;
            }
            if (find)
                break;
        }
        if (find)
            break;
    }

    size_t target_idx = Idx.x + targetDim.x * (Idx.y + targetDim.y * Idx.z);
    target_vec[target_idx] = find;
}


bool PreProcess::test_imdilate() {
    int3 op1_size{128, 128, 128};
    size_t op1_elements = op1_size.x * op1_size.y * op1_size.z;
    std::vector<uint8_t> op1_array(op1_elements, 0);

    float3 center{(float)op1_size.x, (float)op1_size.y, (float)op1_size.z};
    center -= 1.0f;
    center *= 0.5f;

    float radius = 64;
    for (int k=0; k<op1_size.z; k++) {
        for (int j=0; j<op1_size.y; j++) {
            for (int i=0; i<op1_size.x; i++) {
                float3 coords{(float)i, (float)j, (float)k};
                coords -= center;
                float dist_square = length(coords);

                size_t op1_idx = i + op1_size.x * (j + op1_size.y * k);
                op1_array[op1_idx] = (dist_square < radius);
            }
        }
    }
    #if false
        for (int k=radius; k<radius+1; k++) {
            for (int j=0; j<op1_size.y; j++) {
                for (int i=0; i<op1_size.x; i++) {
                    size_t op1_idx = i + op1_size.x * (j + op1_size.y * k);
                    std::cout << (int)op1_array[op1_idx] << " ";
                }
                std::cout << "\n";
            }
        }
    #endif
    cudaVolume op1(op1_size);
    checkCudaErrors(cudaMemcpy(op1._vect, op1_array.data(),
        op1_array.size()*sizeof(uint8_t), cudaMemcpyHostToDevice));

    
    int3 op2_size{11, 11, 1};
    size_t op2_elements = op2_size.x * op2_size.y * op2_size.z;
    std::vector<uint8_t> op2_array(op2_elements, 0);
    for (int i=0; i<op2_size.x; i++) {
        size_t idx = i + op2_size.x * i;
        op2_array[idx] = 1;
    }
    cudaVolume op2(op2_size);
    checkCudaErrors(cudaMemcpy(op2._vect, op2_array.data(),
        op2_array.size()*sizeof(uint8_t), cudaMemcpyHostToDevice));

    cudaVolume op3;
    if(imdilate(op3, op1, op2)) {
        std::cerr << "imdilate error." << std::endl;
        return 1;
    }

    int3 op3_size = op3.size;
    size_t op3_elements = op3_size.x * op3_size.y * op3.size.z;
    std::vector<uint8_t> op3_array(op3_elements, 0);
    checkCudaErrors(cudaMemcpy(op3_array.data(), op3._vect, op3_elements*sizeof(uint8_t), cudaMemcpyDeviceToHost));

    for (int k=32; k<33; k++) {
        for (int j=0; j<op1_size.y; j++) {
            for (int i=0; i<op1_size.x; i++) {
                size_t op1_idx = i + op1_size.x * (j + op1_size.y * k);
                std::cout << (int)op3_array[op1_idx] << " ";
            }
            std::cout << "\n";
        }
    }

    return 0;
}


bool PreProcess::CreateRingStructure(
    ROIMaskList& roi_list, RTStruct& rtstruct,
    const FloatVolume& ctdata, const FloatVolume& density, bool verbose
) {
    const std::string ptv_name = getarg<std::string>("ptv_name");
    const std::string bbox_name = getarg<std::string>("bbox_name");

    int ptv_idx = getROIIndex(rtstruct, ptv_name, true, verbose);
    int bbox_idx = getROIIndex(rtstruct, bbox_name, true, verbose);

    StructureSet ptv, bbox;
    if (loadStructureSet(ptv, rtstruct, ptv_idx, verbose)
        || loadStructureSet(bbox, rtstruct, bbox_idx, verbose)) {
        std::cerr << "Failed to load ptv or bbox" << std::endl;
        return 1;
    }

    Volume<uint8_t> ptv_mask{}, bbox_mask{};
    ptv_mask = generateContourMask(ptv, ctdata.get_frame(), density.get_frame());
    bbox_mask = generateContourMask(bbox, ctdata.get_frame(), density.get_frame());

    // calculate the number of voxels in ptv_mask
    int ptv_num_voxels = 0;
    for (int i=0; i<ptv_mask._vect.size(); i++) {
        ptv_num_voxels += (ptv_mask._vect[i] > 0);
    }
    float radius = 2 * powf(
        3.0f * ptv_num_voxels / (4.0f * M_PI),
        0.33333f);
    radius = min(radius, 20.0f);
    int radius_int = (int)round(radius);
    int3 sphereSize{2*radius_int+1, 2*radius_int+1, 2*radius_int+1};
    float3 sphereCenter{(float)radius_int, (float)radius_int, (float)radius_int};

    size_t num_elements = sphereSize.x * sphereSize.y * sphereSize.z;
    std::vector<uint8_t> sphereArray(num_elements, 0);
    for (int k=0; k<sphereSize.z; k++) {
        for (int j=0; j<sphereSize.y; j++) {
            for (int i=0; i<sphereSize.x; i++) {
                float3 coords{(float)i, (float)j, (float)k};
                coords -= sphereCenter;
                int sphereIdx = i + sphereSize.x * (j + sphereSize.y * k);
                sphereArray[sphereIdx] = length(coords) < radius_int;
            }
        }
    }

    int3 gapSize{3, 3, 3};
    std::vector<uint8_t> gapArray(gapSize.x * gapSize.y * gapSize.z, 1);

    if (ptv_mask.size.x != bbox_mask.size.x
        || ptv_mask.size.y != bbox_mask.size.y
        || ptv_mask.size.z != bbox_mask.size.z) {
        std::cerr << "The sizes of ptv and bbox are inconsistent" << std::endl;
        return 1;
    }



    int3 ptv_size{(int)ptv_mask.size.x, (int)ptv_mask.size.y, (int)ptv_mask.size.z};
    cudaVolume PTVcu(ptv_size);
    int num_elements_ptv = ptv_size.x * ptv_size.y * ptv_size.z;
    checkCudaErrors(cudaMemcpy(PTVcu._vect, ptv_mask._vect.data(),
        num_elements_ptv*sizeof(uint8_t), cudaMemcpyHostToDevice));
    
    cudaVolume SPHERE_cu(sphereSize);
    cudaVolume GAP_cu(gapSize);
    checkCudaErrors(cudaMemcpy(SPHERE_cu._vect, sphereArray.data(),
        sphereArray.size()*sizeof(uint8_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(GAP_cu._vect, gapArray.data(),
        gapArray.size()*sizeof(uint8_t), cudaMemcpyHostToDevice));
    
    cudaVolume PTV_dilate_SPHERE, PTV_dilate_GAP;
    if (imdilate(PTV_dilate_SPHERE, PTVcu, SPHERE_cu) ||
        imdilate(PTV_dilate_GAP, PTVcu, GAP_cu)) {
        return 1;
    }


    std::vector<uint8_t> PTV_dilate_SPHERE_cpu(num_elements_ptv, 0);
    std::vector<uint8_t> PTV_dilate_GAP_cpu(num_elements_ptv, 0);
    checkCudaErrors(cudaMemcpy(PTV_dilate_SPHERE_cpu.data(), PTV_dilate_SPHERE._vect,
        num_elements_ptv*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(PTV_dilate_GAP_cpu.data(), PTV_dilate_GAP._vect,
        num_elements_ptv*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    std::vector<uint8_t> RingStruct_array(num_elements_ptv, 0);
    for (int i=0; i<num_elements_ptv; i++) {
        RingStruct_array[i] = bbox_mask._vect[i] && PTV_dilate_SPHERE_cpu[i]
        && (! PTV_dilate_GAP_cpu[i]);
    }

    std::string RingStruct_name("RingStructure");
    ArrayProps RingStruct_bbox;
    RingStruct_bbox.size = bbox_mask.size;
    RingStruct_bbox.crop_size = bbox_mask.size;
    RingStruct_bbox.crop_start = uint3{0, 0, 0};
    roi_list.push_back(new DenseROIMask(RingStruct_name, RingStruct_array, RingStruct_bbox));

    std::cout << "Ring structure added." << std::endl;
    return 0;
}
