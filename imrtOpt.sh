#!/bin/bash

if false; then
    outputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/optimize"
    if [ ! -d ${outputFolder} ]; then
        mkdir ${outputFolder}
    fi;
    OMP_NUM_THREADS=128 ./build/bin/IMRT \
        --phantomDim 220 220 149 \
        --voxelSize 0.25 0.25 0.25 \
        --SAD 100 \
        --density "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/density.raw" \
        --masks "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/structs.h5" \
        --primaryROI "PTV" \
        --bboxROI "BODY" \
        --structureInfo "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/StructureInfo.csv" \
        --params "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/params.txt" \
        --beamlist "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/beamlist.txt" \
        --mode 2 \
        --deviceIdx 3 \
        --spectrum "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/spec_6mv.spec" \
        --kernel "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/kernel_exp_6mv.txt" \
        --subFluenceDim 16 \
        --concurrency 1 \
        --outputFolder ${outputFolder} \
        --nBeamsReserve 452
fi



if true; then
    outputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result"
    if [ ! -d ${outputFolder} ]; then
        mkdir ${outputFolder}
    fi;
    OMP_NUM_THREADS=128 ./build/bin/IMRT \
        --phantomDim 220 220 149 \
        --voxelSize 0.25 0.25 0.25 \
        --SAD 100 \
        --density "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/density.raw" \
        --masks "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/roi_list.h5" \
        --primaryROI "PTV_ENLARGED" \
        --bboxROI "Skin" \
        --structureInfo "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/StructureInfo.csv" \
        --params "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/params.txt" \
        --beamlist "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/beamlist.txt" \
        --mode 2 \
        --deviceIdx 3 \
        --spectrum "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/spec_6mv.spec" \
        --kernel "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/kernel_exp_6mv.txt" \
        --subFluenceDim 16 \
        --concurrency 1 \
        --outputFolder ${outputFolder} \
        --nBeamsReserve 452
fi