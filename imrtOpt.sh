#!/bin/bash

if false; then
    outputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/optimize"
    if [ ! -d ${outputFolder} ]; then
        mkdir ${outputFolder}
    fi;
    dimFile="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/dimension.txt"
    readarray -t lines < ${dimFile}
    phantomDim=${lines[0]}
    voxelSize=${lines[1]}
    VOIs=${lines[2]}

    OMP_NUM_THREADS=128 ./build/bin/IMRT \
        --phantomDim ${phantomDim} \
        --voxelSize ${voxelSize} \
        --SAD 100 \
        --density "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/density.raw" \
        --structures ${VOIs} \
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
    inputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output"
    outputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result"
    if [ ! -d ${outputFolder} ]; then
        mkdir ${outputFolder}
    fi;
    dimFile="${inputFolder}/dimension.txt"
    readarray -t lines < ${dimFile}
    phantomDim=${lines[0]}
    voxelSize=${lines[1]}
    VOIs=${lines[2]}

    OMP_NUM_THREADS=64 ./build/bin/IMRT \
        --phantomDim ${phantomDim} \
        --voxelSize ${voxelSize} \
        --SAD 100 \
        --density "${inputFolder}/density.raw" \
        --structures ${VOIs} \
        --masks "${inputFolder}/roi_list.h5" \
        --primaryROI "PTV_ENLARGED" \
        --bboxROI "Skin" \
        --structureInfo "${inputFolder}/StructureInfo.csv" \
        --params "tuesday" \
        --beamlist "${inputFolder}/beamlist.txt" \
        --mode 0 \
        --deviceIdx 0 \
        --spectrum "${inputFolder}/spec_6mv.spec" \
        --kernel "${inputFolder}/kernel_exp_6mv.txt" \
        --subFluenceDim 16 \
        --concurrency 1 \
        --outputFolder ${outputFolder} \
        --nBeamsReserve 452
fi