#!/bin/bash

if false; then
    ./build/bin/IMRT \
        --phantomDim 220 220 149 \
        --voxelSize 0.25 0.25 0.25 \
        --SAD 100 \
        --density "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/density.raw" \
        --masks "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/structs.h5" \
        --primaryROI "PTV" \
        --bboxROI "BODY" \
        --structureInfo "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/StructureInfo.csv" \
        --params "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/params.txt" \
        --beamlist "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/beamlist.txt" \
        --deviceIdx 2 \
        --spectrum "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/spec_6mv.spec" \
        --kernel "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/kernel_exp_6mv.txt" \
        --subFluenceDim 16 \
        --concurrency 4 \
        --outputFolder "/data/qifan/FastDoseWorkplace/BOOval/LUNG/optimize" \
        --nBeamsReserve 452
fi


if true; then
    resultFolder="/data/qifan/FastDoseWorkplace/BOOval/LUNG/experiment"
    if [ ! -d ${resultFolder} ]; then
        mkdir ${resultFolder}
    fi

    computeResult="${resultFolder}/interpBench.ncu-rep"
    /usr/local/cuda/bin/ncu -f \
        --target-processes all \
        --section SchedulerStats \
        --section WarpStateStats \
        --section SourceCounters \
        --section Occupancy \
        -o ${computeResult} ./build/bin/IMRT \
        --phantomDim 220 220 149 \
        --voxelSize 0.25 0.25 0.25 \
        --SAD 100 \
        --density "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/density.raw" \
        --masks "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/structs.h5" \
        --primaryROI "PTV" \
        --bboxROI "BODY" \
        --structureInfo "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/StructureInfo.csv" \
        --params "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/params.txt" \
        --beamlist "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/beamlist.txt" \
        --deviceIdx 2 \
        --spectrum "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/spec_6mv.spec" \
        --kernel "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/kernel_exp_6mv.txt" \
        --subFluenceDim 16 \
        --concurrency 4 \
        --outputFolder "/data/qifan/FastDoseWorkplace/BOOval/LUNG/optimize" \
        --nBeamsReserve 452
fi

if false; then
    resultFolder="/data/qifan/FastDoseWorkplace/BOOval/LUNG/experiment"
        if [ ! -d ${resultFolder} ]; then
            mkdir ${resultFolder}
        fi
    profResult="${resultFolder}/interpProf"
    nsys profile --stats=true -o ${profResult} ./build/bin/IMRT \
        --phantomDim 220 220 149 \
        --voxelSize 0.25 0.25 0.25 \
        --SAD 100 \
        --density "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/density.raw" \
        --masks "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/structs.h5" \
        --primaryROI "PTV" \
        --bboxROI "BODY" \
        --structureInfo "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/StructureInfo.csv" \
        --params "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/params.txt" \
        --beamlist "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/beamlist.txt" \
        --deviceIdx 2 \
        --spectrum "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/spec_6mv.spec" \
        --kernel "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/kernel_exp_6mv.txt" \
        --subFluenceDim 16 \
        --concurrency 4 \
        --outputFolder "/data/qifan/FastDoseWorkplace/BOOval/LUNG/optimize" \
        --nBeamsReserve 452
fi