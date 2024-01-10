#!/bin/bash

./build/bin/IMRT \
    --phantomDim 220 220 149 \
    --voxelSize 0.25 0.25 0.25 \
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
    --outputFolder "/data/qifan/FastDoseWorkplace/BOOval/LUNG/optimize" \
    --nBeamsReserve 452

# --masks "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/roi_list.h5"