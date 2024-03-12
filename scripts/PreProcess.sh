#!/bin/bash

inputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output"
if [ ! -d ${inputFolder} ]; then
    mkdir ${inputFolder}
fi

./build/bin/preprocess \
    --dicomFolder "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/data" \
    --structuresFile "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/structures.json" \
    --ptv_name "PTV_ENLARGED" \
    --bbox_name "Skin" \
    --inputFolder ${inputFolder}

# break /data/qifan/projects/FastDose/PreProcess/include/rtimages.h:56
# break /data/qifan/projects/FastDose/PreProcess/src/PreProcessInit.cpp:44