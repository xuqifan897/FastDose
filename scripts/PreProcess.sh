#!/bin/bash

inputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output"
if [ ! -d ${inputFolder} ]; then
    mkdir ${inputFolder}
fi

sharedData="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/sharedData"
./build/bin/preprocess \
    --dicomFolder "${sharedData}/data" \
    --structuresFile "${sharedData}/structures.json" \
    --ptv_name "PTV_ENLARGED" \
    --bbox_name "Skin" \
    --inputFolder ${inputFolder}