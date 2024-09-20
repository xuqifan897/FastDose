#!/bin/bash

SOURCE_FOLDER="/data/qifan/projects/FastDoseWorkplace/dockerExample/PancreasPatient001"
INPUT_FOLDER="prep_output"
SOURCE_INPUT_FOLDER="${SOURCE_FOLDER}/${INPUT_FOLDER}"
if [ ! -d ${SOURCE_INPUT_FOLDER} ]; then
    mkdir ${SOURCE_INPUT_FOLDER}
fi
chmod -R 777 ${SOURCE_INPUT_FOLDER}

sudo docker run \
    --rm \
    --gpus all \
    --mount type=bind,source=${SOURCE_FOLDER},target="/data" \
    fastdose/fastdoseimage preprocess \
        --mode 1 \
        --structuresFile "/data/structures.json" \
        --ptv_name "ROI" \
        --bbox_name "SKIN" \
        --voxelSize 0.25 \
        --inputFolder "/data/${INPUT_FOLDER}" \
        --shape 220 220 160 \
        --phantomPath "/data/density_raw.bin" \
        --RescaleSlope 1.0 \
        --RescaleIntercept -1000.0 \
        --maskFolder "/data/InputMask"


if false; then
    # Alternatively, the preprocessing code also supports taking a dicom 
    # folder containing the CT slices and the RTSTRUCT file as input. Providing
    # the voxel size, it resample the phantom to produce an isotropic-resolution
    # phantom.
    sudo docker run \
        --rm \
        --gpus all \
        --mount type=bind,source=${SOURCE_FOLDER},target="/data" \
            fastdose/fastdoseimage preprocess \
                --mode 0 \
                --dicomFolder ${DICOM_FOLDER} \
                --structuresFile ${STRUCTURES_FILE}.json \
                --ptv_name ${PTV_NAME} \
                --bbox_name ${BBOX_NAME} \
                --voxelSize 0.25 \
                --inputFolder ${INPUT_FOLDER}
fi