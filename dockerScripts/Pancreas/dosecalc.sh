#!/bin/bash

# SPECFOLDER="/data/qifan/projects/FastDoseDocker/FastDose/scripts"
SOURCE_FOLDER="/data/qifan/projects/FastDoseWorkplace/dockerExample/PancreasPatient001"
INPUT_FOLDER="prep_output"
if [ ! -d "${SOURCE_FOLDER}/${INPUT_FOLDER}" ]; then
    echo "The folder ${INPUT_FOLDER} does not exist. Did you run the preprocess step?"
    exit 1
fi
dimFile="${SOURCE_FOLDER}/${INPUT_FOLDER}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

for ((segment=1; segment<=2; segment++)); do
    # Here we split all beams into 2 beam lists
    outputFolder="${SOURCE_FOLDER}/doseMat${segment}"
    if [ ! -d ${outputFolder} ]; then
        mkdir ${outputFolder}
    fi
    chmod -R 777 ${outputFolder}

    sudo docker run \
        --rm \
        --gpus all \
        --mount type=bind,source=${SOURCE_FOLDER},target="/data" \
        fastdose/fastdoseimage IMRT \
            --phantomDim ${phantomDim} \
            --voxelSize ${voxelSize} \
            --SAD 100 \
            --density "/data/${INPUT_FOLDER}/density.raw" \
            --structures ${VOIs} \
            --masks "/data/${INPUT_FOLDER}/roi_list.h5" \
            --primaryROI "ROI" \
            --bboxROI "SKIN" \
            --structureInfo - \
            --params - \
            --beamlist "/data/beamlist${segment}.txt" \
            --mode 0 \
            --deviceIdx 1 \
            --spectrum "/data/spec_6mv.spec" \
            --kernel "/data/kernel_exp_6mv.txt" \
            --fluenceDim 20 \
            --subFluenceDim 16 \
            --outputFolder "/data/doseMat${segment}" \
            --planFolder - \
            --nBeamsReserve 471 \
            --EstNonZeroElementsPerMat 8000000
done