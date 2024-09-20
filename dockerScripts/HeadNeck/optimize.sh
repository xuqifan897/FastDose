#!/bin/bash

SOURCE_FOLDER="/data/qifan/projects/FastDoseWorkplace/dockerExample/HeadNeck002"
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

planFolder="plan"
sourcePlanFolder="${SOURCE_FOLDER}/${planFolder}"
if [ ! -d ${sourcePlanFolder} ]; then
    mkdir ${sourcePlanFolder}
fi
chmod -R 777 ${sourcePlanFolder}

OMP_NUM_THREADS=64 sudo docker run \
    --rm \
    --gpus all \
    --mount type=bind,source=${SOURCE_FOLDER},target="/data" \
        fastdose/fastdoseimage IMRT \
        --phantomDim ${phantomDim} \
        --voxelSize ${voxelSize} \
        --SAD 100 \
        --density - \
        --structures ${VOIs} \
        --masks "/data/${INPUT_FOLDER}/roi_list.h5" \
        --primaryROI "PTVMerge" \
        --bboxROI "SKIN" \
        --structureInfo "/data/StructureInfo.csv" \
        --params "/data/params.txt" \
        --beamlist - \
        --mode 1 \
        --deviceIdx 1 \
        --spectrum - \
        --kernel - \
        --fluenceDim 20 \
        --subFluenceDim 16 \
        --outputFolder /data/doseMat0 /data/doseMat1 /data/doseMat2 /data/doseMat3 \
        --planFolder /data/${planFolder} \
        --nBeamsReserve 0 \
        --EstNonZeroElementsPerMat 0