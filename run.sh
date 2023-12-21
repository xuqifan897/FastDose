#!/bin/bash

./build/bin/example \
    --inputFolder /data/qifan/projects/EndtoEnd/results/CCCSclone/tmp \
    --outputFolder /data/qifan/projects/EndtoEnd/results/CCCSBench \
    --deviceIdx 3 \
    --dicomVolumeDimension 103 103 103 \
    --voxelSize 0.25 0.25 0.25 \
    --doseBoundingBoxStartIndices 1 1 1 \
    --doseBoundingBoxDimensions 101 101 101 \


# gdb --args ./build/bin/example \
#     --inputFolder /data/qifan/projects/EndtoEnd/results/CCCSclone/tmp \
#     --outputFolder /data/qifan/projects/EndtoEnd/results/CCCSBench \
#     --deviceIdx 3 \
#     --dicomVolumeDimension 103 103 103 \
#     --voxelSize 0.25 0.25 0.25 \
#     --doseBoundingBoxStartIndices 1 1 1 \
#     --doseBoundingBoxDimensions 101 101 101 \