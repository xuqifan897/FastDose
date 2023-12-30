#!/bin/bash

for FmapOn in 2 4 6 8 16; do
    ./build/bin/example \
        --inputFolder /data/qifan/FastDoseWorkplace/tmp \
        --outputFolder /data/qifan/FastDoseWorkplace/CCCSslab \
        --deviceIdx 2 \
        --dicomVolumeDimension 103 103 103 \
        --voxelSize 0.25 0.25 0.25 \
        --doseBoundingBoxStartIndices 1 1 1 \
        --doseBoundingBoxDimensions 101 101 101 \
        --FmapOn "${FmapOn}" \
        --beamletSize 0.08
done

# ./build/bin/example \
#     --inputFolder /data/qifan/FastDoseWorkplace/tmp \
#     --outputFolder /data/qifan/FastDoseWorkplace/CCCSslab \
#     --deviceIdx 2 \
#     --dicomVolumeDimension 103 103 103 \
#     --voxelSize 0.25 0.25 0.25 \
#     --doseBoundingBoxStartIndices 1 1 1 \
#     --doseBoundingBoxDimensions 101 101 101 \
#     --FmapOn 8 \
#     --beamletSize 0.08