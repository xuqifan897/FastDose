#!/bin/bash

for FmapOn in 2 4 6 8 16; do
    ./build/bin/example \
        --inputFolder /data/qifan/projects/EndtoEnd/results/CCCSclone/tmp \
        --outputFolder /data/qifan/projects/EndtoEnd/results/CCCSslab \
        --deviceIdx 0 \
        --dicomVolumeDimension 103 103 103 \
        --voxelSize 0.25 0.25 0.25 \
        --doseBoundingBoxStartIndices 1 1 1 \
        --doseBoundingBoxDimensions 101 101 101 \
        --FmapOn ${FmapOn} \
        --beamletSize 0.08
done


# gdb --args ./build/bin/example \
#     --inputFolder /data/qifan/projects/EndtoEnd/results/CCCSclone/tmp \
    # --outputFolder /data/qifan/projects/EndtoEnd/results/CCCSslab \
    # --deviceIdx 0 \
    # --dicomVolumeDimension 103 103 103 \
    # --voxelSize 0.25 0.25 0.25 \
    # --doseBoundingBoxStartIndices 1 1 1 \
    # --doseBoundingBoxDimensions 101 101 101 \