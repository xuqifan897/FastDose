#!/bin/bash
# sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'

resultFolder="/data/qifan/projects/EndtoEnd/results/CCCSBench"
if [ ! -d ${resultFolder} ]; then
    mkdir ${resultFolder}
fi

computeResult="${resultFolder}/TermaBench.ncu-rep"
/usr/local/cuda/bin/ncu -f \
    --target-processes all \
    --section SchedulerStats \
    --section WarpStateStats \
    --section SourceCounters \
    --section Occupancy \
    -o ${computeResult} ./build/bin/example \
    --inputFolder /data/qifan/projects/EndtoEnd/results/CCCSclone/tmp \
    --outputFolder /data/qifan/projects/EndtoEnd/results/CCCSBench \
    --deviceIdx 0 \
    --dicomVolumeDimension 103 103 103 \
    --voxelSize 0.25 0.25 0.25 \
    --doseBoundingBoxStartIndices 1 1 1 \
    --doseBoundingBoxDimensions 101 101 101 \