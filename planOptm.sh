#!/bin/bash

./build/bin/PlanOptm \
    --phantom "/data/qifan/FastDoseWorkplace/PhtmOptm/slabPhantom/density.raw" \
    --phantomDim 103 103 103 \
    --voxelSize 0.25 0.25 0.25 \
    --isocenter 12.875 12.875 12.875 \
    --SAD 100.0 \
    --boundingBoxStart 1 1 1 \
    --boundingBoxDimensions 100 100 100 \
    --beamlist "/data/qifan/FastDoseWorkplace/PhtmOptm/slabPhantom/beamlist_full.txt" \
    --deviceIdx 3 \
    --spectrum "/data/qifan/FastDoseWorkplace/PhtmOptm/slabPhantom/spec_6mv.spec" \
    --kernel "/data/qifan/FastDoseWorkplace/PhtmOptm/slabPhantom/kernel_exp_6mv.txt" \
    --nPhi 8 \
    --longSpacing 0.25 \
    --outputFolder "/data/qifan/FastDoseWorkplace/PhtmOptm"