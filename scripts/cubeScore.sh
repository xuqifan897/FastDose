#!/bin/bash

ProjectFolder="/data/qifan/projects/FastDose"
exec="${ProjectFolder}/build/bin/cubeScore"

resultFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/water/width5mm"
if [ ! -d ${resultFolder} ]; then
    mkdir ${resultFolder}
fi
logFile="${resultFolder}/MC.log"

${exec} \
    --SpectrumFile "${ProjectFolder}/cubeScore/spectrum.csv" \
    --SlabPhantomFile "${ProjectFolder}/cubeScore/SlabPhantom.csv" \
    --MaterialFile "${ProjectFolder}/cubeScore/material.csv" \
    --OutputFile "${resultFolder}/MCDose.bin" \
    --nParticles 100000 \
    --logFreq 10000 \
    | tee ${logFile}