#!/bin/bash

outputFolder="/data/qifan/projects/EndtoEnd/results/CCCSBench"
logFile="${outputFolder}/MClog.txt"

if true; then
    nohup ./build/bin/KernelGen \
        --outputFolder /data/qifan/projects/EndtoEnd/results/CCCSBench \
        --spectrumFile /data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/spec_6mv.spec \
        --nParticles 1000000000 \
        2>&1 > ${logFile} &
fi

if false; then
    nohup ./build/bin/KernelGen \
        --outputFolder /data/qifan/projects/EndtoEnd/results/CCCSBench \
        --spectrumFile /data/qifan/projects/EndtoEnd/results/CCCSBench/spec_6mev.spec \
        2>&1 > ${logFile} &
fi