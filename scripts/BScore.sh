#!bin/bash

beamletSize=0.64

resultFolder="/data/qifan/projects/EndtoEnd/results/CCCSslab/boxScore${beamletSize}"
if [ ! -d ${resultFolder} ]; then
    mkdir ${resultFolder}
fi
logFile="${resultFolder}/log.txt"

for ((i=0; i<16; i++)); do
    ./build/bin/boxScore \
    --nParticles 10000000 \
    --dimXY 99 \
    --beamlet-size ${beamletSize} \
    --resultFolder ${resultFolder} \
    --iteration ${i}
done