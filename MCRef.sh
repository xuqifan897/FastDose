#!/bin/bash

scoringSliceSize=1
# for ((scoringStartIdx = 0; scoringStartIdx < 103; scoringStartIdx += ${scoringSliceSize})); do
#     ./build/bin/MCReference \
#         --SpectrumFile "/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/spec_6mv.spec" \
#         --nParticles 10000000 \
#         --voxelSize 0.125 \
#         --phantomDim 103 103 103 \
#         --phantomPath "/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/density.raw" \
#         --SAD 100.0 \
#         --FmapOn 0 \
#         --scoringStartIdx ${scoringStartIdx} \
#         --scoringSliceSize ${scoringSliceSize} \
#         --superSampling 10 \
#         --outputFolder "/data/qifan/projects/EndtoEnd/results/CCCSslab" \
#         --logFrequency 1000000
# done

./build/bin/MCReference \
    --SpectrumFile "/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/spec_6mv.spec" \
    --nParticles 10000000 \
    --voxelSize 0.125 \
    --phantomDim 103 103 103 \
    --phantomPath "/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/density.raw" \
    --SAD 100.0 \
    --FmapOn 0 \
    --scoringStartIdx 0 \
    --scoringSliceSize 1 \
    --superSampling 10 \
    --outputFolder "/data/qifan/projects/EndtoEnd/results/CCCSslab" \
    --logFrequency 1000000