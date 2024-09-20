#!/bin/bash

export CMAKE_MODULE_PATH="/packages/geant4_install:${CMAKE_MODULE_PATH}"
export Geant4_DIR="/packages/geant4_install/lib/cmake/Geant4"
export LD_LIBRARY_PATH="/app:/packages/geant4_install/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="/app:/packages/geant4_install/lib:${LIBRARY_PATH}"
# Check if the first argument is provided
if [ -z "$1" ]; then
    echo "No executable specified. Available options: IMRT, KernelGen, MaskGen, OptBench, PlanOptm, boxScore, cubeScore, example, preprocess, singleBeamBEV"
    exit 1
fi

# Use the first argument as the executable name
EXEC="$1"

# Shift removes the first argument, so the
# remaining arguments will be passed to the executable
shift

case ${EXEC} in
    IMRT)
        /app/IMRT "$@"
        ;;
    KernelGen)
        /app/KernelGen "$@"
        ;;
    MaskGen)
        /app/MaskGen "$@"
        ;;
    OptBench)
        /app/OptBench "$@"
        ;;
    PlanOptm)
        /app/PlanOptm "$@"
        ;;
    boxScore)
        /app/boxScore "$@"
        ;;
    cubeScore)
        /app/cubeScore "$@"
        ;;
    example)
        /app/example "$@"
        ;;
    preprocess)
        /app/preprocess "$@"
        ;;
    singBeamBEV)
        /app/singleBeamBEV "$@"
        ;;
    *)
        echo "Invalid executable: ${EXEC}"
        exit 1
        ;;
esac