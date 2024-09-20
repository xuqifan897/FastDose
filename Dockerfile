ARG CUDA_VERSION=12.6.1

# build stage
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS build

RUN apt-get update -y && \
    apt-get install -y \
        build-essential \
        cmake \
        wget \
        tar \
        git \
        libboost-all-dev \
        libhdf5-serial-dev \
        libeigen3-dev \
        rapidjson-dev \
        libopencv-dev \
    && apt install -y libdcmtk-dev

# install geant4
RUN git clone https://github.com/Geant4/geant4.git /packages/geant4_source \
    && cd /packages/geant4_source \
    && git checkout e58e650b32b961c8093f3dd6a2c3bc917b2552be \
    && mkdir /packages/geant4_build \
    && cd /packages/geant4_build \
    && cmake ../geant4_source \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/packages/geant4_install \
    -DGEANT4_BUILD_MULTITHREADED=ON \
    -DGEANT4_INSTALL_DATA=ON \
    -DGEANT4_USE_G3TOG4=OFF \
    -DGEANT4_USE_GDML=OFF \
    -DGEANT4_USE_INVENTOR=OFF \
    -DGEANT4_USE_INVENTOR_QT=OFF \
    -DGEANT4_USE_OPENGL_X11=OFF \
    -DGEANT4_USE_QT=OFF \
    -DGEANT4_USE_RAYTRACER_X11=OFF \
    -DGEANT4_USE_SYSTEM_CLHEP=OFF \
    -DGEANT4_USE_SYSTEM_EXPAT=OFF \
    -DGEANT4_USE_SYSTEM_ZLIB=OFF \
    -DGEANT4_USE_XM=OFF \
    && make -j64 \
    && make install

ENV CMAKE_MODULE_PATH="/packages/geant4_install" \
    Geant4_DIR="/packages/geant4_install/lib/cmake/Geant4" \
    LD_LIBRARY_PATH=/app:/packages/geant4_install/lib:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=/app:/packages/geant4_install/lib:${LIBRARY_PATH}

WORKDIR /FastDose
COPY . .
RUN mkdir build \
    && cd build \
    && cmake .. \
    && make -j64


# runtime stag
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04
RUN apt-get update -y && \
    apt-get install -y \
        build-essential \
        cmake \
        wget \
        tar \
        git \
        libboost-all-dev \
        libhdf5-serial-dev \
        libeigen3-dev \
        rapidjson-dev \
        libopencv-dev \
    && apt install -y libdcmtk-dev

COPY --from=build \
    /packages/geant4_install \
    /packages/geant4_instal

RUN useradd -m user
USER user

COPY --chown=user:user --from=build \
    /FastDose/build/bin \
    /app
COPY --chown=user:user --from=build \
    /FastDose/dockerScripts/dockerRun.sh \
    /app

ENTRYPOINT ["/bin/bash", "/app/dockerRun.sh"]