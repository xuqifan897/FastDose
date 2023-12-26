import os
import numpy as np
import matplotlib.pyplot as plt

def viewDose():
    filePath = "/data/qifan/projects/EndtoEnd/results/CCCSslab/MCRefFmap0.000000/slice_0_10.bin"
    phantomPath = "/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/density.raw"
    shape = (103, 103, 103)
    voxelRes = 0.25

    doseArray = np.fromfile(filePath, dtype=np.double)
    doseArray = np.reshape(doseArray, shape)
    densityArray = np.fromfile(phantomPath, dtype=np.float32)
    densityArray = np.reshape(densityArray, shape)
    # doseArray = doseArray / densityArray

    # view centerline
    doseCenterline = doseArray[51, :, 51]
    depth = np.arange(shape[1]) * voxelRes
    plt.plot(depth, doseCenterline)
    plt.xlabel('depth (cm)')
    plt.ylabel('dose (a.u.)')
    plt.title('Monte Carlo centerline energy deposition')
    figureFile = './figures/MCCenterline.png'
    plt.savefig(figureFile)


def viewDose1():
    resultFolder = "/data/qifan/projects/EndtoEnd/results/CCCSslab/MCRefFmap0.000000"
    shape = (103, 103, 103)
    sliceSize = 10
    resolution = 0.25
    totalEdep = np.zeros(shape, dtype=np.double)
    for i in range(11):
        idx_start = i * sliceSize
        idx_end = (i + 1) * sliceSize
        idx_end = min(idx_end, shape[1])
        file = os.path.join(resultFolder, 'slice_{}_{}.bin'.format(idx_start, idx_end))
        localEdep = np.fromfile(file, dtype=np.double)
        localEdep = np.reshape(localEdep, shape)
        totalEdep += localEdep
    centerline = totalEdep[51, :, 51]
    depth = np.arange(shape[1]) * resolution
    plt.plot(depth, centerline)
    figureFile = './figures/MCCenterline.png'
    plt.savefig(figureFile)


if __name__ == '__main__':
    viewDose1()