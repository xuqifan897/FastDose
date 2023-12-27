import os
import numpy as np
import matplotlib.pyplot as plt

def viewDose():
    FmapOn = 0.32
    # FmapOn = 0.64
    resultFolder = "/data/qifan/projects/EndtoEnd/results/CCCSslab/boxScore{}".format(FmapOn)
    nSlices = 16
    sliceShape = (16, 99, 99)
    res = 0.1
    globalShape = (sliceShape[0]*nSlices, sliceShape[1], sliceShape[2])
    globalDose = np.zeros(globalShape, dtype=np.double)

    for i in range(nSlices):
        file = os.path.join(resultFolder, 'SD{:03d}.bin'.format(i+1))
        localDose = np.fromfile(file, dtype=np.double)
        localDose = np.reshape(localDose, sliceShape)
        globalDose[i*sliceShape[0]: (i+1)*sliceShape[0], :, :] = localDose
    

    matMap = ['adipose', 'muscle', 'bone', 'muscle'] + \
        ['lung'] * 6 + \
        ['muscle', 'bone', 'adipose', 'bone', 'muscle', 'adipose']
    densityMap = {
        'adipose': 0.95610404,
        'muscle': 1.0652606,
        'bone': 1.6093124,
        'lung': 0.2481199
        }
    sliceSize = 16
    densityPlot = np.zeros(256)
    offset = 0
    for mat in matMap:
        densityValue = densityMap[mat]
        for i in range(sliceSize):
            densityPlot[offset] = densityValue
            offset += 1
    densityPlot = np.expand_dims(densityPlot, axis=(1, 2))
    globalDose /= densityPlot
    
    centerIdx = int((sliceShape[1] - 1) / 2)
    centerline = globalDose[:, centerIdx, centerIdx].copy()
    centerline /= np.max(centerline)
    depth = np.arange(globalShape[0]) * res
    plt.plot(depth, centerline)

    # CCCS value
    if FmapOn == 0.32:
        CCCSfile = "/data/qifan/projects/EndtoEnd/results/CCCSslab/DosePVCSFmap8.bin"
    elif FmapOn == 0.64:
        CCCSfile = "/data/qifan/projects/EndtoEnd/results/CCCSslab/DosePVCSFmap16.bin"
    CCCSshape = (103, 103, 103)
    CCCSRes = 0.25
    CCCSDose = np.fromfile(CCCSfile, dtype=np.float32)
    CCCSDose = np.reshape(CCCSDose, CCCSshape)
    CCCSCenterline = CCCSDose[51, :, 51].copy()
    CCCSCenterline /= np.max(CCCSCenterline)
    CCCSDepth = np.arange(CCCSshape[1]) * CCCSRes
    plt.plot(CCCSDepth, CCCSCenterline)
    plt.xlabel('depth (cm)')
    plt.ylabel('dose (a.u.)')
    plt.title('Depth dose curve with beamlet size {}cm'.format(FmapOn*2))
    plt.legend(['Monte Carlo', "CCCS"])
    figureFile = './figures/MCFmap{}Centerline.png'.format(FmapOn)
    plt.savefig(figureFile)
    plt.clf()


    # view the lateral dose profile at depth = 10cm
    depth = 10.
    SliceIdxMC = int(depth / res)
    lateralMC = globalDose[SliceIdxMC, centerIdx, :].copy()
    lateralMC /= np.max(lateralMC)
    lateralAxisMC = (np.arange(sliceShape[2]) - (sliceShape[2] - 1) / 2) * res
    plt.plot(lateralAxisMC, lateralMC)

    SliceIdxCCCS = int(depth / CCCSRes)
    lateralCCCS = CCCSDose[51, SliceIdxCCCS, :].copy()
    lateralCCCS /= np.max(lateralCCCS)
    lateralAxisCCCS = (np.arange(CCCSshape[2]) - (CCCSshape[2] - 1) / 2) * CCCSRes
    plt.plot(lateralAxisCCCS, lateralCCCS)
    plt.xlabel('off-axis distance (cm)')
    plt.ylabel('dose (a.u.)')
    plt.title("Beamlet size {}cm\nLateral dose profile at depth {}cm".format(FmapOn*2, depth))
    plt.legend(["Monte Carlo", "CCCS"])
    figureFile = './figures/MCFmap{}Lateral.png'.format(FmapOn)
    plt.savefig(figureFile)
    plt.clf()


def viewDensity():
    phantomPath = "/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/density.raw"
    phantomShape = (103, 103, 103)
    res = 0.25
    phantom = np.fromfile(phantomPath, dtype=np.float32)

    uniqueValues = []
    for i in range(phantom.size):
        localValue = phantom[i]
        found = False
        for a in uniqueValues:
            if (abs(a - localValue)) < 1e-4:
                found = True
                break
        if not found:
            uniqueValues.append(localValue)
    print(uniqueValues)


if __name__ == '__main__':
    viewDose()
    # viewDensity()