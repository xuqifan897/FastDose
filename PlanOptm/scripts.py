import os
import numpy as np
import matplotlib.pyplot as plt

def FPangles():
    """
    This function reproduces results of FPangels.m
    """
    angles = 6
    N = round(180/angles)
    thetaY = 0
    result = []
    while (thetaY <= 180):
        # print(thetaY)
        nDeltaX = 2 * np.pi * np.cos((thetaY - 90) / 180 * np.pi) \
            / (angles / 180 * np.pi)
        nDeltaX = abs(nDeltaX)
        deltaX = round(360 / nDeltaX)
        thetaX = 0
        while (thetaX <= 359):
            result.append((thetaX, thetaY))
            thetaX += deltaX
        thetaY += angles
    result = np.array(result)
    return result


def FPangles2Coords(fp_angles):
    """
    This function converts the FPangles into azimuth and zenith angles
    """
    angle0 = fp_angles[:, 0]
    angle1 = fp_angles[:, 1]
    angle0 = np.expand_dims(angle0, axis=1)
    angle1 = np.expand_dims(angle1, axis=1)

    x_coords = - np.cos(angle1)
    y_coords = - np.sin(angle1)
    z_coords = - y_coords * np.sin(angle0)
    y_coords = y_coords * np.cos(angle0)
    result = np.concatenate((x_coords, y_coords, z_coords), axis=1)
    return result


def ZenithAzimuth2Coords():
    beamlist_file = "/data/qifan/projects/EndtoEnd/results/PlanOptm/slabPhantom/beamlist_full.txt"
    with open(beamlist_file, 'r') as f:
        lines = f.readlines()
    angles = np.zeros((len(lines), 2))
    for i in range(len(lines)):
        line = lines[i]
        line = line.split(' ')
        angles[i, 0] = float(line[0])
        angles[i, 1] = float(line[1])
    
    angles = angles * np.pi / 180
    angle0 = angles[:, 0]
    angle1 = angles[:, 1]
    angle0 = np.expand_dims(angle0, axis=1)
    angle1 = np.expand_dims(angle1, axis=1)

    y_coords = np.sin(angle0)
    z_coords = np.cos(angle0)
    x_coords = - y_coords * np.sin(angle1)
    y_coords = y_coords * np.cos(angle1)
    result = np.concatenate((x_coords, y_coords, z_coords), axis=1)
    return result


def showFPangles():
    fp_angles = FPangles()
    fp_angles = fp_angles * np.pi / 180

    coords_3d = FPangles2Coords(fp_angles)
    coords_3d_comp = ZenithAzimuth2Coords()
    dot_product = np.sum(coords_3d * coords_3d_comp, axis=1)
    print(dot_product)


def checkBeamletDose():
    dataFolder = "/data/qifan/FastDoseWorkplace/PhtmOptm/canonical"
    nBeamlets = 256
    fluenceShape = (16, 16)
    long_spacing = 0.25 

    for i in range(nBeamlets):
        file = os.path.join(dataFolder, 'DoseBEV_beamlet{}.bin'.format(i+1))
        data = np.fromfile(file, dtype=np.float32)
        dim_long = int(data.size / (fluenceShape[0] * fluenceShape[1]))
        doseShape = (dim_long,) + fluenceShape
        data = np.reshape(data, doseShape)

        if False:        
            dataSlice = data[:, 8, :]
            figureFile = './figures/beamletDoseBEV0.png'
            plt.imsave(figureFile, dataSlice)

        centerline = data[:, 8, 8]
        depth = np.arange(dim_long) * long_spacing
        plt.plot(depth, centerline)
        plt.xlabel('depth (cm)')
        plt.ylabel('dose (a.u.)')
        figureFile = './figures/beamletDoseCenterline{}.png'.format(i+1)
        plt.savefig(figureFile)
        plt.clf()


def checkBeamletPVCS():
    dataFolder = "/data/qifan/FastDoseWorkplace/PhtmOptm/canonical"
    nBeamlets = 256
    PVCSshape = (103, 103, 103)
    sliceIdx = 51
    long_spacing = 0.25
    figureFolder = './figures/canonical'
    if not os.path.isdir(figureFolder):
        os.mkdir(figureFolder)
    
    if False:
        sliceTotal = np.zeros((PVCSshape[0], PVCSshape[2]), dtype=np.float32)
        for i in range(nBeamlets):
            file = os.path.join(dataFolder, 'DosePVCS_beamlet{}.bin'.format(i+1))
            array = np.fromfile(file, dtype=np.float32)
            array = np.reshape(array, PVCSshape)
            slice = array[:, sliceIdx, :]
            sliceFile = os.path.join(figureFolder, 'slice{}.png'.format(i+1))
            plt.imsave(sliceFile, slice)
            sliceTotal += slice
        
        sliceFile = os.path.join(figureFolder, 'DosePVCS_total.png')
        plt.imsave(sliceFile, sliceTotal)
    
    if True:
        doseTotal = np.zeros(PVCSshape, dtype=np.float32)
        for i in range(nBeamlets):
            file = os.path.join(dataFolder, 'DosePVCS_beamlet{}.bin'.format(i+1))
            array = np.fromfile(file, dtype=np.float32)
            array = np.reshape(array, PVCSshape)
            doseTotal += array
        centerline = doseTotal[sliceIdx, :, sliceIdx]
        depth = np.arange(PVCSshape[1]) * long_spacing
        plt.plot(depth, centerline)
        plt.xlabel('depth (cm)')
        plt.ylabel('dose (a.u.)')
        plt.title('Beam size: 6.4 x 6.4 cm^2 centerline')
        figureFile = os.path.join(figureFolder, 'beamCenterlien.png')
        plt.savefig(figureFile)


def examine_individual_partial():
    dataFolder = "/data/qifan/FastDoseWorkplace/PhtmOptm/canonical"
    nBeamlets = 256
    PVCSshape = (103, 103, 103)
    long_spacing = 0.25
    figureFolder = './figures/canonical_partial'

    if not os.path.isdir(figureFolder):
        os.mkdir(figureFolder)
    depth = np.arange(PVCSshape[1]) * long_spacing

    for i in range(nBeamlets):
        file = os.path.join(dataFolder, 'DosePVCS_beamlet{}.bin'.format(i+1))
        array = np.fromfile(file, dtype=np.float32)
        array = np.reshape(array, PVCSshape)
        partial = np.sum(array, axis=(0, 2))
        plt.plot(depth, partial)
        plt.xlabel('depth (cm)')
        plt.ylabel('dose (a.u.)')
        plt.title('Sum dose beamlet {}'.format(i+1))
        figureFile = os.path.join(figureFolder, 'PartialPVCS{}.png'.format(i+1))
        plt.savefig(figureFile)
        plt.clf()


def examine_individual_BEV():
    dataFolder = "/data/qifan/FastDoseWorkplace/PhtmOptm/canonical"
    nBeamlets = 256
    fluenceShape = (16, 16)
    centerIdx = 8
    long_spacing = 0.25
    
    if False:
        figureFolder = './figures/canonical_BEV_centerline'
        if not os.path.isdir(figureFolder):
            os.mkdir(figureFolder)
        for i in range(nBeamlets):
            file = os.path.join(dataFolder, 'DoseBEV_beamlet{}.bin'.format(i+1))
            array = np.fromfile(file, dtype=np.float32)
            dim_long = int (array.size / (fluenceShape[0] * fluenceShape[1]))
            BEVshape = (dim_long, ) + fluenceShape
            array = np.reshape(array, BEVshape)
            arrayCenterline = array[:, centerIdx, centerIdx]
            depth = np.arange(dim_long) * long_spacing
            plt.plot(depth, arrayCenterline)
            plt.xlabel('depth (cm)')
            plt.ylabel('dose (a.u.)')
            plt.title('Depth dose of beamlet {}'.format(i+1))
            figureFile = os.path.join(figureFolder, 'beamlet{}.png'.format(i+1))
            plt.savefig(figureFile)
            plt.clf()

    if True:
        figureFolder = './figures/canonical_BEV_partial'
        if not os.path.isdir(figureFolder):
            os.mkdir(figureFolder)
        for i in range(nBeamlets):
            file = os.path.join(dataFolder, 'DoseBEV_beamlet{}.bin'.format(i+1))
            array = np.fromfile(file, dtype=np.float32)
            dim_long = int (array.size / (fluenceShape[0] * fluenceShape[1]))
            BEVshape = (dim_long, ) + fluenceShape
            array = np.reshape(array, BEVshape)
            arrayCenterline = np.sum(array, axis=(1, 2))
            depth = np.arange(dim_long) * long_spacing
            plt.plot(depth, arrayCenterline)
            plt.xlabel('depth (cm)')
            plt.ylabel('dose (a.u.)')
            plt.title('Depth dose of beamlet {}'.format(i+1))
            figureFile = os.path.join(figureFolder, 'beamlet{}.png'.format(i+1))
            plt.savefig(figureFile)
            plt.clf()
    
    if True:
        figureFolder = './figures/canonical_BEV_slice'
        if not os.path.isdir(figureFolder):
            os.mkdir(figureFolder)
        for i in range(nBeamlets):
            file = os.path.join(dataFolder, 'DoseBEV_beamlet{}.bin'.format(i+1))
            array = np.fromfile(file, dtype=np.float32)
            dim_long = int (array.size / (fluenceShape[0] * fluenceShape[1]))
            BEVshape = (dim_long, ) + fluenceShape
            array = np.reshape(array, BEVshape)
            slice = array[:, centerIdx, :]
            figureFile = os.path.join(figureFolder, 'beamlet{}.png'.format(i+1))
            plt.imsave(figureFile, slice)
            plt.clf()


def examine_projection():
    dataFolder = "/data/qifan/FastDoseWorkplace/PhtmOptm/canonical"
    PVCSshape = (103, 103, 103)
    centerIdx = 51
    figureFolder = './figures/canonical_PVCS_proj'
    if not os.path.isdir(figureFolder):
        os.mkdir(figureFolder)

    if False:
        beamletIndices = [118, 119, 120]
        projShape = (PVCSshape[1], PVCSshape[2])
        projSum = np.zeros(projShape, dtype=np.float32)
        for idx in beamletIndices:
            dataFile = os.path.join(dataFolder, 'DosePVCS_beamlet{}.bin'.format(idx))
            array = np.fromfile(dataFile, dtype=np.float32)
            array = np.reshape(array, PVCSshape)
            arrayProjection = np.sum(array, axis=0)
            projSum += arrayProjection
            figureFile = os.path.join(figureFolder, 'PVCSProj{}.png'.format(idx))
            plt.imsave(figureFile, arrayProjection)
        figureFile = os.path.join(figureFolder, 'projSum.png')
        plt.imsave(figureFile, projSum)

    if False:
        doseTotal = np.zeros(PVCSshape, dtype=np.float32)
        nBeamlets = 256
        for i in range(nBeamlets):
            file = os.path.join(dataFolder, 'DosePVCS_beamlet{}.bin'.format(i+1))
            dataArray = np.fromfile(file, dtype=np.float32)
            dataArray = np.reshape(dataArray, PVCSshape)
            doseTotal += dataArray
        # show dose slice
        totalDoseSlice = doseTotal[centerIdx, :, :]
        sliceFile = os.path.join(figureFolder, 'totalDoseSlice.png')
        plt.imsave(sliceFile, totalDoseSlice)

        totalDoseProj = np.sum(doseTotal, axis=0)
        sliceFile = os.path.join(figureFolder, 'totalDosePartial.png')
        plt.imsave(sliceFile, totalDoseProj)
    
    if True:
        # to find out which beamlet has such a high profile
        doseTotal = np.zeros(PVCSshape, dtype=np.float32)
        idx_total = 0
        for j in range(7, 9):
            for i in range(16):
                idx = i + j * 16 + 1
                file = os.path.join(dataFolder, 'DosePVCS_beamlet{}.bin'.format(idx))
                dataArray = np.fromfile(file, dtype=np.float32)
                dataArray = np.reshape(dataArray, PVCSshape)
                doseTotal += dataArray
                file = os.path.join(figureFolder, 'ablation{}.png'.format(idx_total))
                doseArraySlice = doseTotal[centerIdx, :, :]
                plt.imsave(file, doseArraySlice)
                idx_total += 1
    
    if False:
        probIdx = 7 * 16 + 11
        probPVCSFile = os.path.join(dataFolder, 'DosePVCS_beamlet{}.bin'.format(probIdx))
        probPVCS = np.fromfile(probPVCSFile, dtype=np.float32)
        probPVCS = np.reshape(probPVCS, PVCSshape)
        slice = probPVCS[centerIdx, :, :]
        file = os.path.join(figureFolder, 'probSlice.png')
        plt.imsave(file, slice)

        probPVCSFile = os.path.join(dataFolder, 'DosePVCS_beamlet{}.bin'.format(probIdx+1))
        probPVCS = np.fromfile(probPVCSFile, dtype=np.float32)
        probPVCS = np.reshape(probPVCS, PVCSshape)
        slice = probPVCS[centerIdx, :, :]
        file = os.path.join(figureFolder, 'probSliceAdjacent.png')
        plt.imsave(file, slice)


if __name__ == '__main__':
    # showFPangles()
    # checkBeamletDose()
    # checkBeamletPVCS()
    # examine_individual_partial()
    # examine_individual_BEV()
    examine_projection()