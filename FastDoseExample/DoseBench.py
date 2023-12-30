import os
import numpy as np
import matplotlib.pyplot as plt


def viewDoseBEV():
    dataFolder = "/data/qifan/FastDoseWorkplace/CCCSslab"
    for FmapOn in [2, 4, 8, 16]:
        resZ = 0.08  # cm
        fluence_dim = (16, 16)
        DoseBEVFile = os.path.join(dataFolder, 'DoseBEVFmap{}.bin'.format(FmapOn))
        DoseBEV = np.fromfile(DoseBEVFile, dtype=np.float32)
        long_dim = int(DoseBEV.size / (fluence_dim[0] * fluence_dim[1]))
        shape = (long_dim, fluence_dim[0], fluence_dim[1])
        DoseBEV = np.reshape(DoseBEV, shape)
        centerSliceIdx = int(fluence_dim[0] / 2)
        centerSlice = DoseBEV[:, :, centerSliceIdx]
        figureFile = './figures/DoseBEVFmap{}.png'.format(FmapOn)
        plt.imsave(figureFile, centerSlice)

        centerLine = DoseBEV[:, centerSliceIdx, centerSliceIdx]
        depth = np.arange(long_dim) * resZ
        plt.plot(depth, centerLine)
        plt.xlabel('depth (cm)')
        plt.ylabel('Dose (a.u.)')
        beamWidth = FmapOn * resZ
        plt.title('Beam width: {}cm'.format(beamWidth))
        figureFile = './figures/DoseBEVCenterlineFmap{}.png'.format(FmapOn)
        plt.savefig(figureFile)
        plt.clf()


def viewDosePVCS():
    """
    In the experiment, we set the fluence map size to be 0.5cm, 1cm, 2cm, and 4cm, respectively
    """
    dataFolder = "/data/qifan/projects/EndtoEnd/results/CCCSslab"
    for FmapOn in [2, 4, 8, 16]:
        shape = (103, 103, 103)
        DosePVCSFile = os.path.join(dataFolder, 'DosePVCSFmap{}.bin'.format(FmapOn))
        DosePVCS = np.fromfile(DosePVCSFile, dtype=np.float32)
        DosePVCS = np.reshape(DosePVCS, shape)
        centerIdx = int((shape[0]-1)/2)
        DoseSlice = DosePVCS[centerIdx, :, :]
        figureFile = './figures/DosePVCSFmap{}.png'.format(FmapOn)
        plt.imsave(figureFile, DoseSlice)


def viewDoseOptm():
    """
    In the experiment, we set the fluence map size to be 0.5cm, 1cm, 2cm, and 4cm, respectively
    """
    resZ = 0.1  # cm
    fluence_dim = (16, 16)
    DosePVCSFile = "/data/qifan/FastDoseWorkplace/PhtmOptm/DosePVCS_0.4_1.6.bin"
    PVCSShape = (103, 103, 103)
    PVCSRes = 0.25
    centerlineIdx = 51

    DosePVCS = np.fromfile(DosePVCSFile, dtype=np.float32)
    DosePVCS = np.reshape(DosePVCS, PVCSShape)
    Centerline = DosePVCS[centerlineIdx, :, centerlineIdx]
    depth = np.arange(PVCSShape[1]) * PVCSRes
    plt.plot(depth, Centerline)
    plt.xlabel('depth (cm)')
    plt.ylabel('dose (a.u.)')
    plt.title('Centerline dose\nBeamlet size: 0.4cm, extent: 1.6cm')
    figureFile = './figures/DosePVCS_0.4_1.6.png'
    plt.savefig(figureFile)
    plt.clf()

    depthIdx = int(10.0 / PVCSRes)
    lateralProfile = DosePVCS[centerlineIdx, depthIdx, :]
    offAxis = (np.arange(PVCSShape[1]) - centerlineIdx) * PVCSRes
    plt.plot(offAxis, lateralProfile)
    plt.xlabel("Off axis distance (cm)")
    plt.ylabel('dose (a.u.)')
    plt.title("Lateral dose\nBeamlet size: 0.4cm, extent: 1.6cm")
    figureFile = './figures/DoseLateral_0.4_1.6.png'
    plt.savefig(figureFile)
    plt.clf()


if __name__ == '__main__':
    # viewDoseBEV()
    # viewDosePVCS()
    viewDoseOptm()