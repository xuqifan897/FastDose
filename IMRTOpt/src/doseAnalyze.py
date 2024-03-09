import os
import numpy as np
import matplotlib.pyplot as plt
import h5py


def doseAnalyze():
    inputFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/experiment"
    resultFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/optimize"
    phantomShape = (149, 220, 220)
    VOI_exclude = ["Outer_Contour", "Skin", "RLL_ITV"]

    doseFile = os.path.join(resultFolder, "dose.bin")
    dose = np.fromfile(doseFile, dtype=np.float32)
    dose = np.reshape(dose, phantomShape)
    maskFile = os.path.join(inputFolder, "Dose_Coefficients.mask")
    file = h5py.File(maskFile, 'r')
    structures = file.keys()
    structures_filtered = structures - VOI_exclude
    print("Structures to show: ", structures_filtered)

    for struct_name in structures_filtered:
        struct = file[struct_name]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]

        structSize = np.flip(structSize, axis=0)
        structCropSize = np.flip(structCropSize, axis=0)
        structCropStart = np.flip(structCropStart, axis=0)
        print(structSize, structCropSize, structCropStart)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        struct_dose = dose[struct_mask].copy()
        print(struct_dose.shape)
        DrawDVHLine(struct_dose)
    
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Relative volume (%)")
    plt.legend(structures_filtered)
    plotFile = os.path.join(resultFolder, "DVH.png")
    plt.savefig(plotFile)


def DrawDVHLine(struct_dose):
    """
    This function draws the DVH curve for one structure
    """
    struct_dose = np.sort(struct_dose)
    struct_dose = np.insert(struct_dose, 0, 0.0)
    num_voxels = struct_dose.size
    percentile = (num_voxels - np.arange(num_voxels)) / num_voxels * 100

    plt.plot(struct_dose, percentile)


if __name__ == '__main__':
    doseAnalyze()
