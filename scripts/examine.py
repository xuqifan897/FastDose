import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors  as mcolors
from skimage import measure
import h5py


def checkPrep():
    folder = "/data/qifan/projects/FastDoseWorkplace/CORTTune/Liver/prep_output"
    shape = (168, 260, 260)
    densityFile = os.path.join(folder, "density.raw")
    rtFile = os.path.join(folder, "roi_list.h5")

    density = np.fromfile(densityFile, dtype=np.float32)
    density = np.reshape(density, shape)
    file = h5py.File(rtFile, "r")
    names = list(file.keys())
    masks = {}
    for name in names:
        struct = file[name]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]

        structSize = np.flip(structSize, axis=0)
        structCropSize = np.flip(structCropSize, axis=0)
        structCropStart = np.flip(structCropStart, axis=0)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        masks[name] = struct_mask
        print(name, np.sum(struct_mask))
        
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    imageFolder = os.path.join(folder, "view")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    for i in range(shape[0]):
        slice = density[i, :, :]
        plt.imshow(slice, cmap="gray")
        for j, entry in enumerate(masks.items()):
            name, mask = entry
            color = colors[j]
            maskSlice = mask[i, :, :]
            contours = measure.find_contours(maskSlice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend()
        file = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(file)
        plt.clf()
        print(file)


def StructInfoGen():
    """
    This function generates the StructureInfo.csv file for a new plan
    """
    if True:
        # Specify the variables below
        globalFolder = "/data/qifan/projects/FastDoseWorkplace/CORTTune/Liver"
        dimensionFile = os.path.join(globalFolder, "prep_output", "dimension.txt")
        PTV_name = "PTV"
        BODY_name = "Skin"

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    organs = lines[2]
    organs = organs.split(" ")
    organs.remove("")
    
    # bring the PTV and BODY to front
    assert PTV_name in organs and BODY_name in organs, "PTV_name and BODY_name provided is not included."
    organs.remove(PTV_name)
    organs.remove(BODY_name)
    organs = [PTV_name, BODY_name] + organs

    # prepare the StructureInfo content
    content = "Name,maxWeights,maxDose,minDoseTargetWeights,minDoseTarget,OARWeights,IdealDose"
    for name in organs:
        if name == PTV_name:
            newline = "{},100,20,100,20,NaN,20".format(name)
        elif name == BODY_name:
            newline = "{},0,18,NaN,NaN,0,0".format(name)
        else:
            newline = "{},0,18,NaN,NaN,5,0".format(name)
        content = content + "\n" + newline
    
    outputFile = os.path.join(globalFolder, "StructureInfo.csv")
    with open(outputFile, 'w') as f:
        f.writelines(content)


def doseAnalyze():
    prep_output = "/data/qifan/projects/FastDoseWorkplace/CORTTune/Liver/prep_output"
    prep_result = "/data/qifan/projects/FastDoseWorkplace/CORTTune/Liver/plan1"
    dimension = (168, 260, 260)
    VOI_exclude = ["RingStructure", "Skin"]

    roi_listFile = os.path.join(prep_output, "roi_list.h5")
    densityFile = os.path.join(prep_output, "density.raw")
    doseFile = os.path.join(prep_result, "dose.bin")

    density = np.fromfile(densityFile, dtype=np.float32)
    dose = np.fromfile(doseFile, dtype=np.float32)
    density = np.reshape(density, dimension)
    dose = np.reshape(dose, dimension)
    file = h5py.File(roi_listFile, 'r')
    structures_filtered = list(file.keys())
    structures_filtered = [a for a in structures_filtered if a not in VOI_exclude]
    structures_filtered.sort()
    print("Structures to show: ", structures_filtered)

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

    plt.figure(figsize=(8.0, 4.8))
    for struct_name, color in zip(structures_filtered, colors):
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
        DrawDVHLine(struct_dose, color)
    
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Relative volume (%)")
    plt.legend(structures_filtered, bbox_to_anchor=(1.05, 1.05), loc="upper left")
    plt.subplots_adjust(right=0.7)
    plotFile = os.path.join(prep_result, "DVH.png")
    plt.savefig(plotFile)


def DrawDVHLine(struct_dose, color, linestyle='-'):
    """
    This function draws the DVH curve for one structure
    """
    struct_dose = np.sort(struct_dose)
    struct_dose = np.insert(struct_dose, 0, 0.0)
    num_voxels = struct_dose.size
    percentile = (num_voxels-1 - np.arange(num_voxels)) / num_voxels * 100

    plt.plot(struct_dose, percentile, color=color, linestyle=linestyle)


def drawDoseWash():
    prep_output = "/data/qifan/projects/FastDoseWorkplace/CORTTune/Liver/prep_output"
    prep_result = "/data/qifan/projects/FastDoseWorkplace/CORTTune/Liver/plan1"
    dimension = (168, 260, 260)
    VOI_exclude = []

    roi_listFile = os.path.join(prep_output, "roi_list.h5")
    densityFile = os.path.join(prep_output, "density.raw")
    doseFile = os.path.join(prep_result, "dose.bin")

    density = np.fromfile(densityFile, dtype=np.float32)
    dose = np.fromfile(doseFile, dtype=np.float32)
    density = np.reshape(density, dimension)
    dose = np.reshape(dose, dimension)
    dose /= np.max(dose)  # normalize

    file = h5py.File(roi_listFile, 'r')
    structures_filtered = list(file.keys())
    structures_filtered = [a for a in structures_filtered if a not in VOI_exclude]
    structures_filtered.sort()
    print("Structures to show: ", structures_filtered)

    masks = {}
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
        masks[struct_name] = struct_mask
    
    numStructs = len(masks)
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

    imageFolder = os.path.join(prep_result, "doseWash")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)

    for i in range(dimension[0]):
        densitySlice = density[i, :, :]
        plt.imshow(densitySlice, cmap='gray')
        for j in range(numStructs):
            color = colors[j]
            structure_name = structures_filtered[j]
            mask = masks[structure_name]
            mask_slice = mask[i, :, :]
            contours = measure.find_contours(mask_slice, 0.5)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0],
                            linewidth=1, color=color, label=structure_name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color)
        plt.imshow(dose[i, :, :], cmap="jet", vmin=0.0, vmax=1.0, alpha=0.5)
        plt.colorbar()
        imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.legend()
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


if __name__ == "__main__":
    # checkPrep()
    # StructInfoGen()
    doseAnalyze()
    # drawDoseWash()