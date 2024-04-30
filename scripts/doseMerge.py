import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure, transform
import h5py


def doseMatMerge():
    """
    This function merges different matrices into one
    """
    NonZeroElements_collection = []
    numRowsPerMat_collection = []
    offsetsBuffer_collection = []
    columnsBuffer_collection = []
    valuesBuffer_collection = []
    fluenceMap_collection = []

    expFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient005/FastDose"
    numMatrices = 2
    for i in range(1, numMatrices+1):
        doseMatFolder = os.path.join(expFolder, "doseMat{}/doseMatFolder".format(i))
        NonZeroElementsFile = os.path.join(doseMatFolder, "NonZeroElements.bin")
        numRowsPerMatFile = os.path.join(doseMatFolder, "numRowsPerMat.bin")
        offsetsBufferFile = os.path.join(doseMatFolder, "offsetsBuffer.bin")
        columnsBufferFile = os.path.join(doseMatFolder, "columnsBuffer.bin")
        valuesBufferFile = os.path.join(doseMatFolder, "valuesBuffer.bin")
        fluenceMapFile = os.path.join(doseMatFolder, "fluenceMap.bin")

        NonZeroElements = np.fromfile(NonZeroElementsFile, dtype=np.uint64)
        numRowsPerMat = np.fromfile(numRowsPerMatFile, dtype=np.uint64)
        offsetsBuffer = np.fromfile(offsetsBufferFile, dtype=np.uint64)
        columnsBuffer = np.fromfile(columnsBufferFile, dtype=np.uint64)
        valuesBuffer = np.fromfile(valuesBufferFile, dtype=np.float32)
        fluenceMap = np.fromfile(fluenceMapFile, dtype=np.uint8)

        NonZeroElements_collection.append(NonZeroElements)
        numRowsPerMat_collection.append(numRowsPerMat)
        offsetsBuffer_collection.append(offsetsBuffer)
        columnsBuffer_collection.append(columnsBuffer)
        valuesBuffer_collection.append(valuesBuffer)
        fluenceMap_collection.append(fluenceMap)
        print(doseMatFolder)
    
    NonZeroElements = np.concatenate(NonZeroElements_collection)
    numRowsPerMat = np.concatenate(numRowsPerMat_collection, axis=0)
    offsetsBuffer = np.concatenate(offsetsBuffer_collection, axis=0)
    columnsBuffer = np.concatenate(columnsBuffer_collection, axis=0)
    valuesBuffer = np.concatenate(valuesBuffer_collection, axis=0)
    fluenceMap = np.concatenate(fluenceMap_collection, axis=0)
    print("concatenation")

    targetFolder = os.path.join(expFolder, "doseMatMerge/doseMatFolder")
    if not os.path.isdir(targetFolder):
        os.makedirs(targetFolder)
    NonZeroElements.tofile(os.path.join(targetFolder, "NonZeroElements.bin"))
    numRowsPerMat.tofile(os.path.join(targetFolder, "numRowsPerMat.bin"))
    offsetsBuffer.tofile(os.path.join(targetFolder, "offsetsBuffer.bin"))
    columnsBuffer.tofile(os.path.join(targetFolder, "columnsBuffer.bin"))
    valuesBuffer.tofile(os.path.join(targetFolder, "valuesBuffer.bin"))
    fluenceMap.tofile(os.path.join(targetFolder, "fluenceMap.bin"))
    print(targetFolder)


def StructInfoGen():
    """
    This function generates the StructureInfo.csv file for a new plan
    """
    if True:
        # Specify the variables below
        expFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient005/FastDose"
        dimensionFile = os.path.join(expFolder, "prep_output", "dimension.txt")

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    organs = lines[2]
    organs = organs.split(" ")
    organs.remove("")

    PTVs = [("PTV", 20)]
    irrelevant = ["SKIN"]
    PTV_names = [a[0] for a in PTVs]
    exclude = irrelevant + PTV_names
    OARs = [a for a in organs if a not in exclude]
    print(OARs)

    # prepare the StructureInfo content
    content = "Name,maxWeights,maxDose,minDoseTargetWeights,minDoseTarget,OARWeights,IdealDose"
    for name, dose in PTVs:
        newline = "{},100,{},100,{},NaN,{}".format(name, dose, dose, dose)
        content = content + "\n" + newline
    for name in OARs:
        newline = newline = "{},0,18,NaN,NaN,5,0".format(name)
        content = content + "\n" + newline
    print(content)
    
    outputFile = os.path.join(expFolder, "StructureInfo.csv")
    with open(outputFile, 'w') as f:
        f.write(content)


def getStructures(maskFile:str):
    dataset = h5py.File(maskFile, "r")
    structures_names = list(dataset.keys())
    result = []
    for struct_name in structures_names:
        struct = dataset[struct_name]
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
        result.append((struct_name, struct_mask))
    return result


def drawDoseWash():
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient005"
    expFolder = os.path.join(patientFolder, "FastDose")
    prep_output = os.path.join(expFolder, "prep_output")
    doseShape = (128, 200, 200)
    VOI_exclude = ["RingStructure"]

    roi_listFile = os.path.join(prep_output, "roi_list.h5")
    densityFile = os.path.join(prep_output, "density.raw")
    doseFile = os.path.join(expFolder, "plan1", "dose.bin")

    density = np.fromfile(densityFile, dtype=np.float32)
    dose = np.fromfile(doseFile, dtype=np.float32)
    density = np.reshape(density, doseShape)
    dose = np.reshape(dose, doseShape)
    dose /= np.max(dose)  # normalize

    Masks = getStructures(roi_listFile)
    Masks = [a for a in Masks if a[0] not in VOI_exclude]
    
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

    imageFolder = os.path.join(expFolder, "plan1", "doseWash")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)

    for i in range(doseShape[0]):
        densitySlice = density[i, :, :]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(densitySlice, cmap='gray')
        ax.imshow(dose[i, :, :], cmap="jet", vmin=0, vmax=1, alpha=0.3)
        for j in range(len(Masks)):
            color = colors[j]
            name, mask = Masks[j]
            mask_slice = mask[i, :, :]
            contours = measure.find_contours(mask_slice, 0.5)
            initial = True
            for contour in contours:
                if initial:
                    ax.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    ax.plot(contour[:, 1], contour[:, 0], color=color)
        ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1))
        imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


def drawDVH_opt():
    """
    Draw DVH for the optimized plan
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient005"
    planFolder = os.path.join(patientFolder, "FastDose", "plan1")
    doseFile = os.path.join(planFolder, "dose.bin")
    doseShape = (128, 200, 200)

    maskFile = os.path.join(patientFolder, "FastDose", "prep_output", "roi_list.h5")
    Masks = getStructures(maskFile)

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    exclude = ["SKIN", "RingStructure", "duodenum"]
    PrimaryPTV = "PTV"
    primaryMask = None
    for name, mask in Masks:
        if name == PrimaryPTV:
            primaryMask = mask
            break
    
    Masks = [a for a in Masks if a[0] not in exclude]
    if False:
        names = [a[0] for a in Masks]
        print(names)
        return
    
    primaryMask = primaryMask > 0
    primaryDose = doseArray[primaryMask]
    thresh = np.percentile(primaryDose, 5)
    doseArray = doseArray / thresh * 20
    
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, entry in enumerate(Masks):
        color = colors[i]
        name, mask = entry
        mask = mask > 0
        struct_dose = doseArray[mask]
        struct_dose = np.sort(struct_dose)
        struct_dose = np.insert(struct_dose, 0, 0.0)
        numPoints = struct_dose.size
        percentile = np.linspace(100, 0, numPoints)
        ax.plot(struct_dose, percentile, color=color, label=name)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    DVH_opt_file = os.path.join(patientFolder, "DVH_opt.png")
    plt.savefig(DVH_opt_file)
    plt.clf()


if __name__ == "__main__":
    # doseMatMerge()
    # StructInfoGen()
    drawDoseWash()
    # drawDVH_opt()