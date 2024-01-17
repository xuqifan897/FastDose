import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pydicom
from rt_utils import RTStructBuilder

def examineMasks():
    patFolder = "/data/qifan/FastDoseWorkplace/BOOval/HN02"
    optFolder = os.path.join(patFolder, "experiment")
    maskFile = os.path.join(optFolder, 'Dose_Coefficients.mask')
    with h5py.File(maskFile, 'r') as file:
        print_hdf5_structure(file)


def print_hdf5_structure(item, indent=0):
    """Recursively print the structure of HDF5 file including attributes."""
    if isinstance(item, h5py.File) or isinstance(item, h5py.Group):
        for key in item.keys():
            if True and key == "ArrayProps":
                it = item[key]
                print(it.attrs["size"], type(it.attrs["size"]), it.attrs["size"].dtype)
                print(it.attrs["crop_size"])
                print(it.attrs["crop_start"])
            print(" " * indent + f"Group: {key}")
            print_hdf5_structure(item[key], indent + 2)
    elif isinstance(item, h5py.Dataset):
        print(" " * indent + f"Dataset: {item.name} (Shape: {item.shape}, Dtype: {item.dtype})")


def readMasks():
    patFolder = "/data/qifan/FastDoseWorkplace/BOOval/LUNG"
    optFolder = os.path.join(patFolder, "experiment")
    maskFile = os.path.join(optFolder, 'Dose_Coefficients.mask')
    file = h5py.File(maskFile, 'r')
    structures = file.keys()
    print(structures)
    
    # structName = 'PTV_ENLARGED'
    structName = 'Skin'
    assert structName in structures, "The structure '{}' not in structures".format(structName)
    struct = file[structName]
    structProps = struct["ArrayProps"]
    structMask = struct["mask"]

    structSize = structProps.attrs["size"]
    structCropSize = structProps.attrs["crop_size"]
    structCropStart = structProps.attrs["crop_start"]

    structSize = np.flip(structSize, axis=0)
    structCropSize = np.flip(structCropSize, axis=0)
    structCropStart = np.flip(structCropStart, axis=0)
    print(structSize, structCropSize, structCropStart)

    # construct structure mask matrix
    structMask = np.array(structMask)
    structMask = np.reshape(structMask, structCropSize)
    struct_mask = np.zeros(structSize, dtype=structMask.dtype)
    struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
        structCropStart[1]: structCropStart[1] + structCropSize[1],
        structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
    
    # construct density matrix
    densityFile = os.path.join(patFolder, 'tmp', 'density.raw')
    density = np.fromfile(densityFile, dtype=np.float32)
    density = np.reshape(density, structSize)

    # output figures
    figurePath = os.path.join(patFolder, 'PTVShow')
    if not os.path.isdir(figurePath):
        os.mkdir(figurePath)
    nSlices = structSize[0]
    for i in range(nSlices):
        densitySlice = density[i, :, :]
        maskSlice = struct_mask[i, :, :]
        plt.imshow(densitySlice, cmap='gray')
        plt.contour(maskSlice, linewidths=1, color='g')
        figureFile = os.path.join(figurePath, "{:03d}.png".format(i+1))
        plt.savefig(figureFile)
        plt.clf()
        print(figureFile)


def examinePTV():
    patFolder = "/data/qifan/FastDoseWorkplace/BOOval/HN02"
    optFolder = os.path.join(patFolder, "experiment")
    maskFile = os.path.join(optFolder, 'Dose_Coefficients.mask')
    file = h5py.File(maskFile, 'r')
    structures = file.keys()

    if False:
        structName = "PTVTotal"
        assert structName in structures, "The structure '{}' not in structures".format(structName)
        struct = file[structName]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]
        voxelSize = 0.25

        print("{} mask shape: {}, size: {}".format(structName,
            structCropSize, structCropSize*voxelSize))
    
    for name in structures:
        struct = file[name]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]
        voxelSize = 0.25
        print("Structure: {}, shape: {}, size: {}".format(
            name, structCropSize, structCropSize*voxelSize))
    

def examineDcm():
    dcmFolder = "/data/qifan/FastDoseWorkplace/BOOval/HN04/data"

    if True:
        RTstructFile = os.path.join(dcmFolder, 'RTSTRUCT0001.dcm')
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=dcmFolder, rt_struct_path=RTstructFile)
        structures = rtstruct.get_roi_names()
        
        structName = "PTV_Total"
        assert structName in structures, "{} not in structure list".format(structName)
        structMask = rtstruct.get_roi_mask_by_name(structName)
        print(structMask.shape)
        
        # calculate the bounding box
        axis_list = [0, 1, 2]
        for i in range(3):
            axes = axis_list.copy()
            axes.remove(i)
            axes = tuple(axes)
            partial = np.sum(structMask, axis=axes)
            idx_start = 0
            for idx_start in range(partial.size):
                if partial[idx_start] > 0:
                    break
            idx_stop = partial.size - 1
            while (partial[idx_stop] == 0):
                idx_stop -= 1
            print("dimension: {}, idx_start: {}, idx_stop: {}, range: {}".format(
                i, idx_start, idx_stop, idx_stop - idx_start))
    
    
    exampleFile = "/data/qifan/FastDoseWorkplace/BOOval/HN04/data/1.3.6.1.4.1.14519.5.2.1.2193.7172.101707942003276935481153655764.dcm"
    dicom = pydicom.dcmread(exampleFile)
    PixelSpacing = dicom.PixelSpacing
    SliceThickness = dicom.SliceThickness
    print("haha!")


def maskAlign():
    """
    This function attempts to add a RingStructure into the mask list.
    This structure is created after the dose calculation. Before that,
    we first check the alignment of the masks
    """
    maskFileH5 = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/roi_list.h5'
    maskFolder = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/fullStruct'
    file = h5py.File(maskFileH5, 'r')
    structures = file.keys()
    if True:
        for structName in structures:
            struct = file[structName]
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
            struct_mask = np.zeros(structSize, dtype=structMask.dtype)
            struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
                structCropStart[1]: structCropStart[1] + structCropSize[1],
                structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
            

            maskFile = os.path.join(maskFolder, '{}.bin'.format(structName))
            if not os.path.isfile(maskFile):
                print("Mask file not found for structure: {}\n".format(structName))
                continue
            else:
                print(structName)

            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, structSize)
            maskArray = np.transpose(maskArray, axes=(0, 2, 1))
            diff = maskArray ^ struct_mask
            print(np.sum(maskArray), np.sum(struct_mask), np.sum(diff), '\n')
    
    if False:
        # compare the two masks
        BODY = 'Outer_Contour'
        assert BODY in structures, "{} not in structures".format(BODY)
        struct = file[BODY]
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
        struct_mask = np.zeros(structSize, dtype=structMask.dtype)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask

        figuresFolder = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/BODYcomp'
        if not os.path.isdir(figuresFolder):
            os.mkdir(figuresFolder)
        
        if False:
            nSlices = structSize[0]
            for i in range(nSlices):
                slice = struct_mask[i, :, :]
                file = os.path.join(figuresFolder, 'H5_{:03d}.png'.format(i+1))
                plt.imsave(file, slice)
                print(file)
        
        BODY = 'BODY'
        maskFile = os.path.join(maskFolder, '{}.bin'.format(BODY))
        maskArray = np.fromfile(maskFile, dtype=np.uint8)
        maskArray = np.reshape(maskArray, structSize)
        # manipulation
        maskArray = np.transpose(maskArray, axes=(0, 2, 1))
        if False:
            nSlices = structSize[0]
            for i in range(nSlices):
                slice = maskArray[i, :, :]
                file = os.path.join(figuresFolder, 'mat_{:03d}.png'.format(i+1))
                plt.imsave(file, slice)
                print(file)
        
        if True:
            nSlices = structSize[0]
            for i in range(nSlices):
                struct_mask_slice = struct_mask[i, :, :]
                maskArray_slice = maskArray[i, :, :]
                plt.contour(struct_mask_slice, colors='g')
                plt.contour(maskArray_slice, colors='r')
                plt.legend(['H5', 'mat'])
                file = os.path.join(figuresFolder, '{:03d}.png'.format(i+1))
                plt.savefig(file)
                plt.clf()


def MasksH5Gen():
    """
    This function generates the mask h5 file to keep in
    correspondence with the structures in the StructureInfo.mat
    """
    maskFolder = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/fullStruct'
    files = os.listdir(maskFolder)
    structs = [a.split('.')[0] for a in files]
    shape = (149, 220, 220)
    shape_attribute = np.array((220, 220, 149), dtype=np.uint16)

    if False:
        srcFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/roi_list.h5'
        file = h5py.File(srcFile, 'r')
        print(file.name)
    
    # sort the structs
    structs.remove('PTV')
    structs.remove('BODY')
    structs.insert(0, 'BODY')
    structs.insert(0, 'PTV')
    
    destFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/structs.h5'
    if os.path.isfile(destFile):
        os.remove(destFile)
    with h5py.File(destFile, 'w') as file:
        for idx, struct in enumerate(structs):
            maskFile = os.path.join(maskFolder, '{}.bin'.format(struct))
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, shape)
            maskArray = np.transpose(maskArray, axes=(0, 2, 1))
            maskArray = maskArray.flatten()
            assert maskArray.size == shape[0] * shape[1] * shape[2], "Array size doesn't match"

            subgroup = file.create_group(struct)
            subgroup.attrs["index"] = idx + 1
            subgroup.attrs["name"] = struct
            
            dataset = subgroup.create_dataset("mask", dtype='uint8', data=maskArray)
            ArrayProps = subgroup.create_group("ArrayProps")
            ArrayProps.attrs["size"] = shape_attribute
            ArrayProps.attrs["crop_size"] = shape_attribute
            ArrayProps.attrs["crop_start"] = np.array((0, 0, 0), dtype=np.uint16)


def examineMaskGen():
    """
    Examines the mask generated above
    """
    if False:
        sourceFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/roi_list.h5'
        # sourceFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/structs.h5'
        file = h5py.File(sourceFile, 'r')
        print_hdf5_structure(file)
        return
    
        structures = file.keys()
        for structName in structures:
            struct = file[structName]
            idx = struct.attrs["index"]
            print(structName, idx)
        return

    destFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/structs.h5'
    file = h5py.File(destFile, 'r')
    structures = file.keys()
    Masks = {}
    for structName in structures:
        struct = file[structName]
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
        struct_mask = np.zeros(structSize, dtype=structMask.dtype)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        Masks[structName] = struct_mask
    
    densityFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/density.raw'
    density = np.fromfile(densityFile, dtype=np.float32)
    shape = (149, 220, 220)
    density = np.reshape(density, shape)

    resultFolder = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/structs_new'
    if not os.path.isdir(resultFolder):
        os.mkdir(resultFolder)

    nSlices = shape[0]
    for i in range(nSlices):
        sliceArray = density[i, :, :]
        plt.imshow(sliceArray, cmap='gray')
        for key, value in Masks.items():
            maskSlice = value[i, :, :]
            maskSum = np.sum(maskSlice)
            if (maskSum == 0):
                continue
            plt.contour(maskSlice, levels=[0.5], linewidths=1)

        figureFile = os.path.join(resultFolder, '{:03d}.png'.format(i+1))
        plt.savefig(figureFile)
        plt.clf()


def getOffset(folder, fileName):
    file = os.path.join(folder, fileName)
    info = pydicom.dcmread(file)
    return int(info.InstanceNumber)


def MaskH5Copy():
    """
    Problems with the attempt above. Try to copy
    """
    sourceFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/roi_list.h5'
    destFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/structs_new.h5'
    with h5py.File(sourceFile, 'r') as sourcefile:
        structures = sourcefile.keys()
        with h5py.File(destFile, 'w') as destfile:
            for struct in structures:
                old_group = sourcefile[struct]
                new_group = destfile.copy(old_group, name=struct)
                new_group.attrs.update(old_group.attrs)
                new_group["ArrayProps"].attrs.update(old_group["ArrayProps"].attrs)


def writeRingStruct():
    """
    write ring structures
    """
    shape = (149, 220, 220)
    maskFile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/fullStruct/RingStructure.bin'
    maskArray = np.fromfile(maskFile, dtype=np.uint8)
    maskArray = np.reshape(maskArray, shape)
    maskArray = np.transpose(maskArray, axes=(0, 2, 1))
    maskArray = maskArray.flatten()
    destfile = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/RingStructure.bin'
    maskArray.tofile(destfile)


def viewPreSampling():
    folder = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/optimize'
    file = os.path.join(folder, 'preSamplingView.bin')
    nBeamlets = 895
    preSamplingBeamlet = (19, 28, 28)

    globalShape = (nBeamlets, ) + preSamplingBeamlet
    array = np.fromfile(file, dtype=np.uint8)
    array = np.reshape(array, globalShape)
    
    figuresFolder = os.path.join(folder, 'preSamplingProj')
    if (not os.path.isdir(figuresFolder)):
        os.mkdir(figuresFolder)

    if False:
        for i in range(nBeamlets):
            local_array = array[i, :, :, :]
            proj = np.sum(local_array, axis=0)
            file = os.path.join(figuresFolder, 'beamlet{:03d}.png'.format(i))
            plt.imsave(file, proj)
            print(file)


def viewPVCSDose():
    folder = '/data/qifan/FastDoseWorkplace/BOOval/LUNG/optimize'
    file = os.path.join(folder, 'DoseInterp.bin')
    denseVolume = (149, 220, 220)
    nElementsPerVolume = denseVolume[0] * denseVolume[1] * denseVolume[2]

    if False:
        totalArray = np.fromfile(file, dtype=np.float32)
        nBeamlets = int(totalArray.size / nElementsPerVolume)
        print("Number of beamlets: {}".format(nBeamlets))
        totalShape = (nBeamlets,) + denseVolume
        totalArray = np.reshape(totalArray, totalShape)
        ProjArray = np.sum(totalArray, axis=1)

        figuresFolder = os.path.join(folder, 'samplingProj')
        if not os.path.isdir(figuresFolder):
            os.mkdir(figuresFolder)
        for i in range(nBeamlets):
            filename = os.path.join(figuresFolder, '{:03d}.png'.format(i))
            plt.imsave(filename, ProjArray[i, :, :])
            print(filename)
        
    if True:
        densityFile = "/data/qifan/FastDoseWorkplace/BOOval/LUNG/input/density.raw"
        density = np.fromfile(densityFile, dtype=np.float32)
        density = np.reshape(density, denseVolume)

        nBeamlets = 225
        to_load = nElementsPerVolume * nBeamlets
        totalArray = np.fromfile(file, dtype=np.float32, count=to_load)
        totalShape = (nBeamlets, ) + denseVolume
        totalArray = np.reshape(totalArray, totalShape)
        # sum across all beamlets within the first beam
        totalArray = np.sum(totalArray, axis=0)

        figureFolder = './figures'

        densitySlice = density[:, 110, :]
        cross_section = totalArray[:, 110, :]
        plt.imshow(densitySlice, cmap='gray')
        plt.imshow(cross_section, alpha=0.5)
        image = os.path.join(figureFolder, 'beamOneCrossSection1.png')
        plt.savefig(image)
        plt.clf()

        densitySlice = density[90, :, :]
        cross_section = totalArray[90, :, :]
        plt.imshow(densitySlice, cmap='gray')
        plt.imshow(cross_section, alpha=0.5)
        image = os.path.join(figureFolder, 'beamOneCrossSection2.png')
        plt.savefig(image)
        plt.clf()

        densitySlice = density[:, :, 70]
        cross_section = totalArray[:, :, 70]
        plt.imshow(densitySlice, cmap='gray')
        plt.imshow(cross_section, alpha=0.5)
        image = os.path.join(figureFolder, 'beamOneCrossSection3.png')
        plt.savefig(image)
        plt.clf()


def viewBEVDose():
    rootFolder = "/data/qifan/FastDoseWorkplace/BOOval/LUNG/optimize"
    dataFolder = os.path.join(rootFolder, 'doseCompDebug')
    fmapDim = (16, 16)
    nBeamlets = 202

    imageFolder = os.path.join(rootFolder, 'doseCompDebugView')
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)

    for i in range(nBeamlets):
        file = os.path.join(dataFolder, 'BEVDensity{}.bin'.format(i))
        array = np.fromfile(file, dtype=np.float32)

        dimLong = int(array.size / (fmapDim[0] * fmapDim[1]))
        shape = (dimLong, ) + fmapDim
        array = np.reshape(array, shape)
        proj = array[:, :, 8]
        imageFile = os.path.join(imageFolder, 'BEVDensity{}.png'.format(i))
        plt.imsave(imageFile, proj)

        file = os.path.join(dataFolder, 'BEVDose{}.bin'.format(i))
        array = np.fromfile(file, dtype=np.float32)
        array = np.reshape(array, shape)
        proj = array[:, :, 8]
        imageFile = os.path.join(imageFolder, 'BEVDose{}.png'.format(i))
        plt.imsave(imageFile, proj)

        file = os.path.join(dataFolder, 'BEVTerma{}.bin'.format(i))
        array = np.fromfile(file, dtype=np.float32)
        array = np.reshape(array, shape)
        proj = array[:, :, 8]
        imageFile = os.path.join(imageFolder, 'BEVTerma{}.png'.format(i))
        plt.imsave(imageFile, proj)
        
        print("Beamlet {}".format(i))


if __name__ == '__main__':
    # examineMasks()
    # readMasks()
    # examinePTV()
    # examineDcm()
    # maskAlign()
    # MasksH5Gen()
    # examineMaskGen()
    # MaskH5Copy()
    # writeRingStruct()
    # viewPreSampling()
    # viewPVCSDose()
    viewBEVDose()