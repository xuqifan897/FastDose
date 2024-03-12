import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

def read_h5():
    file = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/experiment/Dose_Coefficients.h5"
    file = h5py.File(file, 'r')
    # show_h5_structure(file)

    calc_specs_name = "calc_specs";
    calc_specs = file[calc_specs_name]
    calc_bbox_size = calc_specs.attrs["calc_bbox_size"]
    calc_bbox_start = calc_specs.attrs["calc_bbox_start"]
    full_dicom_size = calc_specs.attrs["full_dicom_size"]
    full_dicom_size = np.flip(full_dicom_size, 0)
    full_dicom_size = np.int32(full_dicom_size)
    print(full_dicom_size)

    arraySize = full_dicom_size[0] * full_dicom_size[1] * full_dicom_size[2]
    beams = file["beams"]
    beams_data = beams["data"]["beam_00000"]
    beams_metadata = beams["metadata"]["beam_00000"]
    beamlet_collection = None
    for key, beamlet in beams_data.items():
        coeffs = np.array(beamlet["coeffs"])
        lindex = np.array(beamlet["lindex"])
        beamlet_local = np.zeros(arraySize, dtype=coeffs.dtype)
        beamlet_local[lindex] = coeffs
        if beamlet_collection is None:
            beamlet_collection = beamlet_local
        else:
            beamlet_collection += beamlet_local
    beamlet_collection = np.reshape(beamlet_collection, full_dicom_size)
    outputFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/experiment/DoseOld.npy"
    np.save(outputFile, beamlet_collection)


def show_h5_structure(group, indent=0):
    """
    Recursively prints the structure of an HDF5 group or dataset.
    """
    if isinstance(group, h5py.Group):
        for attr_name, attr_value in group.attrs.items():
            print(" " * indent + "Attribute: {} - Value: {}".format(attr_name, attr_value))
        for key in group.keys():
            print(" " * indent + f"Group: {key}")
            show_h5_structure(group[key], indent + 2)
    elif isinstance(group, h5py.Dataset):
        print(" " * indent + f"Dataset: {group.name} (shape: {group.shape}, dtype: {group.dtype})")


def doseWash_old():
    expFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/experiment"
    oldDoseFile = os.path.join(expFolder, "DoseOld.npy")
    oldDose = np.load(oldDoseFile)
    shape = oldDose.shape

    densityFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/density.raw"
    density = np.fromfile(densityFile, dtype=np.float32)
    density = np.reshape(density, shape)

    oldDoseOutputFolder = os.path.join(expFolder, "dose_old")
    if (not os.path.isdir(oldDoseOutputFolder)):
        os.mkdir(oldDoseOutputFolder)
    for i in range(shape[0]):
        doseSlice = oldDose[i, :, :]
        imageSlice = density[i, :, :]
        plt.imshow(imageSlice, cmap='gray')
        plt.imshow(doseSlice, alpha=0.5)
        figureFile = os.path.join(oldDoseOutputFolder, 'oldDose{:03d}.png'.format(i))
        plt.savefig(figureFile)
        plt.clf()

        print("{} done!".format(figureFile))


def readDoseNew():
    doseMatFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/" \
        "LUNG/prep_result/doseMatFolder"
    NonZeroElementsFile = os.path.join(doseMatFolder, 'NonZeroElements.bin')
    numRowsPerMatFile = os.path.join(doseMatFolder, 'numRowsPerMat.bin')
    offsetsBufferFile = os.path.join(doseMatFolder, 'offsetsBuffer.bin')
    columnsBufferFile = os.path.join(doseMatFolder, 'columnsBuffer.bin')
    valuesBufferFile = os.path.join(doseMatFolder, 'valuesBuffer.bin')
    phantomShape = [149, 220, 220]
    numElements = phantomShape[0] * phantomShape[1] * phantomShape[2]

    # extract the first matrix
    Non_zero_elements = np.fromfile(NonZeroElementsFile, np.uint64, 1)
    Num_rows = np.fromfile(numRowsPerMatFile, np.uint64, 1)
    Non_zero_elements = Non_zero_elements[0].item()
    Num_rows = Num_rows[0].item()
    
    offsetsBuffer = np.fromfile(offsetsBufferFile, np.uint64, Num_rows+1)
    columnsBuffer = np.fromfile(columnsBufferFile, np.uint64, Non_zero_elements)
    valuesBuffer = np.fromfile(valuesBufferFile, np.float32, Non_zero_elements)
    
    beamlet_collection = None
    for i in range(Num_rows):
        # slice the current row out
        idx_start = offsetsBuffer[i]
        idx_end = offsetsBuffer[i+1]
        columnsSlice = columnsBuffer[idx_start: idx_end]
        valuesSlice = valuesBuffer[idx_start: idx_end]

        # convert the current row into a matrix
        beamlet_local = np.zeros(numElements, dtype=valuesSlice.dtype)
        beamlet_local[columnsSlice] = valuesSlice
        if beamlet_collection is None:
            beamlet_collection = beamlet_local
        else:
            beamlet_collection += beamlet_local
    beamlet_collection = np.reshape(beamlet_collection, phantomShape)
    file = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/experiment/DoseNew.npy"
    np.save(file, beamlet_collection)


def doseWash_new():
    newDoseFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/experiment/DoseNew.npy"
    newDose = np.load(newDoseFile)
    shape = newDose.shape

    densityFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/input/density.raw"
    density = np.fromfile(densityFile, dtype=np.float32)
    density = np.reshape(density, shape)

    newDoseOutputFolder = "/data/qifan/projects/FastDoseWorkplace"\
        "/BOOval/LUNG/experiment/dose_new"
    if not os.path.isdir(newDoseOutputFolder):
        os.mkdir(newDoseOutputFolder)
    for i in range(shape[0]):
        doseSlice = newDose[i, :, :]
        imageSlice = density[i, :, :]
        plt.imshow(imageSlice, cmap='gray')
        plt.imshow(doseSlice, alpha=0.5)
        figureFile = os.path.join(newDoseOutputFolder, 'newDose{:03d}.png'.format(i))
        plt.savefig(figureFile)
        plt.clf()

        print("{} done!".format(figureFile))


def read_metadata():
    file = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/experiment/Dose_Coefficients.h5"
    file = h5py.File(file, 'r')
    keys = list(file.keys())
    
    beams = file["beams"]
    calc_specs = file["calc_specs"]
    filetype = file["filetype"]
    # print(list(calc_specs.attrs))
    # print(list(filetype.attrs))

    beams_metadata = beams["metadata"]
    beams_metadata_beam0 = beams_metadata["beam_00000"]
    print(list(beams_metadata_beam0.attrs))

    beams_data = beams["data"]
    beams_data_beam0 = beams_data["beam_00000"]
    print(list(beams_data_beam0.attrs))


if __name__ == '__main__':
    # read_h5()
    # doseWash_old()
    # readDoseNew()
    # doseWash_new()
    read_metadata()