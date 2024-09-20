import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict
import nrrd
from skimage import measure
from io import BytesIO

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
patientFolder = "/data/qifan/projects/FastDoseWorkplace/dockerExample/PancreasPatient001"
ptv = "ROI"
skin = "SKIN"
beams = "BEAMS"
isoRes = 2.5  # mm

def validAngleListGen():
    """
    This function generates the number of valid beams
    """
    eps = 1e-4
    # Initially, we take the separation between adjacent beams to be 6 degrees, 
    # and generate the corresponding list of beam angles
    angleResInit = 6 * np.pi / 180
    angleListInit = fpanglesGen(angleResInit)
    numBeamsDesired = 400

    # load the masks
    dimension = os.path.join(patientFolder, "prep_output", "dimension.txt")
    with open(dimension, "r") as f:
        dimension = f.readline()
    dimension = eval(dimension.replace(" ", ", "))  # order: (x, y, z)
    # Please not that the array definitions in our C++ applications, MATLAB, and numpy are different.
    # In our C++ applications and MATLAB, the arrays are stored in row-major order, which means that
    # the element at coordinates (i, j, k) of an array of dimension (I, J, K) is stored in the linear
    # memory at index: i + I * (j + J * k); While in numpy, the element is stored in the linear memory
    # at index: k + K * (j + J * i). Thus, we should transpose the dimension accordingly
    dimension_flip = np.flip(dimension)  # order (z, x, y)
    
    maskFolder = os.path.join(patientFolder, "InputMask")
    ptv = os.path.join(maskFolder, "{}.bin".format(ptv))
    ptv = np.fromfile(ptv, dtype=np.uint8)
    ptv = np.reshape(ptv, dimension_flip)  # (z, x, y)
    ptv = np.transpose(ptv, axes=(2, 1, 0))  # (x, y, z)
    # We chose the centroid of the PTV as isocenter. Coordinate order: (x, y, z)
    ptvCentroid = centroidCalc(ptv)
    
    skin = os.path.join(maskFolder, "{}.bin".format(skin))
    skin = np.fromfile(skin, dtype=np.uint8)
    skin = np.reshape(skin, dimension_flip)
    skin = np.transpose(skin, axes=(2, 1, 0))

    # Often in CT, only a portion of the patient is scanned, resulting in a cut-off phantom.
    # In dose calculation, the beams should not irradiate the patient through the cut-off 
    # planes, either the top or the bottom plane. Here we're to find out the planes
    skinProjZ = np.any(skin, axis=(0, 1))
    idxValid = [i for i in range(skinProjZ.size) if skinProjZ[i] > 0]
    idxMin, idxMax = min(idxValid), max(idxValid)
    sliceBottom = skin[:, :, idxMin].copy()
    # convert the binary 2d array into an interpolator, so that we can index it using 
    # floating point indices, like sliceBottom((20.16, 50.33))
    sliceBottom = RegularGridInterpolator((np.arange(dimension[0]), np.arange(dimension[1])),
        sliceBottom, bounds_error=False, fill_value=0)
    sliceTop = skin[:, :, idxMax].copy()
    sliceTop = RegularGridInterpolator((np.arange(dimension[0]), np.arange(dimension[1])),
        sliceTop, bounds_error=False, fill_value=0)
    
    # calculate the list of valid angles
    validAnglesLocal = []
    for angle, direction in angleListInit:
        if abs(direction[2]) < eps:
            # if the z component of the direction is close to 0, the ray is almost 
            # parallel to the x-y plane, thus not intersecting with the cut-off planes.
            # To avoid zero-division, we address this case here.
            validAnglesLocal.append((angle, direction))
            continue
        # calculate the intersection point with the top slice
        k_value = (idxMax - ptvCentroid[2]) / direction[2]
        if k_value < 0:
            # Note that the direction is from the source to the isocenter,
            # so the entrance point should have a negative k value, which
            # means that the entrance point is from the isocenter to the source
            intersectionTop = ptvCentroid + k_value * direction
            intersectionTop = intersectionTop[:2]  # the (x, y) components
            intersectionValue = sliceTop(intersectionTop)
            if intersectionValue < eps:
                validAnglesLocal.append((angle, direction))
            continue

        # calculate the intersection point with the bottom slice
        k_value = (idxMin - ptvCentroid[2]) / direction[2]
        if k_value < 0:
            intersectionBottom = ptvCentroid + k_value * direction
            intersectionBottom = intersectionBottom[:2]
            intersectionValue = sliceBottom(intersectionBottom)
            if intersectionValue < eps:
                validAnglesLocal.append((angle, direction))
            continue
    numPreSelect = len(validAnglesLocal)
    
    # Using the heuristics that the number of beams selected is inversely 
    # proportional to the angle between adjacent beams, we adjust the angle
    # so that the final number of selected beams is around the number of desired beams
    angleResAdjust = angleResInit * np.sqrt(numPreSelect / numBeamsDesired)
    angleListAdjust = fpanglesGen(angleResAdjust)
    validAnglesAdjust = []
    for angle, direction in angleListAdjust:
        if abs(direction[2]) < eps:
            validAnglesAdjust.append((angle, direction))
            continue
        # calculate the intersection with the top slice
        k_value = (idxMax - ptvCentroid[2]) / direction[2]
        if k_value < 0:
            intersectionTop = ptvCentroid + k_value * direction
            intersectionTop = intersectionTop[:2]
            intersectionValue = sliceTop(intersectionTop)
            if intersectionValue < eps:
                validAnglesAdjust.append((angle, direction))
            continue
        
        # calculate the intersection with the bottom slice
        k_value = (idxMin - ptvCentroid[2]) / direction[2]
        if k_value < 0:
            intersectionBottom = ptvCentroid + k_value * direction
            intersectionBottom = intersectionBottom[:2]
            intersectionValue = sliceBottom(intersectionBottom)
            if intersectionValue < eps:
                validAnglesAdjust.append((angle, direction))
    
    # the angles used above are fpangles, we now convert them to 
    # IEC angels, which are gantry ang couch angles
    VarianIECList = []
    for _, direction in validAnglesAdjust:
        gantry, couch = direction2VarianIEC(direction)

        # convert from rad to degrees
        gantryDegree = gantry * 180 / np.pi
        couchDegree = couch * 180 / np.pi
        # We also add the collimator angle, which is set to default value 0
        entry = "{:.4f} {:.4f} {:.4f}".format(gantryDegree, couchDegree, 0)
        VarianIECList.append(entry)
    
    # split the whole beam list into two sub-lists
    beamList1 = VarianIECList[:200]
    beamList2 = VarianIECList[200:]
    beamListText = "\n".join(VarianIECList)
    beamList1Text = "\n".join(beamList1)
    beamList2Text = "\n".join(beamList2)
    # write them to file
    fileIdxList = ["", "1", "2"]
    textLists = [beamListText, beamList1Text, beamList2Text]
    for i in range(3):
        idx = fileIdxList[i]
        text = textLists[i]
        file = os.path.join(patientFolder, "beamlist{}.txt".format(idx))
        with open(file, "w") as f:
            f.write(text)
        print(file)


def fpanglesGen(angleRes: float):
    """
    Generate angles with angle resolution angleRes
    """
    # angleRes unit: rad
    eps = 1e-4
    angleList = []
    numTheta = round(np.pi / angleRes)
    for thetaIdx in range(numTheta + 1):
        theta = thetaIdx * angleRes
        phiTotal = 2 * np.pi * np.sin(theta)
        numPhi = int(np.ceil(phiTotal / angleRes))
        if numPhi == 0:
            numPhi = 1
        deltaPhi = 2 * np.pi / numPhi
        for phiIdx in range(numPhi):
            phi = phiIdx * deltaPhi
            sinTheta = np.sin(theta)
            cosTheta = np.cos(theta)
            sinPhi = np.sin(phi)
            cosPhi = np.cos(phi)
            direction = np.array((sinTheta * cosPhi, sinTheta * sinPhi, cosTheta))
            angleList.append((np.array((theta, phiIdx * deltaPhi)), direction))
    return angleList


def centroidCalc(ptv):
    ptv = ptv > 0
    totalVoxels = np.sum(ptv)
    
    ptvShape = ptv.shape
    xScale = np.arange(ptvShape[0])
    xScale = np.expand_dims(xScale, axis=(1, 2))
    xCoord = np.sum(ptv * xScale) / totalVoxels

    yScale = np.arange(ptvShape[1])
    yScale = np.expand_dims(yScale, axis=(0, 2))
    yCoord = np.sum(ptv * yScale) / totalVoxels

    zScale = np.arange(ptvShape[2])
    zScale = np.expand_dims(zScale, axis=(0, 1))
    zCoord = np.sum(ptv * zScale) / totalVoxels

    return np.array((xCoord, yCoord, zCoord))


def direction2VarianIEC(direction: np.array):
    eps = 1e-4
    cosGantry = direction[1]
    sinGantry = np.sqrt(1 - cosGantry ** 2)
    gantry = np.arccos(cosGantry)
    if sinGantry < eps:
        minusCouch = 0
    else:
        sinMinusCouch = direction[2] / sinGantry
        cosMinusCouch = - direction[0] / sinGantry
        cosMinusCouch = np.clip(cosMinusCouch, -1, 1)
        minusCouch = np.arccos(cosMinusCouch)
        if sinMinusCouch < 0:
            minusCouch = - minusCouch
    couch = - minusCouch
    return (gantry, couch)


def nrrdVerification():
    # This function generates the nrrd file, which, when fed into Slicer,
    # can visualize the optimization result.
    dimension = os.path.join(patientFolder, "prep_output", "dimension.txt")
    with open(dimension, "r") as f:
        dimension = f.readline()
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)
    dimension_flip = np.flip(dimension)
    maskFolder = os.path.join(patientFolder, "InputMask")
    structs = [a.split(".")[0] for a in os.listdir(maskFolder)]
    maskDict = {}
    for struct in structs:
        maskArray = os.path.join(maskFolder, struct + ".bin")
        maskArray = np.fromfile(maskArray, dtype=np.uint8)
        maskArray = np.reshape(maskArray, dimension_flip)
        maskDict[struct] = maskArray
    
    validBeams = os.path.join(patientFolder, "beamlist.txt")
    with open(validBeams, "r") as f:
        validBeams = f.readlines()
    # convert text data to structured data
    for j in range(len(validBeams)):
        line = validBeams[j]
        line = line.replace(" ", ", ")
        line = np.array(eval(line)) * np.pi / 180  # degree to radian
        validBeams[j] = line

    ptvMask = maskDict[ptv]
    beamListSelected = os.path.join(patientFolder, "plan", "metadata.txt")
    with open(beamListSelected, "r") as f:
        beamListSelected = f.readlines()
    beamListSelected = beamListSelected[3]
    beamListSelected = eval(beamListSelected.replace("  ", ", "))
    beamListSelected = [validBeams[k] for k in beamListSelected]
    
    beamMask = genBeamsMask(ptvMask, beamListSelected)
    maskDictResult = maskDict.copy()
    maskDictResult[beams] = beamMask
    
    # generate the colormap
    structsList = list(maskDictResult.keys())
    # change the order of structsList
    structsList.remove(ptv)
    structsList.remove(skin)
    structsList.remove(beams)
    structsList.sort()
    structsList.insert(0, ptv)  # put PTV to the front
    structsList.append(skin)
    structsList.append(beams)  # put skin and beam to the back
    colorMap = {structsList[i]: hex_to_rgb(colors[i]) for i in range(len(structsList))}

    mask, header = nrrdGen(maskDictResult, colorMap, dimension_flip)
    file = os.path.join(patientFolder, "beamsView.nrrd")
    nrrd.write(file, mask, header)
    print(file)


def genBeamsMask(PTVMask, beamsSelect):
    directionsSelect = []
    for angle in beamsSelect:
        axisBEV = np.array((0, 1, 0))
        axisPVCS = inverseRotateBeamAtOriginRHS(axisBEV, angle[0], angle[1], angle[2])
        directionsSelect.append(axisPVCS)
    
    # calculate the coordinates
    coordsShape = PTVMask.shape + (3, )
    coords = np.zeros(coordsShape, dtype=float)
    axis_z = np.arange(coordsShape[0])
    axis_z = np.expand_dims(axis_z, axis=(1, 2))
    axis_y = np.arange(coordsShape[1])
    axis_y = np.expand_dims(axis_y, axis=(0, 2))
    axis_x = np.arange(coordsShape[2])
    axis_x = np.expand_dims(axis_x, axis=(0, 1))
    coords[:, :, :, 0] = axis_z
    coords[:, :, :, 1] = axis_y
    coords[:, :, :, 2] = axis_x
    PTVCentroid = centroidCalc(PTVMask)
    PTVCentroid = np.expand_dims(PTVCentroid, axis=(0, 1, 2))
    coords_minus_isocenter = coords - PTVCentroid

    beamsMask = None
    radius = 2
    barLength = 100
    for direction in directionsSelect:
        # from (x, y, z) to (z, y, x)
        direction = np.flip(direction)
        direction_ = np.expand_dims(direction, axis=(0, 1, 2))
        alongAxisProjection = np.sum(coords_minus_isocenter * direction_, axis=3, keepdims=True)
        perpendicular = coords_minus_isocenter - alongAxisProjection * direction_
        distance = np.linalg.norm(perpendicular, axis=3)

        alongAxisProjection = np.squeeze(alongAxisProjection)
        localMask = distance < radius
        localMask = np.logical_and(localMask, alongAxisProjection < 0)
        localMask = np.logical_and(localMask, alongAxisProjection > -barLength)

        if beamsMask is None:
            beamsMask = localMask
        else:
            beamsMask = np.logical_or(beamsMask, localMask)
    return beamsMask


def inverseRotateBeamAtOriginRHS(vec: np.ndarray, theta: float, phi: float, coll: float):
    tmp = rotateAroundAxisAtOrigin(vec, np.array((0., 1., 0.)), -(phi+coll))  # coll rotation + correction
    sptr = np.sin(-phi)
    cptr = np.cos(-phi)
    rotation_axis = np.array((sptr, 0., cptr))
    result = rotateAroundAxisAtOrigin(tmp, rotation_axis, theta)
    return result


def rotateAroundAxisAtOrigin(p: np.ndarray, r: np.ndarray, t: float):
    # ASSUMES r IS NORMALIZED ALREADY and center is (0, 0, 0)
    # p - vector to rotate
    # r - rotation axis
    # t - rotation angle
    sptr = np.sin(t)
    cptr = np.cos(t)
    result = np.array((
        (-r[0]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[0]*cptr + (-r[2]*p[1] + r[1]*p[2])*sptr,
        (-r[1]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[1]*cptr + (+r[2]*p[0] - r[0]*p[2])*sptr,
        (-r[2]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[2]*cptr + (-r[1]*p[0] + r[0]*p[1])*sptr
    ))
    return result


def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    result = np.array(result) / 255
    result = "{} {} {}".format(*result)
    return result


def nrrdGen(maskDict, colorMap, dimensionOrg):
    nStructs = len(maskDict)
    dimensionFlip = (dimensionOrg[2], dimensionOrg[1], dimensionOrg[0])
    fullDimension = (nStructs,) + dimensionFlip
    fullDimension = np.array(fullDimension, dtype=np.int64)

    space_directions = np.array([
        [np.nan, np.nan, np.nan],
        [isoRes, 0, 0],
        [0, isoRes, 0],
        [0, 0, isoRes]
    ])
    space_origin = np.array((0, 0, 0), dtype=np.float64)

    header_beginning = [
        ("type", "uint8"),
        ("dimension", 4),
        ("space", "left-posterior-superior"),
        ("sizes", fullDimension),
        ("space directions", space_directions),
        ("kinds", ["list", "domain", "domain", "domain"]),
        ("encoding", "gzip"),
        ("space origin", space_origin)
    ]

    header_ending = [
        ("Segmentation_ContainedRepresentationNames", "Binary labelmap|Closed surface|"),
        ("Segmentation_ConversionParameters",""),
        ("Segmentation_MasterRepresentation","Binary labelmap"),
        ("Segmentation_ReferenceImageExtentOffset", "0 0 0")
    ]
    extent_str = "0 {} 0 {} 0 {}".format(*dimensionFlip)

    header_middle = []
    seg_array = np.zeros(fullDimension, dtype=np.uint8)
    for i, entry in enumerate(maskDict.items()):
        name, localMask = entry
        seg_array[i, :, :, :] = np.transpose(localMask)
        key_header = "Segment{}_".format(i)
        
        color = colorMap[name]
        header_middle.append((key_header + "Color", color))
        header_middle.append((key_header + "ColorAutoGenerated", "1"))
        header_middle.append((key_header + "Extent", extent_str))
        header_middle.append((key_header + "ID", name))
        header_middle.append((key_header + "LabelValue", "1"))
        header_middle.append((key_header + "Layer", i))
        header_middle.append((key_header + "Name", name))
        header_middle.append((key_header + "NameAutoGenerated", "1"))
        header_middle.append((key_header + "Tags",
            "DicomRtImport.RoiNumber:{}|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy ".format(i+1)))

    header_middle.sort(key = lambda a: a[0])
    header_result = header_beginning + header_middle + header_ending
    header_result = OrderedDict(header_result)

    return seg_array, header_result


def DVHplot():
    dose = os.path.join(patientFolder, "plan", "dose.bin")
    dose = np.fromfile(dose, dtype=np.float32)
    maskFolder = os.path.join(patientFolder, "InputMask")
    structs = [a.split(".")[0] for a in os.listdir(maskFolder)]
    maskDict = {}
    for struct in structs:
        maskArray = os.path.join(maskFolder, struct + ".bin")
        maskArray = np.fromfile(maskArray, dtype=np.uint8)
        maskDict[struct] = maskArray
    
    structsList = list(maskDict.keys())
    structsList.remove(ptv)
    structsList.remove(skin)  # no skin needed
    structsList.sort()
    structsList.insert(0, ptv)
    colorMap = {structsList[i]: colors[i] for i in range(len(structsList))}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    for struct in structsList:
        structMask = maskDict[struct].astype(bool)
        structDose = dose[structMask].copy()
        structDose = np.sort(structDose)
        structDose = np.insert(structDose, 0, 0.0)  # add one voxel with 0 dose
        nPoints = structDose.size
        yAxis = 100 * (1 - np.arange(nPoints) / (nPoints - 1))
        ax.plot(structDose, yAxis, label=struct, color=colorMap[struct])
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Dose Volume Histogram")
    ax.legend()
    file = os.path.join(patientFolder, "DVH.png")
    plt.savefig(file)
    plt.close(fig)
    plt.clf()
    print(file)


def doseWashPlot():
    ImageHeight = 80
    AxialWidth = 80
    CoronalWidth = 80
    SagittalWidth = 80

    dimension = os.path.join(patientFolder, "prep_output", "dimension.txt")
    with open(dimension, "r") as f:
        dimension = f.readline()
    dimension = eval(dimension.replace(" ", ", "))
    dimension_flip = np.flip(dimension)
    
    density = os.path.join(patientFolder, "density_raw.bin")
    density = np.fromfile(density, dtype=np.uint16)
    density = np.reshape(density, dimension_flip)  # (z, y, x)

    dose = os.path.join(patientFolder, "plan", "dose.bin")
    dose = np.fromfile(dose, dtype=np.float32)
    dose = np.reshape(dose, dimension_flip)
    doseShowMax = np.max(dose)

    masks = {}
    relevantStructures = ["ROI", "Stomach_duo_planCT", "Bowel_sm_planCT",
        "kidney_left", "kidney_right", "liver", "SKIN"]
    colorMap = {relevantStructures[i]: colors[i] for i in range(len(relevantStructures))}
    for name in relevantStructures:
        filename = os.path.join(patientFolder, "InputMask", "{}.bin".format(name))
        maskArray = np.fromfile(filename, dtype=np.uint8).astype(bool)
        maskArray = np.reshape(maskArray, dimension_flip)
        masks[name] = maskArray
    
    # dose normalize
    bodyMask = masks[skin]
    dose[np.logical_not(bodyMask)] = 0
    ptvMask = masks[ptv]
    ptvDose = dose[ptvMask]
    ptvThresh = np.percentile(ptvDose, 5)  # Use D95 as the prescription dose
    doseWashThresh = 0.2 * ptvThresh
    threshMask = dose > doseWashThresh
    z, y, x = calcCentroid(ptvMask).astype(int)

    densityAxial = density[z, :, :]
    doseAxial = dose[z, :, :]
    masksAxial = [(name, array[z, :, :]) for name, array in masks.items()]
    bodyAxial = bodyMask[z, :, :]
    threshMaskAxial = threshMask[z, :, :]
    axialImage = drawSlice(densityAxial, doseAxial, masksAxial, bodyAxial,
        ImageHeight, AxialWidth, colorMap, doseShowMax, threshMaskAxial)
    
    densityCoronal = density[:, y, :]
    doseCoronal = dose[:, y, :]
    masksCoronal = [(name, array[:, y, :]) for name, array in masks.items()]
    bodyCoronal = bodyMask[:, y, :]
    threshMaskCoronal = threshMask[:, y, :]
    coronalImage = drawSlice(densityCoronal, doseCoronal, masksCoronal, bodyCoronal,
        ImageHeight, CoronalWidth, colorMap, doseShowMax, threshMaskCoronal)
    coronalImage = np.flip(coronalImage, axis=0)  # correct for the upside-down

    densitySagittal = density[:, :, x]
    doseSagittal = dose[:, :, x]
    masksSagittal = [(name, array[:, :, x]) for name, array in masks.items()]
    bodySagittal = bodyMask[:, :, x]
    threshMaskSagittal = threshMask[:, :, x]
    sagittalImage = drawSlice(densitySagittal, doseSagittal, masksSagittal, bodySagittal,
        ImageHeight, SagittalWidth, colorMap, doseShowMax, threshMaskSagittal)
    sagittalImage = np.flip(sagittalImage, axis=0)
    
    image = np.concatenate((axialImage, coronalImage, sagittalImage), axis=1)
    imageFile = os.path.join(patientFolder, "doseWash.png")
    plt.imsave(imageFile, image)
    print(imageFile)


def calcCentroid(mask):
    nVoxels = np.sum(mask)
    shape = mask.shape

    xWeight = np.arange(shape[0])
    xWeight = np.expand_dims(xWeight, axis=(1, 2))
    xCoord = np.sum(mask * xWeight) / nVoxels

    yWeight = np.arange(shape[1])
    yWeight = np.expand_dims(yWeight, axis=(0, 2))
    yCoord = np.sum(mask * yWeight) / nVoxels

    zWeight = np.arange(shape[2])
    zWeight = np.expand_dims(zWeight, axis=(0, 1))
    zCoord = np.sum(mask * zWeight) / nVoxels

    result = np.array((xCoord, yCoord, zCoord))
    return result


def calcCentroid2d(mask):
    mask = mask > 0
    nPixels = np.sum(mask)
    shape = mask.shape

    xWeight = np.arange(shape[0])
    xWeight = np.expand_dims(xWeight, axis=1)
    xCoord = np.sum(xWeight * mask) / nPixels

    yWeight = np.arange(shape[1])
    yWeight = np.expand_dims(yWeight, axis=0)
    yCoord = np.sum(yWeight * mask) / nPixels

    result = np.array((xCoord, yCoord))
    return result


def drawSlice(densitySlice, doseSlice, maskSlice, bodySlice,
    height, width, colorMap, doseShowMax, threshMask):
    doseThresh = 10
    maskCentroid = calcCentroid2d(bodySlice)
    densityCrop = crop_and_fill(densitySlice, maskCentroid, height, width)
    doseCrop = crop_and_fill(doseSlice, maskCentroid, height, width)
    threshMaskCrop = crop_and_fill(threshMask, maskCentroid, height, width)
    maskSliceCrop = []
    for name, mask_slice in maskSlice:
        mask_slice_crop = crop_and_fill(mask_slice, maskCentroid, height, width)
        maskSliceCrop.append((name, mask_slice_crop))
    fig, ax = plt.subplots(figsize=(width/50, height/50), dpi=200)
    ax.imshow(densityCrop, cmap="gray", vmin=0, vmax=2000)
    for name, mask in maskSliceCrop:
        color = colorMap[name]
        contours = measure.find_contours(mask)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=0.5)
    ax.imshow(doseCrop, cmap="jet", vmin=0, vmax=doseShowMax, alpha=threshMaskCrop*0.3)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    plt.clf()

    buf.seek(0)
    image = plt.imread(buf)
    buf.close()
    return image


def crop_and_fill(array, center, height, width):
    # height and width are in half
    # crop an array, and fill the out-of-range values with 0
    topLeftAngleArray = np.array((center[0] - height, center[1] - width)).astype(int)
    bottomRightAngleArray = np.array((center[0] + height, center[1] + width)).astype(int)

    topLeftBound = topLeftAngleArray.copy()
    topLeftBound[topLeftBound < 0] = 0
    bottomRightBound = bottomRightAngleArray.copy()
    if bottomRightBound[0] >= array.shape[0] - 1:
        bottomRightBound[0] = array.shape[0] - 1
    if bottomRightBound[1] >= array.shape[1] - 1:
        bottomRightBound[1] = array.shape[1] - 1
    
    startIdx = topLeftBound - topLeftAngleArray
    endIdx = bottomRightBound - topLeftAngleArray
    canvas = np.zeros((2*height, 2*width), dtype=array.dtype)
    canvas[startIdx[0]: endIdx[0], startIdx[1]: endIdx[1]] = \
        array[topLeftBound[0]: bottomRightBound[0], topLeftBound[1]: bottomRightBound[1]]
    return canvas


if __name__ == "__main__":
    # validAngleListGen()
    # nrrdVerification()
    # DVHplot()
    doseWashPlot()