import os
import numpy as np
import nrrd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
from io import BytesIO

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
sourceFolder = "/data/qifan/projects/FastDoseWorkplace/dockerExample/HeadNeck002"
isoRes = 2.5
# structures irrelevant in the optimization. We have already converted these
# structures to standardized PTV structures with prescription doses 70 and 56
structsExclude = ["TransPTV56", "CTV56", "TransPTV70", "GTV", "CTV56", "avoid",
    "RingStructModify", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "PTVMerge"]
PTVList = ["PTV70", "PTV56"]
body="SKIN"

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

    dimension = os.path.join(sourceFolder, "prep_output", "dimension.txt")
    with open(dimension, "r") as f:
        dimension = f.readline()
    dimension = eval(dimension.replace(" ", ", "))  # order: (x, y, z)
    # Please not that the array definitions in our C++ applications, MATLAB, and numpy are different.
    # In our C++ applications and MATLAB, the arrays are stored in row-major order, which means that
    # the element at coordinates (i, j, k) of an array of dimension (I, J, K) is stored in the linear
    # memory at index: i + I * (j + J * k); While in numpy, the element is stored in the linear memory
    # at index: k + K * (j + J * i). Thus, we should transpose the dimension accordingly
    dimension_flip = np.flip(dimension)  # order (z, x, y)

    # There are four PTVs
    PTVCentroidList = []
    for i in range(4):
        maskLocal = os.path.join(sourceFolder, "InputMask", "PTVSeg{}.bin".format(i))
        maskLocal = np.fromfile(maskLocal, dtype=np.uint8)
        maskLocal = np.reshape(maskLocal, dimension_flip)  # (z, y, x)
        maskLocal = np.transpose(maskLocal, axes=(2, 1, 0))  # (x, y, z)
        # We chose the centroid of the PTV as isocenter. Coordinate order: (x, y, z)
        centroidLocal = centroidCalc(maskLocal)
        PTVCentroidList.append(centroidLocal)
    
    # load SKIN masks
    skin = os.path.join(sourceFolder, "InputMask", "{}.bin".format(body))
    skin = np.fromfile(skin, dtype=np.uint8)
    skin = np.reshape(skin, dimension_flip)
    skin = np.transpose(skin, axes=(2, 1, 0))  # (x, y, z)

    # Often in CT, only a portion of the patient is scanned, resulting in a cut-off phantom.
    # In dose calculation, the beams should not irradiate the patient through the cut-off 
    # planes, either the top or the bottom plane. Here we're to find out the planes
    partialSkin = np.any(skin, axis=(0, 1))
    idxZ = [i for i in range(partialSkin.size) if partialSkin[i]]
    idxZ_min, idxZ_max = min(idxZ), max(idxZ)
    sliceBottom = skin[:, :, idxZ_min]
    # convert the binary 2d array into an interpolator, so that we can index it using 
    # floating point indices, like sliceBottom((20.16, 50.33))
    sliceBottom = RegularGridInterpolator((np.arange(sliceBottom.shape[0]),
        np.arange(sliceBottom.shape[1])), sliceBottom, bounds_error=False, fill_value=0)
    sliceTop = skin[:, :, idxZ_max]
    sliceTop = RegularGridInterpolator((np.arange(sliceTop.shape[0]),
        np.arange(sliceTop.shape[1])), sliceTop, bounds_error=False, fill_value=0)
    
    # calculate the list of valid angles
    validBeamList = []
    for i, entry in enumerate(angleListInit):
        angle, direction = entry
        # alternately assign beams to PTVs
        ptvIdx = i % len(PTVCentroidList)
        ptvCentroid = PTVCentroidList[ptvIdx]

        if abs(direction[2]) < eps:
            # if the z component of the direction is close to 0, the ray is almost 
            # parallel to the x-y plane, thus not intersecting with the cut-off planes.
            # To avoid zero-division, we address this case here.
            validBeamList.append((ptvIdx, ptvCentroid, angle, direction))
            continue
        # calculate the intersection point with the top slice
        k_value = (idxZ_max - ptvCentroid[2]) / direction[2]
        if k_value < 0:
            # Note that the direction is from the source to the isocenter,
            # so the entrance point should have a negative k value, which
            # means that the entrance point is from the isocenter to the source
            intersectionTop = ptvCentroid + k_value * direction
            intersectionTop = intersectionTop[:2]
            intersectionValue = sliceTop(intersectionTop)
            if intersectionValue < eps:
                validBeamList.append((ptvIdx, ptvCentroid, angle, direction))
            continue
        
        # calculate the intersection point with the bottom slice
        k_value = (idxZ_min - ptvCentroid[2]) / direction[2]
        if k_value < 0:
            intersectionBottom = ptvCentroid + k_value * direction
            intersectionBottom = intersectionBottom[:2]
            intersectionValue = sliceBottom(intersectionBottom)
            if intersectionValue < eps:
                validBeamList.append((ptvIdx, ptvCentroid, angle, direction))
    numPreSelect = len(validBeamList)
    
    # Using the heuristics that the number of beams selected is inversely 
    # proportional to the angle between adjacent beams, we adjust the angle
    # so that the final number of selected beams is around the number of desired beams
    angleResAdjust = angleResInit * np.sqrt(numPreSelect / numBeamsDesired)
    angleListAdjust = fpanglesGen(angleResAdjust)
    validAnglesAdjust = []
    for j, entry in enumerate(angleListAdjust):
        angle, direction = entry
        ptvIdx = j % len(PTVCentroidList)
        ptvCentroid = PTVCentroidList[ptvIdx]
        if abs(direction[2]) < eps:
            validAnglesAdjust.append((ptvIdx, ptvCentroid, angle, direction))
            continue
        # calculate the intersection with the top slice
        k_value = (idxZ_max - ptvCentroid[2]) / direction[2]
        if k_value < 0:
            intersectionTop = ptvCentroid + k_value * direction
            intersectionTop = intersectionTop[:2]
            intersectionValue = sliceTop(intersectionTop)
            if intersectionValue < eps:
                validAnglesAdjust.append((ptvIdx, ptvCentroid, angle, direction))
            continue
        
        k_value = (idxZ_min - ptvCentroid[2]) / direction[2]
        if k_value < 0:
            intersectionBottom = ptvCentroid + k_value * direction
            intersectionBottom = intersectionBottom[:2]
            intersectionValue = sliceBottom(intersectionBottom)
            if intersectionValue < eps:
                validAnglesAdjust.append((ptvIdx, ptvCentroid, angle, direction))

    # then separate the list "validAnglesAdjust" into different lists, each
    # corresponding to a PTV segment
    beamListList = [[] for i in range(len(PTVCentroidList))]
    for i in range(len(validAnglesAdjust)):
        entry = validAnglesAdjust[i]
        ptvIdx, direction = entry[0], entry[3]
        gantry, phi = direction2VarianIEC(direction)
        gantryDegree = gantry * 180 / np.pi
        phiDegree = phi * 180 / np.pi
        textLine = "{:.4f} {:.4f} {:.4f}".format(gantryDegree, phiDegree, 0)
        beamListList[ptvIdx].append(textLine)

    # write to file
    for i in range(len(PTVCentroidList)):
        fullText = "\n".join(beamListList[i])
        file = os.path.join(sourceFolder, "beamlist{}.txt".format(i))
        with open(file, "w") as f:
            f.write(fullText)
        print(file)


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


def nrrdVerification():
    nSegs = 4  # 4 PTV segments
    dimension = os.path.join(sourceFolder, "prep_output", "dimension.txt")
    with open(dimension, "r") as f:
        dimension = f.readline()
    dimension = eval(dimension.replace(" ", ", "))
    dimension_flip = np.flip(dimension)

    maskFolder = os.path.join(sourceFolder, "InputMask")
    masks = [b for a in os.listdir(maskFolder) if (b:=a.split(".")[0]) not in structsExclude]
    
    maskDict = {}
    for name in masks:
        maskFile = os.path.join(maskFolder, name + ".bin")
        mask = np.fromfile(maskFile, dtype=np.uint8)
        mask = np.reshape(mask, dimension_flip)  # (z, y, x)
        maskDict[name] = mask

    # generate the beamMask
    PTVSegMaskList = []
    for segIdx in range(nSegs):
        PTVSegMask = os.path.join(maskFolder, "PTVSeg{}.bin".format(segIdx))
        PTVSegMask = np.fromfile(PTVSegMask, dtype=np.uint8)
        PTVSegMask = np.reshape(PTVSegMask, dimension_flip)  # (z, y, x)
        PTVSegMaskList.append(PTVSegMask)

    # generate the beam list
    isocenterMapping = {}
    cumuIdx = 0
    for segIdx in range(nSegs):
        beamListFile = os.path.join(sourceFolder, "beamlist{}.txt".format(segIdx))
        with open(beamListFile, "r") as f:
            beamListLocal = f.readlines()
        beamListLocal = [eval(a.replace(" ", ", ")) for a in beamListLocal]
        beamListLocal = [np.array(a) * np.pi / 180 for a in beamListLocal]
        for entry in beamListLocal:
            isocenterMapping[cumuIdx] = (entry, segIdx)
            cumuIdx += 1

    selectedBeams = os.path.join(sourceFolder, "plan", "metadata.txt")
    with open(selectedBeams, "r") as f:
        selectedBeams = f.readlines()
    selectedBeams = selectedBeams[3]
    selectedBeams = eval(selectedBeams.replace("  ", ", "))
    selectedBeams = [isocenterMapping[a] for a in selectedBeams]
    beamResultList = [[] for i in range(nSegs)]
    for angle, segIdx in selectedBeams:
        beamResultList[segIdx].append(angle)

    beamMaskFastDoseDict = {}
    for segIdx in range(nSegs):
        PTVSegMask = PTVSegMaskList[segIdx]
        PTVSegBeamList = beamResultList[segIdx]
        PTVBeamsMask = genBeamsMask(PTVSegMask, PTVSegBeamList)
        beamMaskFastDoseDict["PTVSeg{}".format(segIdx)] = PTVBeamsMask
        print("Done generating beam mask {}".format(segIdx))
    maskDictResult = maskDict.copy()
    for key, value in beamMaskFastDoseDict.items():
        maskDictResult[key] = value

    structsList = list(maskDictResult.keys())
    colorMap = {structsList[i]: hex_to_rgb(colors[i]) for i in range(len(structsList))}
    
    mask, header = nrrdGen(maskDictResult, colorMap, dimension_flip)
    file = os.path.join(sourceFolder, "beamsView.nrrd")
    nrrd.write(file, mask, header)
    print(file)


def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    result = np.array(result) / 255
    result = "{} {} {}".format(*result)
    return result


def genBeamsMask(PTVSegMask, beamAngles):
    # PTVSegMask order: (z, y, x)
    def rotateAroundAxisAtOriginRHS(p, axis, angle):
        # p: the vector to rotate
        # axis: the rotation axis
        # angle: in rad. The angle to rotate
        sint = np.sin(angle)
        cost = np.cos(angle)
        one_minus_cost = 1 - cost
        p_dot_axis = np.dot(p, axis)
        first_term_coeff = one_minus_cost * p_dot_axis
        result = first_term_coeff * axis + \
            cost * p + \
            sint * np.cross(axis, p)
        return result

    def inverseRotateBeamAtOriginRHS(vec, theta, phi, coll):
        # convert BEV coords to PVCS coords
        tmp = rotateAroundAxisAtOriginRHS(vec, np.array((0, 1, 0)), -(phi + coll))
        sptr = -np.sin(phi)
        cptr = np.cos(phi)
        rotation_axis = np.array((sptr, 0, cptr))
        result = rotateAroundAxisAtOriginRHS(tmp, rotation_axis, theta)
        return result

    def calcCentroid(mask):
        mask = mask > 0
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

    directionsSet = []
    for angle_entry in beamAngles:
        axisBEV = np.array((0, 1, 0))
        axisPVCS = inverseRotateBeamAtOriginRHS(axisBEV,
            angle_entry[0], angle_entry[1], angle_entry[2])
        directionsSet.append(axisPVCS)  # order: (x, y, z)
    
    # calculate the coordinates
    coordsShape = PTVSegMask.shape + (3, )
    coords = np.zeros(coordsShape, dtype=float)
    axis_z = np.arange(coordsShape[0], dtype=float)
    axis_z = np.expand_dims(axis_z, axis=(1, 2))
    axis_y = np.arange(coordsShape[1], dtype=float)
    axis_y = np.expand_dims(axis_y, axis=(0, 2))
    axis_x = np.arange(coordsShape[2], dtype=float)
    axis_x = np.expand_dims(axis_x, axis=(0, 1))
    coords[:, :, :, 0] = axis_z
    coords[:, :, :, 1] = axis_y
    coords[:, :, :, 2] = axis_x
    PTVCentroid = calcCentroid(PTVSegMask)
    PTVCentroid = np.expand_dims(PTVCentroid, axis=(0, 1, 2))
    coords_minus_isocenter = coords - PTVCentroid

    beamsMask = None
    radius = 2
    barLength = 100
    for direction in directionsSet:
        # from (x, y, z) to (z, y, x)
        direction_ = np.array((direction[2], direction[1], direction[0]))
        direction_ = np.expand_dims(direction_, axis=(0, 1, 2))
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
    dose = os.path.join(sourceFolder, "plan", "dose.bin")
    dose = np.fromfile(dose, dtype=np.float32)
    maskFolder = os.path.join(sourceFolder, "InputMask")
    structs = [b for a in os.listdir(maskFolder) if (b:=a.split(".")[0]) not in structsExclude]
    
    # sort the structures
    for a in PTVList:
        structs.remove(a)
    structs.remove(body)
    structs.sort()
    structs.insert(0, PTVList[1])
    structs.insert(0, PTVList[0])
    colorMap = {structs[i]: colors[i] for i in range(len(structs))}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    for struct in structs:
        mask = os.path.join(maskFolder, struct + ".bin")
        mask = np.fromfile(mask, dtype=np.uint8).astype(bool)
        maskDose = dose[mask].copy()
        maskDose = np.sort(maskDose)
        maskDose = np.insert(maskDose, 0, 0.0)
        nPoints = maskDose.size
        yAxis = 100 * (1 - np.arange(nPoints) / (nPoints - 1))
        ax.plot(maskDose, yAxis, label=struct, color=colorMap[struct])
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Dose Volume Histogram")
    ax.legend()
    file = os.path.join(sourceFolder, "DVH.png")
    plt.savefig(file)
    plt.close(fig)
    plt.clf()
    print(file)


def doseWashPlot():
    ImageHeight = 80
    AxialWidth = 80
    CoronalWidth = 80
    SagittalWidth = 80

    dimension = os.path.join(sourceFolder, "prep_output", "dimension.txt")
    with open(dimension, "r") as f:
        dimension = f.readline()
    dimension = eval(dimension.replace(" ", ", "))
    dimension_flip = np.flip(dimension)
    
    density = os.path.join(sourceFolder, "density_raw.bin")
    density = np.fromfile(density, dtype=np.uint16)
    density = np.reshape(density, dimension_flip)  # (z, y, x)

    dose = os.path.join(sourceFolder, "plan", "dose.bin")
    dose = np.fromfile(dose, dtype=np.float32)
    dose = np.reshape(dose, dimension_flip)

    maskFolder = os.path.join(sourceFolder, "InputMask")
    structs = [b for a in os.listdir(maskFolder) if (b:=a.split(".")[0]) not in structsExclude]
    masks = {}
    for struct in structs:
        maskArray = os.path.join(sourceFolder, "InputMask", struct+".bin")
        maskArray = np.fromfile(maskArray, dtype=np.uint8)
        maskArray = np.reshape(maskArray, dimension_flip)
        masks[struct] = maskArray

    bodyMask = masks[body]
    ptv70Name = PTVList[0]
    ptv70Mask = masks[ptv70Name].astype(bool)
    ptvDose = dose[ptv70Mask]
    thresh = np.percentile(ptvDose, 10)
    dose *= 70 / thresh
    dose[np.logical_not(bodyMask)] = 0.0
    threshMask = dose > 0.2 * thresh
    
    ptv56name = PTVList[1]
    ptv56Mask = masks[ptv56name].astype(bool)
    ptvMerge = np.logical_or(ptv70Mask, ptv56Mask)
    z, y, x = calcCentroid(ptvMerge).astype(int)

    colorMap = {structs[i]: colors[i] for i in range(len(structs))}
    doseShowMax = np.max(dose)

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
    file = os.path.join(sourceFolder, "doseWash.png")
    plt.imsave(file, image)
    print(file)


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