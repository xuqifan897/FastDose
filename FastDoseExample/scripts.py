import os
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def beamListGen():
    """
    This function generate the beamlist.
    order:
    azimuth, zenith, coll, sad | isocenter.x, isocenter.y, isocenter.z | dimension.x, dimension.y
    """
    nbeams = 1152
    inputFolder = '/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp'

    width = 15
    entries = ['azimuth', 'zenith', 'coll', 'sad', 'isocenter.x', 
        'isocenter.y', 'isocenter.z', 'beamlet_size.x', 'beamlet_size.y',
        'fmap_size.x', 'fmap_size.y', 'long_spacing']
    entries = [a.center(width) for a in entries]
    content = ''.join(entries) + '\n'
    for i in range(nbeams):
        azimuth = (random.random() - 0.5) * 2 * np.pi
        zenith = (random.random() - 0.5) * np.pi / 3
        coll = random.random() * 2 * np.pi
        sad = 100.
        isocenter = [12.875, 12.875, 12.875]
        beamlet_size = [0.25, 0.25]
        dimension = [16, 16]
        long_spacing = 0.25
        
        line_content = [azimuth, zenith, coll, sad] + isocenter + beamlet_size
        line_content = ['{:.4f}'.format(a).center(width) for a in line_content]
        line_content = line_content + [str(a).center(width) for a in dimension]
        line_content = line_content + ['{:.4f}'.format(long_spacing).center(width)]
        if (i != nbeams-1):
            line = ''.join(line_content) + '\n'
        else:
            line = ''.join(line_content)
        content = content + line

    beamlist_file = os.path.join(inputFolder, 'beam_lists.txt')
    with open(beamlist_file, 'w') as f:
        f.write(content)


def range_check():
    """
    This function draws the range graph
    """
    source = (-33.4078, 100.344, -1.51296)
    intersections = [
        (19.3501, 0.250122, 23.1904),
        (6.04131, 25.5001, 16.9587),
        (0.249916, 24.0306, 10.4496),
        (10.7382, 0.250114, 14.1773),
        (25.4999, 5.96876, 14.7989),
        (13.3088, 25.5001, 11.4231),
        (19.7565, 0.250114, 6.86708),
        (6.34518, 25.5001, 4.75311),
        (19.5552, 0.250114, 14.9517),
        (6.19468, 25.5001, 10.7983)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    vertices = [
        [(0.25, 0.25, 0.25), (0.25, 25.5, 0.25), (0.25, 25.5, 25.5), (0.25, 0.25, 25.5)],
        [(25.5, 0.25, 0.25), (25.5, 25.5, 0.25), (25.5, 25.5, 25.5), (25.5, 0.25, 25.5)],

        [(0.25, 0.25, 0.25), (0.25, 0.25, 25.5), (25.5, 0.25, 25.5), (25.5, 0.25, 0.25)],
        [(0.25, 25.5, 0.25), (0.25, 25.5, 25.5), (25.5, 25.5, 25.5), (25.5, 25.5, 0.25)],

        [(0.25, 0.25, 0.25), (0.25, 25.5, 0.25), (25.5, 25.5, 0.25), (25.5, 0.25, 0.25)],
        [(0.25, 0.25, 25.5), (0.25, 25.5, 25.5), (25.5, 25.5, 25.5), (25.5, 0.25, 25.5)]
    ]

    for vertice in vertices:
        plane = Poly3DCollection([vertice], facecolors='blue', alpha=0.25)
        ax.add_collection3d(plane)

    for i in range(len(intersections)):
        intersection = intersections[i]
        x = [source[0], intersection[0]]
        y = [source[1], intersection[1]]
        z = [source[2], intersection[2]]
        ax.plot(x, y, z, label='3D Curve')
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-30, 30])
    ax.set_box_aspect([1, 1, 1])
    file = './figures/rangebench.png'
    fig.savefig(file)


def view_Terma_BEV():
    file = '/data/qifan/projects/EndtoEnd/results/CCCSBench/TermaBEV.bin'
    array = np.fromfile(file, dtype=np.float32)
    fmap_size_x = 16
    fmap_size_y = 16
    assert array.size % (fmap_size_x * fmap_size_y) == 0, "the size doesn't match"
    dim_long = int(array.size / (fmap_size_x * fmap_size_y))
    shape = (dim_long, fmap_size_y, fmap_size_x)
    array = np.reshape(array, shape)

    central_long = int(dim_long / 2)

    slice_long = array[:, :, 8]
    slice_trans = array[central_long, :,:]
    file_long = './figures/TermaBEVLong.png'
    file_trans = './figures/TermaBEVTrans.png'
    plt.imsave(file_long, slice_long)
    plt.imsave(file_trans, slice_trans)

    file = '/data/qifan/projects/EndtoEnd/results/CCCSBench/DensityBEV.bin'
    array = np.fromfile(file, dtype=np.float32)
    array = np.reshape(array, shape)
    slice_long = array[:, :, 8]
    slice_trans = array[central_long, :, :]
    file_long = './figures/DensityBEVLong.png'
    file_trans = './figures/DensityBEVTrans.png'
    plt.imsave(file_long, slice_long)
    plt.imsave(file_trans, slice_trans)


def view_Terma_PVCS():
    file = '/data/qifan/projects/EndtoEnd/results/CCCSBench/TermaPVCS.bin'
    shape = (103, 103, 103)
    array = np.fromfile(file, dtype=np.float32)
    array = np.reshape(array, shape)
    center_idx = 51
    slice = array[:, center_idx, :]
    figureFile = './figures/TermaPVCSSlice.png'
    plt.imsave(figureFile, slice)


def kernelView():
    """
    After generating the raw energy deposition tallies,
    we here we view the dose distribution of the point kernel
    """
    kernelFile = '/data/qifan/projects/EndtoEnd/results/CCCSBench/kernel_6mev.bin'
    marginHead = 100
    marginTail = 20
    radiusDim = 100
    heightRes = 0.1
    radiusRes = 0.05
    shape = (marginHead + marginTail, radiusDim)
    array = np.fromfile(kernelFile, dtype=np.double)
    array = np.reshape(array,shape)

    # firstly, show longitudinal distribution
    longitudinal = np.sum(array, axis=1)
    x_axis = (np.arange(marginHead + marginTail) - marginTail) * heightRes
    plt.plot(x_axis, longitudinal)
    plt.xlabel('depth (cm)')
    plt.ylabel("Energy Deposition (a.u.)")
    plt.title("Longitudinal energy deposition")
    plt.show()
    file = "./figures/KernelLong.png"
    plt.savefig(file)
    plt.clf()

    # secondly, we choose several representative slices to view radial dose distribution
    slices = [30, 40, 60, 80, 100]
    radius = (np.arange(radiusDim) + 0.5) * radiusRes
    for i in range(4):
        sliceIdx = slices[i]
        slice = array[sliceIdx, :]
        plt.plot(radius, slice)
    plt.legend(['depth = 1cm', 'depth = 2cm', 'depth = 4cm', 'depth = 6cm', 'depth = 8cm'])
    plt.xlabel('radius (cm)')
    plt.ylabel('energy deposition (a.u.)')
    plt.title("Radial energy deposition")
    file = './figures/KernelRadius.png'
    plt.savefig(file)
    plt.clf()


def cross_validate():
    """
    This is to validate the correctness of the kernel calculated above.
    Previously, we generated a kernel using a Cartesian grid.
    """
    cylindricalFile = '/data/qifan/projects/EndtoEnd/results/CCCSBench/kernel_6mev.bin'
    CartesianFile = '/data/qifan/projects/EndtoEnd/results/Sept1Point/pointKernel/SD.bin'

    CyHead = 100
    CyTail = 20
    CyRadial = 100
    CyHeightRes = 0.1
    CyRadialRes = 0.05

    CaHead = 50
    CaTail = 5
    CaTrans = 50
    CaRes = 0.1

    ArrayCylinder = np.fromfile(cylindricalFile, dtype=np.double)
    CyShape = (CyHead+CyTail, CyRadial)
    ArrayCylinder = np.reshape(ArrayCylinder, CyShape)

    ArrayCartesian = np.fromfile(CartesianFile, dtype=np.double)
    CaShape = (CaHead+CaTail, CaTrans, CaTrans)
    ArrayCartesian = np.reshape(ArrayCartesian, CaShape)

    # compare the centerline
    CyCenter = ArrayCylinder[:, 0].copy()
    CyCenter /= np.max(CyCenter)
    CyDepth = (np.arange(CyHead + CyTail) - CyTail) * CyHeightRes
    plt.plot(CyDepth, CyCenter)

    CaCenterIdx = int(CaTrans / 2)
    CaCenter = ArrayCartesian[:, CaCenterIdx, CaCenterIdx].copy()
    CaCenter /= np.max(CaCenter)
    CaDepth = (np.arange(CaHead + CaTail) - CaTail) * CaRes
    plt.plot(CaDepth, CaCenter)
    
    plt.legend(["Cylindrical", "Cartesian"])
    plt.xlabel("Depth (cm)")
    plt.ylabel("Energy deposition (a.u.)")
    plt.title("Centerline energy deposition")
    plt.show()
    file = './figures/CyCaCenter.png'
    plt.savefig(file)
    plt.clf()


    # compare the partial energy deposition
    CyPartial = np.sum(ArrayCylinder, axis=1)
    CyPartial /= np.max(CyPartial)

    CaPartial = np.sum(ArrayCartesian, axis=(1, 2))
    CaPartial /= np.max(CaPartial)

    plt.plot(CyDepth, CyPartial)
    plt.plot(CaDepth, CaPartial)
    plt.legend(["Cylindrical", "Cartesian"])
    plt.xlabel("Depth (cm)")
    plt.ylabel("Energy deposition (a.u.)")
    plt.title("Partial energy deposition")
    plt.show()
    file = './figures/CyCaPartial.png'
    plt.savefig(file)
    plt.clf()


def view_Dose_BEV():
    file = '/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseBEV.bin'
    array = np.fromfile(file, dtype=np.float32)
    fmap_size_x = 16
    fmap_size_y = 16
    assert array.size % (fmap_size_x * fmap_size_y) == 0, "the size doesn't match"
    dim_long = int(array.size / (fmap_size_x * fmap_size_y))
    shape = (dim_long, fmap_size_y, fmap_size_x)
    array = np.reshape(array, shape)
    central_long = int(dim_long / 2)

    slice_long = array[:, :, 8]
    slice_trans = array[central_long, :,:]
    file_long = './figures/DoseBEVLong.png'
    file_trans = './figures/DoseBEVTrans.png'
    plt.imsave(file_long, slice_long)
    plt.imsave(file_trans, slice_trans)


def DoseCompDebug():
    debugLogFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseCompDebug.bin"
    shape = (16, 16)
    array = np.fromfile(debugLogFile, dtype=np.float32)
    array = np.reshape(array, shape)

    if False:
        TermaFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/TermaBEV.bin"
    else:
        TermaFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/DensityBEV.bin"
    TermaArray = np.fromfile(TermaFile, dtype=np.float32)
    assert TermaArray.size % (shape[0] * shape[1]) == 0
    dimZ = int(TermaArray.size / (shape[0] * shape[1]))
    TermaShape = (dimZ, shape[1], shape[0])
    TermaArray = np.reshape(TermaArray, TermaShape)
    TermaSlice = TermaArray[0, :, :]
    diff = array - TermaSlice
    print("diff:")
    print(diff)
    print("\noriginal:")
    print(array)


def examine_Dose_BEV():
    file = '/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseBEV.bin'
    array = np.fromfile(file, dtype=np.float32)
    fmap_size_x = 16
    fmap_size_y = 16
    voxelSize = 0.25
    assert array.size % (fmap_size_x * fmap_size_y) == 0, "the size doesn't match"
    dim_long = int(array.size / (fmap_size_x * fmap_size_y))
    shape = (dim_long, fmap_size_y, fmap_size_x)
    array = np.reshape(array, shape)

    partialFlag = False
    if partialFlag:
        # Show partial dose
        doseLine = np.sum(array, axis=(1, 2))
    else:
        # Show centerline dose
        mid_x = int(fmap_size_x / 2)
        mid_y = int(fmap_size_y / 2)
        doseLine = array[:, mid_x, mid_y]
    depth = np.arange(dim_long) * voxelSize
    plt.plot(depth, doseLine)
    plt.xlabel('depth (cm)')
    plt.ylabel('dose')
    if partialFlag:
        plt.title('Partial dose')
        figureFile = './figures/partialDose.png'
    else:
        plt.title("Centerline dose")
        figureFile = './figures/centerlineDose.png'
    plt.savefig(figureFile)
    plt.clf()


def examine_X():
    """
    This function examines the intermediate values XA
    """
    idx_flag = False
    if idx_flag:
        file = "/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseCompDebug_VoxelIdx.bin"
    else:
        file = "/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseCompDebug.bin"
    shape = (1024, 16, 16)
    array = np.fromfile(file, dtype=np.float32)
    array = np.reshape(array, shape)
    # view the partial sum
    array_partial = np.sum(array, axis=(1, 2))

    # get the non zero values
    nNonzero = np.sum(array_partial > 0)
    print("Number of non-zero values: ", nNonzero)
    # get the corresponding dose matrix.
    DoseFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseBEV.bin"
    DoseArray = np.fromfile(DoseFile, np.float32)
    sliceShape = (16, 16)
    dimZ = int(DoseArray.size / (sliceShape[0] * sliceShape[1]))
    print("Dose dimension Z: ", dimZ)

    x_value = np.arange(nNonzero)
    if False:
        plt.plot(x_value, array_partial[:nNonzero])
        plt.xlabel('layer')
        plt.ylabel('partialX')
        plt.title('XA')
        figureFile = './figures/partialX.png'
        plt.savefig(figureFile)
        plt.clf()
    
    if True:
        array_centerline = array[:, 8, 8]
        plt.plot(x_value, array_centerline[:nNonzero])
        plt.xlabel('layer')
        plt.ylabel('centerlineX')
        plt.title('XA')
        if idx_flag:
            figureFile = './figures/centerlineX.png'
        else:
            figureFile = './figures/rayTraceX.png'
        plt.savefig(figureFile)
        plt.clf()


def examineDoseStep():
    """
    This function examines the dose step by step
    """
    DoseFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseDebug.bin"
    XFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseCompDebug_VoxelIdx.bin"
    shape = (1024, 16, 16)
    centerlineIdx = int(shape[2] / 2)

    DoseArray = np.fromfile(DoseFile, dtype=np.float32)
    DoseArray = np.reshape(DoseArray, shape)
    DoseCenterline = DoseArray[:, centerlineIdx, centerlineIdx]

    XArray = np.fromfile(XFile, dtype=np.float32)
    XArray = np.reshape(XArray, shape)
    XArrayCenterline = XArray[:, centerlineIdx, centerlineIdx]

    # find the last non-zero elements
    non_zero_idx = 0
    for i in range(XArrayCenterline.size):
        if XArrayCenterline[i] > 0:
            non_zero_idx = i
    x_coords = np.arange(non_zero_idx)
    plt.plot(x_coords, XArrayCenterline[:non_zero_idx])
    plt.xlabel('layer')
    plt.ylabel('centerlineX')
    plt.title('XA')
    figureFile = './figures/centerlineX.png'
    plt.savefig(figureFile)
    plt.clf()

    plt.plot(x_coords, DoseCenterline[:non_zero_idx])
    plt.xlabel('layer')
    plt.ylabel('centerline ga (a.u.)')
    plt.title('Dose')
    # figureFile = './figures/centerlineGA.png'
    # figureFile = './figures/centerlineLineSegDose.png'
    figureFile = './figures/centerlineSharedDose.png'
    plt.savefig(figureFile)
    plt.clf()

    TermaFile = '/data/qifan/projects/EndtoEnd/results/CCCSBench/TermaBEV.bin'
    TermaArray = np.fromfile(TermaFile, dtype=np.float32)
    TermaDimZ = int(TermaArray.size / (16 * 16))
    TermaShape = (TermaDimZ, 16, 16)
    TermaArray = np.reshape(TermaArray, TermaShape)
    TermaCenterline = TermaArray[:, 8, 8]
    x_coords = np.arange(TermaDimZ)
    plt.plot(x_coords, TermaCenterline)
    plt.xlabel('layer')
    plt.ylabel('centerline Terma (a.u.)')
    plt.title("Terma")
    figureFile = './figures/centerlineTerma.png'
    plt.savefig(figureFile)
    plt.clf()

    RealDoseFile = '/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseBEV.bin'
    RealDoseArray = np.fromfile(RealDoseFile, dtype=np.float32)
    RealDoseDimZ = int(RealDoseArray.size / (16*16))
    RealDoseShape = (RealDoseDimZ, 16, 16)
    RealDoseArray = np.reshape(RealDoseArray, RealDoseShape)
    RealDoseCenterline = RealDoseArray[:, 8, 8]
    x_coords = np.arange(RealDoseDimZ)
    plt.plot(x_coords, RealDoseCenterline)
    plt.xlabel('depth')
    plt.ylabel('centerline dose (a.u.)')
    plt.title('Dose')
    figureFile = './figures/centerlineRealDose.png'
    plt.savefig(figureFile)
    plt.clf()


def overlayTermaAndX():
    XFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/DoseCompDebug_VoxelIdx.bin"
    shape = (1024, 16, 16)
    centerlineIdx = int(shape[2] / 2)
    XArray = np.fromfile(XFile, dtype=np.float32)
    XArray = np.reshape(XArray, shape)
    XArrayCenterline = XArray[:, centerlineIdx, centerlineIdx]
    non_zero_idx = 0
    for i in range(XArrayCenterline.size):
        if XArrayCenterline[i] > 0:
            non_zero_idx = i
    x_coords = np.arange(non_zero_idx)
    plt.plot(x_coords, XArrayCenterline[:non_zero_idx])

    TermaFile = '/data/qifan/projects/EndtoEnd/results/CCCSBench/TermaBEV.bin'
    TermaArray = np.fromfile(TermaFile, dtype=np.float32)
    TermaDimZ = int(TermaArray.size / (16 * 16))
    TermaShape = (TermaDimZ, 16, 16)
    TermaArray = np.reshape(TermaArray, TermaShape)
    TermaCenterline = TermaArray[:, 8, 8]
    x_coords = np.arange(TermaDimZ)
    plt.plot(x_coords, TermaCenterline)
    plt.legend(["XA", "Terma"])
    figureFile = 'figures/overlayTermaAndX.png'
    plt.savefig(figureFile)
    plt.clf()


def debugB():
    """
    The exponential with B is problematic
    """
    localXBFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/localXB.bin"
    localTermaFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/localTerma.bin"
    localExpFile = "/data/qifan/projects/EndtoEnd/results/CCCSBench/localExp.bin"

    localXB = np.fromfile(localXBFile, dtype=np.float32)
    dimZ = int(localXB.size / (16*16))
    shape = (dimZ, 16, 16)
    localXB = np.reshape(localXB, shape)
    localXBCenterline = localXB[:, 8, 8]

    non_zero_idx = 0
    for i in range(localXBCenterline.size):
        if localXBCenterline[i] > 0:
            non_zero_idx = i

    localTerma = np.fromfile(localTermaFile, dtype=np.float32)
    localTerma = np.reshape(localTerma, shape)
    localTermaCenterline = localTerma[:, 8, 8]

    localExp = np.fromfile(localExpFile, dtype=np.float32)
    localExp = np.reshape(localExp, shape)
    localExpCenterline = localExp[:, 8, 8]

    xCoords = np.arange(non_zero_idx)
    plt.plot(xCoords, localTermaCenterline[:non_zero_idx])
    plt.plot(xCoords, localXBCenterline[:non_zero_idx])
    plt.legend(['localTerma', 'localXB'])
    file = './figures/debugB.png'
    plt.savefig(file)
    plt.clf()


if __name__ == '__main__':
    # beamListGen()
    # range_check()
    # view_Terma_BEV()
    # view_Terma_PVCS()
    # kernelView()
    # cross_validate()
    # view_Dose_BEV()
    # DoseCompDebug()
    # examine_Dose_BEV()
    # examine_X()
    examineDoseStep()
    # overlayTermaAndX()
    # debugB()