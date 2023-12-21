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

    nan_mask = np.isnan(array)
    num_nan_elements = np.sum(nan_mask)
    print("The number of nan elements: {}, the total number of elements: {}".format(num_nan_elements, array.size))

    # Find the slice with the most number of nan and take a view
    nan_mask_partial = np.sum(nan_mask, axis=(1, 2))
    print(nan_mask_partial)


    scale = 255
    if True:
        # take a look at the nan slices
        slice0 = np.uint8(nan_mask[0, :, :] * scale)
        file = './figures/nanSlice.png'
        plt.imsave(file, slice0)


    if False:
        # truncate the nan elements and then take a look
        array_truncate = array[6: -10]
        slice = array_truncate[:, 8, :]
        file = './figures/DoseBEVLong.png'
        plt.imsave(file, slice)


if __name__ == '__main__':
    # beamListGen()
    # range_check()
    # view_Terma_BEV()
    # view_Terma_PVCS()
    # kernelView()
    # cross_validate()
    view_Dose_BEV()