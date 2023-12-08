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
    nbeams = 100
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
        line = ''.join(line_content) + '\n'
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
    figureFile = './figures/DensityPVCSSlice.png'
    plt.imsave(figureFile, slice)


if __name__ == '__main__':
    # beamListGen()
    # range_check()
    # view_Terma_BEV()
    view_Terma_PVCS()