import os
import numpy as np
import random

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
        'fmap_size.x', 'fmap_size.y']
    entries = [a.center(width) for a in entries]
    content = ''.join(entries) + '\n'
    for i in range(nbeams):
        azimuth = (random.random() - 0.5) * 2 * np.pi
        zenith = (random.random() - 0.5) * np.pi / 3
        coll = random.random() * 2 * np.pi
        sad = 100.
        isocenter = [12.875, 12.875, 12.875]
        beamlet_size = [1.0, 1.0]
        dimension = [10, 10]
        
        line_content = [azimuth, zenith, coll, sad] + isocenter + beamlet_size
        line_content = ['{:.4f}'.format(a).center(width) for a in line_content]
        line_content = line_content + [str(a).center(width) for a in dimension]
        line = ''.join(line_content) + '\n'
        content = content + line

    beamlist_file = os.path.join(inputFolder, 'beam_lists.txt')
    with open(beamlist_file, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    beamListGen()