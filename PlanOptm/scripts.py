import os
import numpy as np

def FPangles():
    """
    This function reproduces results of FPangels.m
    """
    angles = 6
    N = round(180/angles)
    thetaY = 0
    result = []
    while (thetaY <= 180):
        # print(thetaY)
        nDeltaX = 2 * np.pi * np.cos((thetaY - 90) / 180 * np.pi) \
            / (angles / 180 * np.pi)
        nDeltaX = abs(nDeltaX)
        deltaX = round(360 / nDeltaX)
        thetaX = 0
        while (thetaX <= 359):
            result.append((thetaX, thetaY))
            thetaX += deltaX
        thetaY += angles
    result = np.array(result)
    return result


def FPangles2Coords(fp_angles):
    """
    This function converts the FPangles into azimuth and zenith angles
    """
    angle0 = fp_angles[:, 0]
    angle1 = fp_angles[:, 1]
    angle0 = np.expand_dims(angle0, axis=1)
    angle1 = np.expand_dims(angle1, axis=1)

    x_coords = - np.cos(angle1)
    y_coords = - np.sin(angle1)
    z_coords = - y_coords * np.sin(angle0)
    y_coords = y_coords * np.cos(angle0)
    result = np.concatenate((x_coords, y_coords, z_coords), axis=1)
    return result


def ZenithAzimuth2Coords():
    beamlist_file = "/data/qifan/projects/EndtoEnd/results/PlanOptm/slabPhantom/beamlist_full.txt"
    with open(beamlist_file, 'r') as f:
        lines = f.readlines()
    angles = np.zeros((len(lines), 2))
    for i in range(len(lines)):
        line = lines[i]
        line = line.split(' ')
        angles[i, 0] = float(line[0])
        angles[i, 1] = float(line[1])
    
    angles = angles * np.pi / 180
    angle0 = angles[:, 0]
    angle1 = angles[:, 1]
    angle0 = np.expand_dims(angle0, axis=1)
    angle1 = np.expand_dims(angle1, axis=1)

    y_coords = np.sin(angle0)
    z_coords = np.cos(angle0)
    x_coords = - y_coords * np.sin(angle1)
    y_coords = y_coords * np.cos(angle1)
    result = np.concatenate((x_coords, y_coords, z_coords), axis=1)
    return result


def showFPangles():
    fp_angles = FPangles()
    fp_angles = fp_angles * np.pi / 180

    coords_3d = FPangles2Coords(fp_angles)
    coords_3d_comp = ZenithAzimuth2Coords()
    dot_product = np.sum(coords_3d * coords_3d_comp, axis=1)
    print(dot_product)


if __name__ == '__main__':
    showFPangles()