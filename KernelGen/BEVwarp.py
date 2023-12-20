import os
import numpy as np
import matplotlib.pyplot as plt

def curveAffine():
    """
    This function shows what a "straight line" is like in an affine coordinate system
    """
    SAD = 100  # cm
    halfFluenceSize = 5  # cm
    halfFov = 20  # cm

    # firstly, draw the bounding box
    plt.plot((-halfFov, -halfFov), (-halfFov, halfFov), color='blue')
    plt.plot((halfFov, halfFov), (-halfFov, halfFov), color='blue')
    plt.plot((-halfFov, halfFov), (halfFov, halfFov), color='blue')
    plt.plot((-halfFov, halfFov), (-halfFov, -halfFov), color='blue')


    # secondly, draw the field of view.
    def line1(y):
        return (y - SAD) * halfFluenceSize / (- SAD)

    def line2(y):
        return (y - SAD) * (- halfFluenceSize) / (-SAD)

    point0 = (line1(halfFov), halfFov)
    point1 = (line1(-halfFov), -halfFov)
    plt.plot((point0[0], point1[0]), (point0[1], point1[1]), color='green', linestyle='--')
    point0 = (line2(halfFov), halfFov)
    point1 = (line2(-halfFov), -halfFov)
    plt.plot((point0[0], point1[0]), (point0[1], point1[1]), color='green', linestyle='--')


    # thirdly, plot the rays
    nAngles = 8
    angles = (np.arange(nAngles) + 0.5) * np.pi / nAngles
    step_size = 0.1
    for angle in angles:
        slope = 1 / np.tan(angle)
        def line(x):
            return x * slope
        x_ = -halfFov
        coords_x = []
        coords_y = []
        while(x_ < halfFov):
            y_ = line(x_)
            x_ += step_size
            if y_ < -halfFov or y_ > halfFov or \
                x_ < line2(y_) or x_ > line1(y_):
                continue
            coords_x.append(x_)
            coords_y.append(y_)
        coords_x = np.array(coords_x)
        coords_y = np.array(coords_y)
        coords_x = coords_x * (SAD - coords_y) / SAD
        plt.plot(coords_x, coords_y, color='red')

    plt.xlabel('x coordinate (cm)')
    plt.ylabel('y coordinate (cm)')
    plt.title('the "straight lines" in affine coordinates')
    file = './figures/curveAffine.png'
    plt.axis("equal")
    plt.savefig(file)


def curvePolar():
    """
    This function plots what a "straight line" is like in a polar coordinate system
    """
    SAD = 100  # cm
    halfFluenceSize = 5  # cm
    halfFov = 20  # cm

    # firstly, draw the bounding box
    plt.plot((-halfFov, -halfFov), (-halfFov, halfFov), color='blue')
    plt.plot((halfFov, halfFov), (-halfFov, halfFov), color='blue')
    plt.plot((-halfFov, halfFov), (halfFov, halfFov), color='blue')
    plt.plot((-halfFov, halfFov), (-halfFov, -halfFov), color='blue')


    # secondly, draw the field of view.
    def line1(y):
        return (y - SAD) * halfFluenceSize / (- SAD)

    def line2(y):
        return (y - SAD) * (- halfFluenceSize) / (-SAD)

    point0 = (line1(halfFov), halfFov)
    point1 = (line1(-halfFov), -halfFov)
    plt.plot((point0[0], point1[0]), (point0[1], point1[1]), color='green', linestyle='--')
    point0 = (line2(halfFov), halfFov)
    point1 = (line2(-halfFov), -halfFov)
    plt.plot((point0[0], point1[0]), (point0[1], point1[1]), color='green', linestyle='--')


    # thirdly, plot the rays
    nAngles = 8
    angles = (np.arange(nAngles) + 0.5) * np.pi / nAngles
    step_size = 0.1
    for angle in angles:
        slope = 1 / np.tan(angle)
        def line(x):
            return x * slope
        x_ = - halfFov
        coords_x = []
        coords_y = []
        while(x_ < halfFov):
            y_ = line(x_)
            x_ += step_size
            if y_ < -halfFov or y_ > halfFov or \
                x_ < line2(y_) or x_ > line1(y_):
                continue
            coords_x.append(x_)
            coords_y.append(y_)
        coords_x = np.array(coords_x)
        coords_y = np.array(coords_y)
        radius = SAD - coords_y
        angle = coords_x / SAD
        coords_x = np.sin(angle) * radius
        coords_y = SAD - np.cos(angle) * radius
        plt.plot(coords_x, coords_y, color='red')
    
    plt.xlabel('x coordinate (cm)')
    plt.ylabel('y coordinate (cm)')
    plt.title('the "straight lines" in polar coordinates')
    file = './figures/curvePolar.png'
    plt.axis("equal")
    plt.savefig(file)


if __name__ == '__main__':
    # curveAffine()
    curvePolar()