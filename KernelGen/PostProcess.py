import os
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import grad, hessian, jit, vmap

eps = 1e-7

def ViewKernelAngles():
    """
    This function views the 
    """
    convPhiAnglesFile = '/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/convolution_phi_angles.raw'
    convThetaAnglesFile = '/data/qifan/projects/EndtoEnd/results/CCCSclone/tmp/convolution_theta_angles.raw'

    convPhiAngles = np.fromfile(convPhiAnglesFile, dtype=np.float32)
    convThetaAngles = np.fromfile(convThetaAnglesFile, dtype=np.float32)
    print(convPhiAngles)
    print(convThetaAngles)

def Process():
    """
    This function transforms the cylindrical dose kernel into a polar kernel
    """
    folder = "/data/qifan/projects/EndtoEnd/results/CCCSBench"
    specFile = os.path.join(folder, "kernel_precise.bin")
    nAngles = 8
    marginHead = 100
    marginTail = 20
    radiusDim = 100
    heightRes = 0.1  # cm
    radiusRes = 0.05 # cm
    superSampling = 100
    # superSampling = 1

    angleInterval = np.pi / nAngles
    CyShape = (marginTail + marginHead, radiusDim)
    CySpec = np.fromfile(specFile, dtype=np.float64)
    CySpec = np.reshape(CySpec, CyShape)
    coeffShape = (CyShape[0]*superSampling, CyShape[1]*superSampling)

    # Assume the radius resolution, as well as the radiusDim, are the same as the raw data.

    xcoordsMat = (np.arange(coeffShape[1]) + 0.5) * radiusRes / superSampling
    xcoordsMat = np.expand_dims(xcoordsMat, axis=0)
    xcoordsMat = [xcoordsMat] * coeffShape[0]
    xcoordsMat = np.concatenate(xcoordsMat, axis=0)

    ycoordsMat = (np.arange(coeffShape[0]) - 0.5 * superSampling - marginTail * superSampling) * heightRes / superSampling
    ycoordsMat = np.expand_dims(ycoordsMat, axis=1)
    ycoordsMat = [ycoordsMat] * coeffShape[1]
    ycoordsMat = np.concatenate(ycoordsMat, axis=1)

    cot_coords = ycoordsMat / xcoordsMat
    distance = np.sqrt(xcoordsMat**2 + ycoordsMat**2)

    # for debug purposes
    if False:
        cot_coords_file = os.path.join(folder, 'cot_coords.png')
        distance_file = os.path.join(folder, 'distance.png')
        plt.imsave(cot_coords_file, np.arctan(cot_coords))
        plt.imsave(distance_file, distance)

    kernelTable = np.zeros((nAngles, radiusDim*2))

    for angleIdx in range(nAngles):
        angleLower = angleIdx * angleInterval
        angleHigher = (angleIdx + 1) * angleInterval

        if angleLower < eps:
            cot_angleLower = 1 / eps
        else:
            cot_angleLower = 1 / np.tan(angleLower)

        if angleLower > np.pi - eps:
            cot_angleHigher = - 1 / eps
        else:
            cot_angleHigher = 1 / np.tan(angleHigher)
        
        flag_angleLower = cot_coords <= cot_angleLower
        flag_angleHigher = cot_coords > cot_angleHigher
        for radiusIdx in range(radiusDim*2):  # the radius resolution is half of the height resolution
            radiusLower = radiusIdx * radiusRes
            radiusHigher = (radiusIdx + 1) * radiusRes
            flag_radiusLower = distance >= radiusLower
            flag_radiusHigher = distance < radiusHigher
            result = flag_angleLower * flag_angleHigher * flag_radiusLower * flag_radiusHigher

            # for debug purposes
            if False and radiusIdx == 20:
                firstSegment_file = os.path.join(folder, 'firstSegment.png')
                plt.imsave(firstSegment_file, result)
                break
            
            # calculate the weight of individual pixels
            sub_result = [np.hsplit(row, CyShape[1]) for row in np.vsplit(result, CyShape[0])]
            sub_result = np.array(sub_result)
            sub_result = np.sum(sub_result, axis=(2, 3))
            sub_result = sub_result / superSampling**2

            if False and radiusIdx == 20:
                firstSegment_small_file = os.path.join(folder, 'firstSegmentSmall.png')
                plt.imsave(firstSegment_small_file, sub_result)
                break

            kernelTable[angleIdx, radiusIdx] = np.sum(sub_result * CySpec)
            print("angle: {}, radius: {}".format(angleIdx, radiusIdx))
    
    resultFile = os.path.join(folder, 'tabulatedKernel_{}_{}.npy'.format(nAngles, radiusDim*2))
    np.save(resultFile, kernelTable)


def kernelView():
    """
    This function is to show the CCCS kernel generated
    """
    folder = "/data/qifan/projects/EndtoEnd/results/CCCSBench"
    nAngles = 8
    radiusDim = 100
    radiusRes = 0.05
    kernelFile = os.path.join(folder, 'tabulatedKernel_{}_{}.npy'.format(nAngles, radiusDim*2))
    kernel = np.load(kernelFile)
    xCoords = (np.arange(radiusDim*2) + 0.5) * radiusRes
    # plt.plot(xCoords, kernel[0, :])
    # plotFile = os.path.join(folder, 'FirstAngleKernel.png')
    # plt.savefig(plotFile)
    # plt.clf()

    for i in range(nAngles):
        plt.plot(xCoords, kernel[i, :])
        plt.xlabel('radius (cm)')
        plt.ylabel('kernel value (a.u.)')
        plt.title('kernel of angle range: [{}pi/8, {}pi/8)'.format(i, i+1))
        file = './figures/kernelPlot{}.png'.format(i)
        plt.savefig(file)
        plt.clf()


def kernelFitting():
    """
    The CCCS algorithm can be accelerated by approximating 
    the energy deposition kernel into an exponential form:
    h(r, \theta) = (A_\theta e^{-a_\theta r} + B_\theta e^{-b_theta r})
    """
    folder = "/data/qifan/projects/EndtoEnd/results/CCCSBench"
    nAngles = 8
    radiusDim = 100
    radiusRes = 0.05
    tabulatedKernelFile = os.path.join(folder,
        'tabulatedKernel_{}_{}.npy'.format(nAngles, radiusDim*2))
    kernel = np.load(tabulatedKernelFile)
    xCoords = (np.arange(radiusDim*2) + 0.5) * radiusRes
    truncates = [200, 200, 100, 100, 100, 80, 60, 40]
    results = np.zeros((nAngles, 4))

    Norm = True
    for i in range(nAngles):
        tr = truncates[i]

        kernelAngle = kernel[i, :tr].copy()
        results[i, :] = fit_kernel(kernelAngle, xCoords[:tr], Norm)
        print(results[i, :])
    
    if Norm:
        kernelFile = os.path.join(folder, 'exponentialKernelNorm.npy')
    else:
        kernelFile = os.path.join(folder, 'exponentialKernel.npy')
    np.save(kernelFile, results)

def fit_kernel(kernel: np.array, xCoords: np.array, Norm: bool):
    """
    We plan to apply Newton's algorithm to obtain the parameters.
    Here, due to the form of exponential kernels, we plan to firstly
    sum (integrate) the kernel, which results in another curve with
    the same decaying rates but smoother.
    """
    cumu_kernel = np.cumsum(kernel)
    if Norm:
        cumu_kernel -= cumu_kernel[0]
    scale = np.max(cumu_kernel)
    cumu_kernel /= scale

    if False:
        plt.plot(xCoords, cumu_kernel)
        file = './figures/cumuKernel.png'
        plt.savefig(file)
        plt.clf()
    
    # parameter initialization
    A = 0.8
    a = 2.1
    B = 0.3
    b = 0.06
    parameters = jnp.array([A, a, B, b])
    nIters = 3000
    minimumLoss = 100.
    result = None
    for i in range(nIters):
        loss = objective_function(parameters, xCoords, cumu_kernel)
        grad_result = grad_objective(parameters, xCoords, cumu_kernel)[0]
        grad_norm = jnp.sqrt(jnp.sum(grad_result**2))
        grad_result = grad_result / grad_norm

        if i < 1000:
            factor = 0.01
        else:
            factor = 0.001

        step_size = jnp.sqrt(jnp.sum(parameters**2)) * factor
        parameters = parameters - grad_result * step_size

        if (loss < minimumLoss):
            minimumLoss = loss
            result = parameters

        if True and i % 300 == 0:
            print(parameters, '\n', grad_norm, '\n', loss, '\n')

    return result[0]*scale, result[1], result[2]*scale, result[3]


def objective_function(parameters, xCoords, cumu_kernel):
    A, a, B, b = parameters
    pred = A * (1-jnp.exp(-a*xCoords)) + \
        B * (1-jnp.exp(-b*xCoords))
    return jnp.sum((pred - cumu_kernel) ** 2)

grad_objective = jax.grad(objective_function, argnums=(0,))


def kernelShow():
    folder = "/data/qifan/projects/EndtoEnd/results/CCCSBench"
    nAngles = 8
    radiusDim = 100
    radiusRes = 0.05
    tabulatedKernelFile = os.path.join(folder,
        'tabulatedKernel_{}_{}.npy'.format(nAngles, radiusDim*2))
    
    Norm = False
    if Norm:
        exponentialKernelFile = os.path.join(folder, 'exponentialKernelNorm.npy')
    else:
        exponentialKernelFile = os.path.join(folder, 'exponentialKernel.npy')

    tabulatedKernel = np.load(tabulatedKernelFile)
    exponentialKernel = np.load(exponentialKernelFile)
    xCoords = (np.arange(radiusDim*2) + 0.5) * radiusRes

    if True:
        for i in range(nAngles):
            tabRow = tabulatedKernel[i, :]
            cumu = np.cumsum(tabRow)
            
            A, a, B, b = exponentialKernel[i, :]
            pred = A * (1 - np.exp(- a * xCoords)) + \
                B * (1 - np.exp(- b * xCoords))
            
            if Norm:
                cumu_initial = cumu[0]
                pred += cumu_initial

            plt.plot(xCoords, cumu)
            plt.plot(xCoords, pred)
            plt.xlabel('radius (cm)')
            plt.ylabel("cumulative kernel (a.u.)")
            plt.legend(["Monte carlo", "Exponential"])
            plt.title("cumulative kernel range: [{}pi/8, {}pi/8]".format(i, i+1))
            file = './figures/cumuFitting{}.png'.format(i)
            plt.savefig(file)
            plt.clf()
    if True:
        for i in range(nAngles):
            tabRow = tabulatedKernel[i, :]
            A, a, B, b = exponentialKernel[i, :]
            A *= a * radiusRes
            B *= b * radiusRes
            pred = A * np.exp(-a * xCoords) + B * np.exp(-b * xCoords)
            plt.plot(xCoords, tabRow)
            plt.plot(xCoords, pred)
            plt.xlabel('radius (cm)')
            plt.ylabel("collapsed-cone kernel (a.u.)")
            plt.legend(['Monte Carlo', 'Exponential'])
            plt.title("cumulative kernel range: [{}pi/8, {}pi/8]".format(i, i+1))
            file = './figures/kernelFitting{}.png'.format(i)
            plt.savefig(file)
            plt.clf()


def printKernel():
    folder = "/data/qifan/projects/EndtoEnd/results/CCCSBench"
    nAngles = 8
    radiusDim = 100
    radiusRes = 0.05
    tabulatedKernelFile = os.path.join(folder,
        'tabulatedKernel_{}_{}.npy'.format(nAngles, radiusDim*2))
    Norm = False
    if Norm:
        exponentialKernelFile = os.path.join(folder, 'exponentialKernelNorm.npy')
    else:
        exponentialKernelFile = os.path.join(folder, 'exponentialKernel.npy')
    exponentialKernel = np.load(exponentialKernelFile)
    xCoords = (np.arange(radiusDim*2) + 0.5) * radiusRes

    kernel_max = np.max(exponentialKernel)
    exponentialKernel[:, 0] /= kernel_max
    exponentialKernel[:, 2] /= kernel_max
    # print(exponentialKernel)

    # we take the first 6 kernel angles
    angles_selected = 6
    angle_range = np.pi / nAngles
    content = '{:^12}{:^12}{:^12}{:^12}{:^12}{:^12}'.format("angleBegin", "angleEnd", "A", "a", "B", "b")
    for i in range(angles_selected):
        angleStart = angle_range * i
        angleEnd = angle_range * (i + 1)
        line = '{:^12.4e}{:^12.4e}{:^12.4e}{:^12.4e}{:^12.4e}{:^12.4e}'.format(
            angleStart, angleEnd, *exponentialKernel[i, :])
        content = content + '\n' + line
    file = os.path.join(folder, 'kernel_exp_6mv.txt')
    with open(file, 'w') as f:
        f.write(content)
    

if __name__ == '__main__':
    # ViewKernelAngles()
    # Process()
    # kernelView()
    # kernelFitting()
    # kernelShow()
    printKernel()