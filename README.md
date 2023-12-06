# Fast dose calculation
This project aims to develop a library for fast radiation dose calculation, with the potential to enable on-the-fly dose calculation in radiotherapy treatment plan optimization. This capability can alleviate memory constraints and allow for a larger search space with more degrees of freedom. The algorithm is based on the Collapsed-Cone Convolution/Superposition (CCCS) algorithm proposed by Weiguo Lu et al. \[[1]\]\[[2]\].

[1]:http://dx.doi.org/10.1088/0031-9155/50/4/007
[2]:http://dx.doi.org/10.1118/1.3551996

## Developer note
There are two coordinate systems:
1. patient volume coordinate systems (PVCS)
2. beam's eye view (BEV)

Below is the dose calculation process.
1. A density volume (BEV) is sampled from the PVCS to BEV.
2. Terma (BEV) is calculated in the density volume.
3. Dose (BEV) is calculated from the Terma volume.
4. Dose (PVCS) is accumulated by summing over the contributions from all beams.

Gradient calculation:

5. Dose gradient (PVCS) is calculation in the patient frame.
6. Dose gradient (BEV) is sampled from the PVCS frame to the BEV frame.
7. Terma gradient (BEV) is calculated from the Dose gradient.
8. The gradient for the fluence map is calculated.

As such, we have the following objects:
1. Fluence map per beam (2d array).
2. The density volume (3d texture).
3. Terma BEV per beam (3d array).
4. Dose BEV per beam (3d surface).
5. Dose PVCS (3d array).
6. Dose grad PVCS (3d surface).
7. Dose grad BEV (3d array).
8. Terma grad BEV (3d array).
9. Fluence grad (23 array).