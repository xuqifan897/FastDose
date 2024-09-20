# Fast Dose Calculation & Optimization
This project provides a comprehensive solution for accelerated external X-ray beam treatment planning, with a special focus on computationally intensive $4\pi$ treatment techniques. The core modules include:
1.	Preprocessing module,
2.	Dose calculation module,
3.	Treatment optimization module.

Additionally, users can access two optional modules:

4. A point dose kernel generation module, which produces analytical kernels used in the CCCS algorithm.
5. A slab phantom dose calculation module, which calculates the dose distribution for a user-defined phantom, serving as a benchmark against the CCCS-calculated dose.

## Quick Start
### Build the Docker Image
Assuming the project root is located at `user@workstation:/home/user/FastDose$`, you can build the Docker image by running:
```
user@workstation:/home/user/FastDose$ bash ./dockerBuild.sh
```
Docker typically requires sudo privileges. If you are able to bypass this, simply remove sudo from the bash script. The Dockerfile installs all dependencies by default, but if you don’t need modules 4 and 5 (which require Geant4), you can:
1. Comment out lines 20–48 in `./Dockerfile` (Geant4 installation).
2. Comment out lines 15–16 in `./CMakeLists.txt` to skip compiling modules 4 and 5.

### Download the Example Dataset
Download and extract the [dataset](https://zenodo.org/records/13800135) to a folder, e.g., `/data/user/dockerExample`. The dataset includes two cases: a pancreas cancer case and a head-and-neck cancer case. Input files for each case are:
1. `InputMask` — folder with linearized binary masks in `uint8`.
2. `structures.json` — lists relevant structures for dose calculation and optimization. Structures in `InputMask` not listed here will be ignored during preprocessing.
3. `density_raw.bin` — linearized density array in `uint16`.
4. `kernel_exp_6mv.txt` and `spec_6mv.spec` — parameters for the analytical CCCS kernel and the X-ray spectrum for a 6 MV beam.
5. `params.txt` and `StructureInfo.csv` — optimization parameters. In `params.txt`, `beamWeight` controls beam sparsity, with higher values reducing valid beams faster but risking early exclusions. `gamma` and `eta` affect the fluence map smoothness.

### Preprocessing
Using the head-and-neck case as an example, set the `SOURCE_FOLDER` environment variable and run:
```
user@workstation:/home/user/FastDose$ bash ./dockerScripts/HeadNeck/preprocess.sh
```
The `--mount` command maps the host directory `${SOURCE_FOLDER}` to `/data` inside the Docker container. To give the container write permissions, you can set permissions on the folder beforehand:
```
user@workstation:/home/user/FastDose$ chmod -R 777 [SOURCE_FOLDER]
```
Modify `[SOURCE_FOLDER]` with the full path. Arguments like `--ptv_name` and `__bbox_name` ensure that the structures are correctly contained in the input but do not directly affect results. Output files will be generated in `${INPUT_FOLDER}`.

To generate the beam list and directions for dose calculation, edit the Python file `/home/user/FastDose/dockerScripts/HeadNeck/scripts.py`. Uncomment the function `validAngleListGen()` and comment out the others. Modify the `sourceFolder` variable to point to the correct directory. Adjust the `numBeamsDesired` variable for more or fewer candidate beams (note that the final number of valid beams may differ slightly).

### Dose Calculation
After setting `${SOURCE_FOLDER}`, run:
```
user@workstation:/home/user/FastDose$ bash ./dockerScripts/HeadNeck/dosecalc.sh
```
For head-and-neck cases, we use a multi-isocenter treatment planning with four isocenters, requiring four iterations for dose calculation. Each computes the dose loading matrix for the beams associated with one isocenter. The `--primaryROI PTVSeg${segment}` argument specifies the PTV. For each beam, dose is only calculated for those beamlets which intersect with the PTV. The `--bbox_roi "SKIN"` argument specifies the bounding box of the valid region, beyond which either the density and the dose are ignored. `--deviceIdx` determines the GPU used for computation. Use `--mode 0` for dose calculation and `--mode 1` for treatment optimization.

### Treatment Plan Optimization
To Optimize the treatment plan, modify `${SOURCE_FOLDER}` and run:
```
user@workstation:/home/user/FastDose$ bash ./dockerScript/HeadNeck/optimize.sh
```
The program takes the four dose loading matrices (specified with `--outputFolder`) and saves the resulting plan to `${planFolder}`.

### Postprocessing and Visualization
In `/home/user/FastDose/dockerScripts/HeadNeck/scripts.py`, the functions `nrrdVerification()`, `DVHplot()`, and `doseWashPlot()` generate the beam view visualization file, dose volume histogram, and dose wash plot, respectively. The `.nrrd` file created by `nrrdVerification()` can be imported into [3D Slicer](https://www.slicer.org/) for visualization.

### Additional Information
For users interested in treatment optimization or developing novel algorithms, you can refer to `/home/user/FastDose/scripts/readDose.m` to load dose matrices into MATLAB.