# 3D Cone-Beam CT Reconstruction & Epipolar Consistency

This directory contains a python tool for 3D Cone-Beam CT (CBCT) reconstruction, with a tiny example data set.

## Project Structure & Reference Data

Please unzip the file `fullscan_180views_600x400.zip` before use.


- **`reconstruct.py`**: Python script for performing Feldkamp-Davis-Kress (FDK) cone-beam reconstruction using the ASTRA toolbox.
- **`fullscan_<N>views_<W>x<H>.nrrd`**: The raw X-ray projection stacks, Synthetic forward projections.
- **`*.ompl`**: Projection geometry matrices for the FD-CT views.
- **`*.json`**: Config file containing settings for reconstructing the example data.


## For more info

For the complete astra reconstruction example and instrcutions on how to create the example data please view this repo:

[GitHub Repo](https://github.com/aaichert/ct_recon_fdk_astra)
