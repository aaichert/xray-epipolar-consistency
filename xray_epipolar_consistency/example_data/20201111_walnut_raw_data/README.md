# Walnut Example Data

Please download the raw data from [Zenodo](https://zenodo.org/records/6986012) and place the .tif files in `example_data/20201111_walnut_raw_data/20201111_walnut_raw_data/` directory.

Here is what the tree of the example data should look like after decompression:

```
example_data/
├── 20201111_walnut_raw_data
│   ├── 20201111_walnut_raw_data
│   │   ├── 20201111_walnut_0001.tif
│   │   ├── 20201111_walnut_0002.tif
│   │   ├── 20201111_walnut_0003.tif
│   │   ├── ...
│   │   ├── 20201111_walnut_0720.tif
│   │   ├── 20201111_walnut_0721.tif
│   │   ├── 20201111_walnut_.txt
│   │   └── process.py
│   ├── README.md
│   ├── suggest_recon_volume.py
│   ├── trajectory_360_4x4.ompl
│   ├── trajectory_720_1x1.ompl
│   ├── trajectory_720_2x2.ompl
│   ├── trajectory_720_4x4.ompl
│   ├── trajectory_as_ompl.py
│   ├── walnut_18_4x4.json
│   ├── walnut_18_4x4.nrrd
│   ├── walnut_18_4x4.ompl
│   ├── walnut_360_4x4.json
│   ├── walnut_360_4x4.nrrd
│   ├── walnut_720_2x2.nrrd
│   ├── walnut_720_4x4.nrrd
│   └── walnut_reconstruction_360_4x4.nrrd
├── proj000.nrrd
├── proj040.nrrd
└── synthetic_pumpkin
    ├── fullscan_180views_600x400.7z
    ├── fullscan_180views_600x400.json
    ├── fullscan_180views_600x400.nrrd
    ├── fullscan_180views_600x400.ompl
    ├── fullscan_18views_600x400.json
    ├── fullscan_18views_600x400.nrrd
    ├── fullscan_18views_600x400.ompl
    ├── README.md
    └── reconstruction.nrrd

```
