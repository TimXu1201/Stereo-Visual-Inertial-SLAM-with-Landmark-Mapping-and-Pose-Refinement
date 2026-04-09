# Visual-Inertial SLAM Code Guide

This folder contains the implementation for stereo visual-inertial SLAM with IMU prediction, landmark mapping, and pose refinement.

## Dependencies

```bash
pip install numpy scipy matplotlib transforms3d
```

## Expected Data Layout

Place the local data in the following structure relative to `main.py`:

- `../data/`
  - `dataset00/dataset00.npy`
  - `dataset01/dataset01.npy`
- `./outputs/`
  Output folder for generated figures and local intermediate files

## Main Stages

- IMU-only localization
- landmark mapping from stereo features
- pose-only visual-inertial refinement with a fixed landmark map

## Run

```bash
python3 main.py
```

The script processes `dataset00` and `dataset01` automatically when the expected data files are available.
