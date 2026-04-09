# Stereo Visual-Inertial SLAM with Landmark Mapping and Pose Refinement

This repository contains my work on **stereo visual-inertial SLAM**, combining IMU prediction, stereo landmark triangulation, map construction, and pose refinement.

The pipeline first predicts motion from IMU measurements, then recovers landmarks from stereo correspondences, and finally uses a fixed landmark map to improve the estimated trajectory through visual reprojection updates.

## Project Highlights

- SE(3)-based IMU trajectory prediction
- robust handling of stereo feature tensor layouts
- stereo triangulation for landmark initialization
- landmark-map construction from repeated feature observations
- pose-only visual-inertial refinement using reprojection error
- trajectory and landmark visualization for multiple datasets

## Repository Structure

- `code/main.py`
  Main visual-inertial SLAM pipeline.
- `code/pr3_utils.py`
  Helper functions for data loading, pose operations, and visualization.
- `code/outputs/`
  Selected trajectory and landmark plots generated during development.
- `276A_project3.pdf`
  Project report.

## Environment

Typical dependencies:

- Python 3.10+
- numpy
- scipy
- matplotlib
- transforms3d

## Notes

- Raw course datasets under `data/` are excluded from the public repository.
- Binary `.npy` intermediate outputs are ignored, while representative `.png` visualizations are kept.
- The original code was developed with a local folder layout; place dataset files in the expected structure before rerunning the full pipeline.
