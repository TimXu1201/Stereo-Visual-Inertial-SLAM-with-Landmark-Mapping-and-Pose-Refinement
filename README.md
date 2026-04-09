# Stereo Visual-Inertial SLAM with Landmark Mapping and Pose Refinement

This repository presents a **stereo visual-inertial SLAM** workflow that combines IMU prediction, stereo landmark triangulation, landmark-map construction, and pose refinement.

The implementation first predicts motion from IMU measurements, then recovers landmarks from stereo correspondences, and finally uses a fixed landmark map to improve the estimated trajectory through visual reprojection updates.

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
- `project_clarification.pdf`
  Project Description.

## Environment

Typical dependencies:

- Python 3.10+
- numpy
- scipy
- matplotlib
- transforms3d

## Notes

- Raw stereo and IMU datasets are not included in the public repository.
- Binary `.npy` intermediate outputs are excluded, while representative `.png` visualizations are kept.
- The current implementation expects a local folder layout for the dataset files before rerunning the full pipeline.
