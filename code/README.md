# ECE276A Project 3: Visual-Inertial SLAM

# Dependencies

pip install numpy scipy matplotlib transforms3d

# Directory Structure

Ensure your files are organized as follows relative to the script (`main.py`):

- `../data/`
  - `dataset00/dataset00.npy`
  - `dataset01/dataset01.npy`
- `./outputs/` (Auto-generated folder where all output figures and arrays will be saved)

# Implemented Parts

This project includes:

- Q1: IMU-only localization
- Q3: Landmark mapping from stereo features
- Q4: Pose-only visual-inertial SLAM with a fixed landmark map

# Input Data

Each dataset `.npy` file should contain:

- `v_t`: linear velocity in IMU frame
- `w_t`: angular velocity in IMU frame
- `timestamps`: timestamps
- `features`: stereo feature coordinates
- `K_l`: left camera intrinsic matrix
- `K_r`: right camera intrinsic matrix
- `extL_T_imu`: left camera extrinsic matrix
- `extR_T_imu`: right camera extrinsic matrix

# Run

Run the main script:

python3 main.py

The script will automatically process:

- `dataset00`
- `dataset01`

# Q1: IMU Localization

This part:

- starts from the identity pose
- integrates IMU linear and angular velocity
- generates the IMU-only trajectory

Saved output:

- `part_a_imu_traj.png`
- `world_T_imu_q1.npy`

# Q3: Landmark Mapping

This part:

- selects valid stereo features
- triangulates 3D landmarks
- updates landmark positions
- removes unstable landmarks

Saved output:

- `part_b_landmarks_xy.png`
- `landmarks_q3.npy`
- `landmark_mask_q3.npy`
- `feature_idx_q3.npy`

# Q4: Visual-Inertial SLAM

This part:

- uses IMU motion for prediction
- uses the Q3 landmark map as a fixed map
- applies pose-only visual correction using reprojection error

Saved output:

- `part_c_slam_traj.png`
- `part_c_slam_xy.png`
- `world_T_imu_q4.npy`

# Output Directory

Results are saved in:

- `outputs/dataset00/`
- `outputs/dataset01/`

# Utility File

`pr3_utils.py` provides helper functions for:

- loading dataset files
- SE(3) pose operations
- inverse pose computation
- projection utilities
- 2D trajectory visualization
