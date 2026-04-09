from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from pr3_utils import (
    axangle2pose,
    inversePose,
    load_data,
    visualize_trajectory_2d,
)

# Basic stuff

def make_se3(xi):
    """Convert a 6D twist vector [v, w] to SE(3)."""
    return axangle2pose(np.asarray(xi, dtype=float)[None, :])[0]

def skew(v):
    x, y, z = v
    return np.array(
        [[0.0, -z, y],
         [z, 0.0, -x],
         [-y, x, 0.0]],
        dtype=float,
    )

def normalize_shapes(v_t, w_t, timestamps, features):
    """Handle either (T,3) or (3,T) inputs robustly."""
    v_t = np.asarray(v_t, dtype=float)
    w_t = np.asarray(w_t, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    features = np.asarray(features, dtype=float)

    if v_t.ndim == 2 and v_t.shape[0] == 3 and v_t.shape[1] != 3:
        v_t = v_t.T
    if w_t.ndim == 2 and w_t.shape[0] == 3 and w_t.shape[1] != 3:
        w_t = w_t.T

    # Expected final shape: (4, M, T)
    if features.ndim != 3:
        raise ValueError(f"Expected features with 3 dims, got {features.shape}")

    if features.shape[0] == 4:
        pass
    elif features.shape[1] == 4:
        features = np.transpose(features, (1, 0, 2))
    elif features.shape[2] == 4:
        features = np.transpose(features, (2, 0, 1))
    else:
        raise ValueError(f"Cannot infer feature layout from shape {features.shape}")

    return v_t, w_t, timestamps, features

REG2OPT = np.array(
    [
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [1.0,  0.0,  0.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ],
    dtype=float,
)


def get_camera_poses(world_T_imu, extL_T_imu, extR_T_imu):
    """
    extL_T_imu / extR_T_imu are ^I T_L and ^I T_R
    (regular camera frame -> IMU frame).

    Return:
        world_T_left_reg, world_T_right_reg,
        leftreg_T_world, rightreg_T_world, imu_T_world
    """
    # ^W T_L = ^W T_I * ^I T_L
    world_T_left_reg = world_T_imu @ extL_T_imu
    world_T_right_reg = world_T_imu @ extR_T_imu

    leftreg_T_world = inversePose(world_T_left_reg[None, ...])[0]
    rightreg_T_world = inversePose(world_T_right_reg[None, ...])[0]
    imu_T_world = inversePose(world_T_imu[None, ...])[0]

    return world_T_left_reg, world_T_right_reg, leftreg_T_world, rightreg_T_world, imu_T_world


def project_point(K, p_opt):
    """Project a point already expressed in the optical frame."""
    x, y, z = p_opt
    if z <= 1e-8:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.array([u, v], dtype=float)


def project_stereo(world_point, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu):
    """
    Project one world landmark into the left/right optical frames.
    """
    _, _, leftreg_T_world, rightreg_T_world, _ = get_camera_poses(
        world_T_imu, extL_T_imu, extR_T_imu
    )

    pw = np.hstack([world_point, 1.0])

    # world -> regular camera
    p_left_reg_h = leftreg_T_world @ pw
    p_right_reg_h = rightreg_T_world @ pw

    # regular -> optical
    p_left_opt = (REG2OPT @ p_left_reg_h)[:3]
    p_right_opt = (REG2OPT @ p_right_reg_h)[:3]

    zl = project_point(K_l, p_left_opt)
    zr = project_point(K_r, p_right_opt)

    if zl is None or zr is None:
        return None, None, None

    zhat = np.array([zl[0], zl[1], zr[0], zr[1]], dtype=float)
    return zhat, p_left_opt, None


def triangulate_landmark(z, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu):
    """
    Triangulate a world landmark from one stereo observation using DLT.
    """
    if np.any(z < 0):
        return None

    ul, vl, ur, vr = z

    _, _, leftreg_T_world, rightreg_T_world, _ = get_camera_poses(
        world_T_imu, extL_T_imu, extR_T_imu
    )

    # world -> optical
    leftopt_T_world = REG2OPT @ leftreg_T_world
    rightopt_T_world = REG2OPT @ rightreg_T_world

    P_l = K_l @ leftopt_T_world[:3, :]
    P_r = K_r @ rightopt_T_world[:3, :]

    A = np.vstack([
        ul * P_l[2] - P_l[0],
        vl * P_l[2] - P_l[1],
        ur * P_r[2] - P_r[0],
        vr * P_r[2] - P_r[1],
    ])

    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    X = vh[-1]
    if abs(X[3]) < 1e-10:
        return None

    world_point = X[:3] / X[3]

    zhat, _, _ = project_stereo(world_point, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
    if zhat is None or not np.all(np.isfinite(zhat)):
        return None

    return world_point



# Feature selection

def choose_feature_subset(features, max_features=None, min_obs=6):
    """
    Read the full feature tensor first, then rank features by:
      - total valid observations over the whole sequence
      - average disparity magnitude
    """
    valid = np.all(features >= 0, axis=0)  # (M, T)
    obs_count = valid.sum(axis=1)

    disparity = np.abs(features[0] - features[2])
    disparity = np.where(valid, disparity, 0.0)
    avg_disp = disparity.sum(axis=1) / np.maximum(obs_count, 1)

    score = obs_count * 100.0 + avg_disp

    good = np.where(obs_count >= min_obs)[0]
    if good.size == 0:
        good = np.arange(features.shape[1])

    good = good[np.argsort(-score[good])]

    if max_features is None or max_features >= good.size:
        return good.astype(int)

    return good[:max_features].astype(int)



# Numerical Jacobians

def pose_jacobian_numeric(world_point, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu, eps=1e-5):
    """
    Numerical Jacobian of stereo reprojection wrt pose update:
        world_T_imu <- world_T_imu @ Exp(delta)
    """
    z0, _, _ = project_stereo(world_point, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
    if z0 is None or not np.all(np.isfinite(z0)):
        return None

    H = np.zeros((4, 6), dtype=float)

    for k in range(6):
        d = np.zeros(6, dtype=float)
        d[k] = eps

        T_plus = world_T_imu @ make_se3(d)
        T_minus = world_T_imu @ make_se3(-d)

        z_plus, _, _ = project_stereo(world_point, T_plus, K_l, K_r, extL_T_imu, extR_T_imu)
        z_minus, _, _ = project_stereo(world_point, T_minus, K_l, K_r, extL_T_imu, extR_T_imu)

        if z_plus is None or z_minus is None:
            return None

        H[:, k] = (z_plus - z_minus) / (2.0 * eps)

    return H


def landmark_jacobian_numeric(world_point, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu, eps=1e-4):
    """
    Numerical Jacobian of stereo reprojection wrt landmark position.
    """
    z0, _, _ = project_stereo(world_point, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
    if z0 is None or not np.all(np.isfinite(z0)):
        return None

    H = np.zeros((4, 3), dtype=float)

    for k in range(3):
        p_plus = world_point.copy()
        p_minus = world_point.copy()
        p_plus[k] += eps
        p_minus[k] -= eps

        z_plus, _, _ = project_stereo(p_plus, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
        z_minus, _, _ = project_stereo(p_minus, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu)

        if z_plus is None or z_minus is None:
            return None

        H[:, k] = (z_plus - z_minus) / (2.0 * eps)

    return H



# Question 1: IMU localization via prediction

def imu_localization_prediction(v_t, w_t, timestamps, process_noise=None):
    T = len(timestamps)
    world_T_imu = np.zeros((T, 4, 4), dtype=float)
    world_T_imu[0] = np.eye(4)

    P_hist = np.zeros((T, 6, 6), dtype=float)
    P = 1e-6 * np.eye(6)

    if process_noise is None:
        process_noise = np.diag([0.02, 0.02, 0.02, 0.005, 0.005, 0.005])

    for k in range(T - 1):
        dt = float(timestamps[k + 1] - timestamps[k])
        xi = np.hstack([v_t[k], w_t[k]])

        world_T_imu[k + 1] = world_T_imu[k] @ make_se3(xi * dt)

        ad = np.block([
            [skew(w_t[k]), skew(v_t[k])],
            [np.zeros((3, 3)), skew(w_t[k])],
        ])
        F = expm(-ad * dt)
        P = F @ P @ F.T + process_noise * max(dt, 1e-3) ** 2
        P_hist[k + 1] = P

    return world_T_imu, P_hist



# Question 3: Landmark mapping

def landmark_mapping(
    world_T_imu_hist,
    features,
    K_l,
    K_r,
    extL_T_imu,
    extR_T_imu,
    max_features=None,
    min_obs=6,
    max_updates_per_step=200,
    meas_noise_px=3.0,
):
    feat_idx = choose_feature_subset(features, max_features=max_features, min_obs=min_obs)
    feats = features[:, feat_idx, :]
    M = feats.shape[1]
    T = feats.shape[2]

    mu = np.zeros((M, 3), dtype=float)
    Sigma = np.array([100.0 * np.eye(3) for _ in range(M)], dtype=float)
    initialized = np.zeros(M, dtype=bool)

    R = (meas_noise_px ** 2) * np.eye(4)

    for t in range(T):
        pose_t = world_T_imu_hist[t]
        visible = np.where(np.all(feats[:, :, t] >= 0, axis=0))[0]

        if visible.size > max_updates_per_step:
            pick = np.linspace(0, visible.size - 1, max_updates_per_step).astype(int)
            visible = visible[pick]

        for j in visible:
            z = feats[:, j, t]

            if not initialized[j]:
                X = triangulate_landmark(z, pose_t, K_l, K_r, extL_T_imu, extR_T_imu)
                if X is not None and np.all(np.isfinite(X)) and np.linalg.norm(X) < 500.0:
                    mu[j] = X
                    Sigma[j] = 5.0 * np.eye(3)
                    initialized[j] = True
                continue

            if not np.all(np.isfinite(mu[j])) or np.linalg.norm(mu[j]) > 500.0:
                initialized[j] = False
                mu[j] = 0.0
                Sigma[j] = 100.0 * np.eye(3)
                continue

            zhat, _, _ = project_stereo(mu[j], pose_t, K_l, K_r, extL_T_imu, extR_T_imu)
            if zhat is None or not np.all(np.isfinite(zhat)):
                continue

            innovation = z - zhat
            if np.linalg.norm(innovation) > 80.0:
                continue

            Hm = landmark_jacobian_numeric(mu[j], pose_t, K_l, K_r, extL_T_imu, extR_T_imu)
            if Hm is None:
                continue

            S = Hm @ Sigma[j] @ Hm.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue

            maha = float(innovation.T @ S_inv @ innovation)
            if maha > 25.0:
                continue

            K_gain = Sigma[j] @ Hm.T @ S_inv
            delta = K_gain @ innovation
            delta = np.clip(delta, -0.5, 0.5)

            mu[j] = mu[j] + delta

            I3 = np.eye(3)
            Sigma[j] = (I3 - K_gain @ Hm) @ Sigma[j] @ (I3 - K_gain @ Hm).T + K_gain @ R @ K_gain.T

            if not np.all(np.isfinite(mu[j])) or np.linalg.norm(mu[j]) > 500.0:
                initialized[j] = False
                mu[j] = 0.0
                Sigma[j] = 100.0 * np.eye(3)

    return mu, initialized, feat_idx



# Question 4: Stabilized visual-inertial SLAM
# fixed map from Question 3 + pose-only visual correction

def visual_inertial_slam_pose_only(
    v_t,
    w_t,
    timestamps,
    features,
    K_l,
    K_r,
    extL_T_imu,
    extR_T_imu,
    map_landmarks,
    map_mask,
    feat_idx,
    pose_process_noise=None,
    meas_noise_px=2.5,
    max_updates_per_step=200,
):
    feats = features[:, feat_idx, :]
    T = feats.shape[2]

    world_T_imu = np.eye(4)
    world_T_hist = np.zeros((T, 4, 4), dtype=float)
    world_T_hist[0] = world_T_imu

    P = 1e-6 * np.eye(6)

    if pose_process_noise is None:
        pose_process_noise = np.diag([0.03, 0.03, 0.03, 0.008, 0.008, 0.008])

    R = (meas_noise_px ** 2) * np.eye(4)

    for t in range(T - 1):
        dt = float(timestamps[t + 1] - timestamps[t])
        xi = np.hstack([v_t[t], w_t[t]])

        # prediction
        world_T_imu = world_T_imu @ make_se3(xi * dt)

        ad = np.block([
            [skew(w_t[t]), skew(v_t[t])],
            [np.zeros((3, 3)), skew(w_t[t])],
        ])
        F = expm(-ad * dt)
        P = F @ P @ F.T + pose_process_noise * max(dt, 1e-3) ** 2

        # update
        visible = np.where(np.all(feats[:, :, t + 1] >= 0, axis=0))[0]
        visible = [j for j in visible if map_mask[j]]

        if len(visible) > max_updates_per_step:
            pick = np.linspace(0, len(visible) - 1, max_updates_per_step).astype(int)
            visible = [visible[idx] for idx in pick]

        for j in visible:
            z = feats[:, j, t + 1]
            world_point = map_landmarks[j]

            if not np.all(np.isfinite(world_point)) or np.linalg.norm(world_point) > 500.0:
                continue

            zhat, _, _ = project_stereo(world_point, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
            if zhat is None or not np.all(np.isfinite(zhat)):
                continue

            innovation = z - zhat
            if np.linalg.norm(innovation) > 80.0:
                continue

            H = pose_jacobian_numeric(world_point, world_T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
            if H is None:
                continue

            S = H @ P @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue

            maha = float(innovation.T @ S_inv @ innovation)
            if maha > 20.0:
                continue

            K_gain = P @ H.T @ S_inv
            delta = K_gain @ innovation

            # damped / clipped pose update
            dpose = delta.copy()
            dpose[:3] = np.clip(0.25 * dpose[:3], -0.08, 0.08)
            dpose[3:] = np.clip(0.25 * dpose[3:], -0.02, 0.02)

            world_T_imu = world_T_imu @ make_se3(dpose)

            I6 = np.eye(6)
            P = (I6 - K_gain @ H) @ P @ (I6 - K_gain @ H).T + K_gain @ R @ K_gain.T

        world_T_hist[t + 1] = world_T_imu

    return world_T_hist



# Plot helper

def plot_landmarks_xy(landmarks, mask, title, out_file=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    if np.any(mask):
        pts = landmarks[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    if out_file is not None:
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_traj_and_landmarks(
    world_T_imu,
    landmarks,
    mask,
    title,
    out_file=None,
    max_dist_to_traj=35.0,
    max_norm=300.0,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    traj_xy = world_T_imu[:, :2, 3]
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "r-", lw=1.5, label="trajectory")
    ax.scatter(traj_xy[0, 0], traj_xy[0, 1], marker="s", s=50, label="start")
    ax.scatter(traj_xy[-1, 0], traj_xy[-1, 1], marker="o", s=50, label="end")

    if np.any(mask):
        pts = landmarks[mask]
        pts = pts[np.linalg.norm(pts[:, :2], axis=1) < max_norm]

        keep = []
        for p in pts[:, :2]:
            d = np.min(np.linalg.norm(traj_xy - p[None, :], axis=1))
            keep.append(d < max_dist_to_traj)
        keep = np.array(keep, dtype=bool)

        pts = pts[keep]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.45, label="landmarks")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()

    if out_file is not None:
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
    return fig, ax



# Runner

def run_one_dataset(dataset_name, data_root, out_root):
    filename = data_root / dataset_name / f"{dataset_name}.npy"
    if not filename.exists():
        print(f"[skip] {filename} not found")
        return

    print(f"\n===== Running {dataset_name} =====")
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(str(filename))
    v_t, w_t, timestamps, features = normalize_shapes(v_t, w_t, timestamps, features)

    ds_out = out_root / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)

    # Question 1
    print("[Q1] IMU localization...")
    world_T_imu_dr, _ = imu_localization_prediction(v_t, w_t, timestamps)

    fig, _ = visualize_trajectory_2d(
        world_T_imu_dr,
        path_name=f"{dataset_name} IMU",
        show_ori=True,
    )
    fig.savefig(ds_out / "part_a_imu_traj.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Question 3
    print("[Q3] Landmark mapping...")
    landmarks_map, map_mask, map_feat_idx = landmark_mapping(
        world_T_imu_dr,
        features,
        K_l,
        K_r,
        extL_T_imu,
        extR_T_imu,
        max_features=None,
        min_obs=6,
        max_updates_per_step=200,
        meas_noise_px=3.0,
    )

    fig, _ = plot_landmarks_xy(
        landmarks_map,
        map_mask,
        title=f"{dataset_name} - Question 3 landmarks",
        out_file=ds_out / "part_b_landmarks_xy.png",
    )
    plt.close(fig)

    # Question 4
    print("[Q4] Visual-inertial SLAM...")
    world_T_imu_slam = visual_inertial_slam_pose_only(
        v_t,
        w_t,
        timestamps,
        features,
        K_l,
        K_r,
        extL_T_imu,
        extR_T_imu,
        map_landmarks=landmarks_map,
        map_mask=map_mask,
        feat_idx=map_feat_idx,
        meas_noise_px=2.5,
        max_updates_per_step=200,
    )

    fig, _ = visualize_trajectory_2d(
        world_T_imu_slam,
        path_name=f"{dataset_name} SLAM",
        show_ori=True,
    )
    fig.savefig(ds_out / "part_c_slam_traj.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_traj_and_landmarks(
        world_T_imu_slam,
        landmarks_map,
        map_mask,
        title=f"{dataset_name} - Question 4 SLAM",
        out_file=ds_out / "part_c_slam_xy.png",
    )
    plt.close(fig)

    np.save(ds_out / "world_T_imu_q1.npy", world_T_imu_dr)
    np.save(ds_out / "world_T_imu_q4.npy", world_T_imu_slam)
    np.save(ds_out / "landmarks_q3.npy", landmarks_map)
    np.save(ds_out / "landmark_mask_q3.npy", map_mask)
    np.save(ds_out / "feature_idx_q3.npy", map_feat_idx)

    print(f"[done] outputs saved to: {ds_out}")


if __name__ == "__main__":
    code_dir = Path(__file__).resolve().parent
    data_root = code_dir.parent / "data"
    out_root = code_dir / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in ["dataset00", "dataset01"]:
        run_one_dataset(dataset_name, data_root, out_root)

    print("\nAll finished.")