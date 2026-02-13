import os
import json
import math
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# =========================
# Utils
# =========================
def load_point_cloud_xy(ply_path: str) -> np.ndarray:
    assert os.path.exists(ply_path), f"PLY not found: {ply_path}"
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points).astype(np.float32)
    if pts.size == 0:
        raise RuntimeError("Point cloud is empty.")
    return pts[:, :2]  # (N,2)


def load_camera_centers_xy(transforms_path: str) -> np.ndarray:
    assert os.path.exists(transforms_path), f"JSON not found: {transforms_path}"
    with open(transforms_path, "r") as f:
        tfm = json.load(f)
    assert "frames" in tfm, "transforms json must have key 'frames'"
    cams = []
    for fr in tfm["frames"]:
        T = np.array(fr["transform_matrix"], dtype=np.float32)  # (4,4) c2w
        cams.append(T[:3, 3])
    cams = np.stack(cams, axis=0).astype(np.float32)  # (F,3)
    return cams[:, :2]  # (F,2)


def filter_pcd_by_cam_bbox(pts_xy: np.ndarray, cams_xy: np.ndarray, pad: float = 8.0):
    cam_min = cams_xy.min(axis=0)
    cam_max = cams_xy.max(axis=0)
    lo = cam_min - pad
    hi = cam_max + pad
    keep = (pts_xy[:, 0] >= lo[0]) & (pts_xy[:, 0] <= hi[0]) & \
           (pts_xy[:, 1] >= lo[1]) & (pts_xy[:, 1] <= hi[1])
    return pts_xy[keep], keep, lo, hi


def build_occupancy_grid(pts_xy: np.ndarray, cams_xy: np.ndarray, res=None, margin=0.5):
    all_xy = np.concatenate([pts_xy, cams_xy], axis=0)
    mn = all_xy.min(axis=0) - margin
    mx = all_xy.max(axis=0) + margin
    extent = mx - mn
    diag = float(np.linalg.norm(extent))

    if res is None:
        res = max(diag / 400.0, 0.02)

    W = int(math.ceil(extent[0] / res))
    H = int(math.ceil(extent[1] / res))
    W = max(W, 10)
    H = max(H, 10)

    def xy_to_ij(xy):
        ij = np.floor((xy - mn) / res).astype(np.int32)
        ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)  # x index
        ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)  # y index
        return ij

    ij_pts = xy_to_ij(pts_xy)
    occ = np.zeros((H, W), dtype=np.int32)
    np.add.at(occ, (ij_pts[:, 1], ij_pts[:, 0]), 1)

    ij_cams = xy_to_ij(cams_xy)
    meta = {"mn": mn, "mx": mx, "res": float(res), "W": W, "H": H, "diag": diag}
    return occ, ij_cams, meta


def grid_to_xy(ix, iy, meta):
    mn = meta["mn"]
    res = meta["res"]
    x = mn[0] + (ix + 0.5) * res
    y = mn[1] + (iy + 0.5) * res
    return x, y


def boundary_mask_from_occ(occ: np.ndarray, q: float = 85.0):
    v = occ[occ > 0]
    if len(v) == 0:
        return np.zeros_like(occ, dtype=bool), 1
    thr = int(np.percentile(v, q))
    thr = max(thr, 1)
    return (occ >= thr), thr


def ray_outer_boundary_from_boundary_mask(boundary_mask: np.ndarray,
                                         ij_cams: np.ndarray,
                                         angle_bins: int = 360,
                                         min_count_per_bin: int = 1):
    ys, xs = np.where(boundary_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32), None

    cxy = ij_cams.mean(axis=0).astype(np.float32)  # [x,y] in grid
    dx = xs.astype(np.float32) - cxy[0]
    dy = ys.astype(np.float32) - cxy[1]
    r = np.sqrt(dx * dx + dy * dy) + 1e-8
    theta = np.arctan2(dy, dx)  # [-pi, pi]
    theta = (theta + 2*np.pi) % (2*np.pi)  # [0, 2pi)

    bin_id = np.floor(theta / (2*np.pi) * angle_bins).astype(np.int32)
    bin_id = np.clip(bin_id, 0, angle_bins - 1)

    best_r = np.full((angle_bins,), -1.0, dtype=np.float32)
    best_xy = np.full((angle_bins, 2), -1, dtype=np.int32)
    bin_cnt = np.zeros((angle_bins,), dtype=np.int32)

    for i in range(len(xs)):
        b = bin_id[i]
        bin_cnt[b] += 1
        if r[i] > best_r[b]:
            best_r[b] = r[i]
            best_xy[b, 0] = xs[i]
            best_xy[b, 1] = ys[i]

    valid = (best_r > 0) & (bin_cnt >= min_count_per_bin)
    boundary_ij = best_xy[valid]
    return boundary_ij.astype(np.int32), cxy


def sample_anchors_from_boundary(boundary_ij: np.ndarray, K: int = 32):
    if boundary_ij.shape[0] == 0:
        return boundary_ij
    if boundary_ij.shape[0] <= K:
        return boundary_ij
    idx = np.linspace(0, boundary_ij.shape[0] - 1, K).astype(np.int32)
    return boundary_ij[idx]


def fallback_sample_from_boundary_mask(boundary_mask: np.ndarray,
                                      occ: np.ndarray,
                                      ij_cams: np.ndarray,
                                      K: int = 32):
    ys, xs = np.where(boundary_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    cxy = ij_cams.mean(axis=0).astype(np.float32)
    dx = xs.astype(np.float32) - cxy[0]
    dy = ys.astype(np.float32) - cxy[1]
    r = np.sqrt(dx * dx + dy * dy) + 1e-8
    s = occ[ys, xs].astype(np.float32)

    score = s + 0.5 * r
    order = np.argsort(-score)

    picked = []
    for idx in order:
        picked.append([xs[idx], ys[idx]])
        if len(picked) >= K:
            break
    return np.array(picked, dtype=np.int32)


def save_scene_figure(fig_path: str,
                      pts_xy: np.ndarray,
                      cams_xy: np.ndarray,
                      anchors_world_xy: np.ndarray,
                      title: str = ""):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.scatter(pts_xy[:, 0], pts_xy[:, 1], s=1, alpha=0.10, label="pcd filtered")
    plt.plot(cams_xy[:, 0], cams_xy[:, 1], "-k", linewidth=2, label="cam path")
    if anchors_world_xy.shape[0] > 0:
        plt.scatter(anchors_world_xy[:, 0], anchors_world_xy[:, 1],
                    s=60, c="cyan", edgecolors="k",
                    label=f"boundary anchors K={anchors_world_xy.shape[0]}")
    plt.axis("equal")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


# =========================
# Per-scene processing
# =========================
def process_scene(
    scene_id: int,
    base_root: str,
    K_ANCHORS: int = 32,
    CAM_PAD: float = 8.0,
    RES=None,
    MARGIN: float = 0.5,
    BOUNDARY_Q: float = 85.0,
    ANGLE_BINS: int = 360,
    save_fig: bool = True,
):
    data_root = os.path.join(base_root, str(scene_id))
    ply_path = os.path.join(data_root, "points3D.ply")
    transforms_path = os.path.join(data_root, "transforms_train.json")

    out_pkl_path = os.path.join(data_root, f"{K_ANCHORS}_boundary_points.pkl")
    out_fig_path = os.path.join(data_root, f"{K_ANCHORS}_boundary_anchors.png")

    if not (os.path.exists(ply_path) and os.path.exists(transforms_path)):
        print(f"[Scene {scene_id:02d}] skip (missing files)")
        return False

    # 1) load
    pts_xy_raw = load_point_cloud_xy(ply_path)
    cams_xy = load_camera_centers_xy(transforms_path)

    # 2) outlier removal by cam bbox + pad
    pts_xy, keep, lo, hi = filter_pcd_by_cam_bbox(pts_xy_raw, cams_xy, pad=CAM_PAD)
    if pts_xy.shape[0] == 0:
        pts_xy, keep, lo, hi = filter_pcd_by_cam_bbox(pts_xy_raw, cams_xy, pad=CAM_PAD * 2.0)
        if pts_xy.shape[0] == 0:
            print(f"[Scene {scene_id:02d}] FAIL: filtered PCD empty even with pad*2")
            return False

    # 3) occupancy grid
    occ, ij_cams, meta = build_occupancy_grid(pts_xy, cams_xy, res=RES, margin=MARGIN)

    # 4) boundary mask from top percentile occupancy
    boundary_mask, thr = boundary_mask_from_occ(occ, q=BOUNDARY_Q)

    # 5) ray outer boundary
    boundary_ij, center_ij = ray_outer_boundary_from_boundary_mask(
        boundary_mask, ij_cams, angle_bins=ANGLE_BINS, min_count_per_bin=1
    )

    # 6) sample anchors
    anchors_ij = sample_anchors_from_boundary(boundary_ij, K=K_ANCHORS)

    # 7) fallback if too few
    if anchors_ij.shape[0] < K_ANCHORS:
        anchors_ij_fb = fallback_sample_from_boundary_mask(boundary_mask, occ, ij_cams, K=K_ANCHORS)
        if anchors_ij_fb.shape[0] >= K_ANCHORS:
            anchors_ij = anchors_ij_fb
        if anchors_ij.shape[0] == 0:
            print(f"[Scene {scene_id:02d}] FAIL: no boundary anchors found (try lower BOUNDARY_Q)")
            return False

    # 8) grid -> world xy -> xyz (z=0)
    anchors_world_xy = np.array([grid_to_xy(x, y, meta) for x, y in anchors_ij], dtype=np.float32)
    if anchors_world_xy.shape[0] > K_ANCHORS:
        anchors_world_xy = anchors_world_xy[:K_ANCHORS]
    anchors_world_xyz = np.concatenate(
        [anchors_world_xy, np.zeros((anchors_world_xy.shape[0], 1), dtype=np.float32)], axis=1
    )  # (K,3)

    # 9) save pkl
    points_dict = {
        "points": anchors_world_xyz.astype(np.float32)[:, :2], # (K,2)
        "colors": np.zeros((anchors_world_xyz.shape[0], 3), dtype=np.float32),  # placeholder
    }
    with open(out_pkl_path, "wb") as f:
        pickle.dump(points_dict, f)

    # 10) save figure
    if save_fig:
        save_scene_figure(
            fig_path=out_fig_path,
            pts_xy=pts_xy,
            cams_xy=cams_xy,
            anchors_world_xy=anchors_world_xy,
            title=f"Scene {scene_id}: {K_ANCHORS} boundary anchors (top-down xy)"
        )

    print(f"[Scene {scene_id:02d}] saved -> {out_pkl_path} | anchors={anchors_world_xyz.shape[0]} "
          f"| kept_pcd={pts_xy.shape[0]}/{pts_xy_raw.shape[0]} | BOUNDARY_Q={BOUNDARY_Q}, thr={thr}"
          + (f" | fig={out_fig_path}" if save_fig else ""))

    return True


# =========================
# Main: generate for 13 scenes
# =========================
def main():
    base_root = "../data_RWAVS"
    scene_ids = list(range(1, 14))  # 1..13

    # knobs
    K_ANCHORS = 32
    CAM_PAD = 8.0
    RES = None
    MARGIN = 0.5
    BOUNDARY_Q = 85.0
    ANGLE_BINS = 360

    ok = 0
    for sid in scene_ids:
        success = process_scene(
            scene_id=sid,
            base_root=base_root,
            K_ANCHORS=K_ANCHORS,
            CAM_PAD=CAM_PAD,
            RES=RES,
            MARGIN=MARGIN,
            BOUNDARY_Q=BOUNDARY_Q,
            ANGLE_BINS=ANGLE_BINS,
            save_fig=False,
        )
        ok += int(success)

    print(f"\nDone. success={ok}/{len(scene_ids)}")


if __name__ == "__main__":
    main()


"""
pkl_path = os.path.join(data_root, "32_boundary_points.pkl")
with open(pkl_path, "rb") as f:
    d = pickle.load(f)
boundary_xy = np.asarray(d["points"]).astype(np.float32)  # (K,2)
boundary_rgb = np.asarray(d["colors"]).astype(np.float32)  # (K,3) placeholder
"""
