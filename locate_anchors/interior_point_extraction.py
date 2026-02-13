import os
import pickle
import numpy as np
from plyfile import PlyData
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# =========================
# Outlier removal (isolated points)
# =========================
def remove_sparse_outliers(xyz: np.ndarray,
                           rgb: np.ndarray,
                           nrm: np.ndarray,
                           nb_neighbors: int = 20,
                           std_ratio: float = 2.0):
    """
    Remove isolated sparse outliers via kNN distance statistics:
    - compute mean distance to k nearest neighbors for each point
    - keep points with mean_dist <= mean + std_ratio * std
    """
    if xyz.shape[0] < nb_neighbors + 2:
        return xyz, rgb, nrm

    nn = NearestNeighbors(n_neighbors=nb_neighbors + 1, algorithm="auto")
    nn.fit(xyz)
    dists, _ = nn.kneighbors(xyz, return_distance=True)  # (N, k+1), first is 0 (self)
    mean_d = dists[:, 1:].mean(axis=1)

    mu = float(mean_d.mean())
    sig = float(mean_d.std() + 1e-12)
    thr = mu + std_ratio * sig
    keep = mean_d <= thr

    xyz2 = xyz[keep]
    rgb2 = rgb[keep] if rgb is not None else None
    nrm2 = nrm[keep] if nrm is not None else None
    return xyz2, rgb2, nrm2


# =========================
# Grid + attribute averaging
# =========================
def round_to_resolution(xyz: np.ndarray, resolution: float) -> np.ndarray:
    return np.round(xyz / resolution) * resolution


def average_attributes(rounded_xyz: np.ndarray, attr: np.ndarray):
    if rounded_xyz.shape[0] == 0:
        return rounded_xyz, attr

    unique_xyz, inv = np.unique(rounded_xyz, axis=0, return_inverse=True)
    M = unique_xyz.shape[0]
    C = attr.shape[1]

    sums = np.zeros((M, C), dtype=np.float32)
    cnt = np.zeros((M,), dtype=np.int32)
    np.add.at(sums, inv, attr.astype(np.float32))
    np.add.at(cnt, inv, 1)
    avg = sums / np.maximum(cnt[:, None], 1)
    return unique_xyz.astype(np.float32), avg.astype(np.float32)


def get_grid(positions, colors, normals, align_grids=None, n_clusters=256, resolution=0.25, k=50):
    # quantize to grid
    rounded_positions = round_to_resolution(positions, resolution)

    # average attributes per unique grid point
    unique_positions, averaged_colors = average_attributes(rounded_positions, colors)
    _, averaged_normals = average_attributes(rounded_positions, normals)

    positions = unique_positions
    colors = averaged_colors
    normals = averaged_normals

    if positions.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3 * k), dtype=np.float32),
            np.zeros((0, 3 * k), dtype=np.float32),
        )

    if align_grids is not None:
        align_grids = align_grids.data.cpu().numpy()
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        kmeans.fit(positions)
        align_grids = np.array(kmeans.cluster_centers_, dtype=np.float32)

    closest_colors = np.zeros((align_grids.shape[0], k, 3), dtype=np.float32)
    closest_normals = np.zeros((align_grids.shape[0], k, 3), dtype=np.float32)

    for i, center in enumerate(align_grids):
        d = np.linalg.norm(positions - center[None, :], axis=1)
        idx = np.argsort(d)[:k]
        if idx.shape[0] < k:
            pad = np.full((k - idx.shape[0],), idx[-1], dtype=idx.dtype)
            idx = np.concatenate([idx, pad], axis=0)
        closest_colors[i] = colors[idx]
        closest_normals[i] = normals[idx]

    return (
        align_grids.astype(np.float32),
        closest_colors.reshape(align_grids.shape[0], 3 * k).astype(np.float32),
        closest_normals.reshape(align_grids.shape[0], 3 * k).astype(np.float32),
    )


# =========================
# I/O + visualization
# =========================
def load_ply_xyz_rgb(path: str):
    assert os.path.exists(path), f"PLY not found: {path}"
    plydata = PlyData.read(path)
    vertices = plydata["vertex"].data

    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)

    if all(k in vertices.dtype.names for k in ["red", "green", "blue"]):
        rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0
    else:
        rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    nrm = np.zeros_like(xyz, dtype=np.float32)
    return xyz, rgb, nrm


def save_anchors_3d_png(fig_path: str, anchors_xyz: np.ndarray, pcd_xyz: np.ndarray = None, title: str = ""):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    if pcd_xyz is not None and pcd_xyz.shape[0] > 0:
        n = pcd_xyz.shape[0]
        m = min(n, 30000)
        idx = np.random.RandomState(0).choice(n, size=m, replace=False)
        p = pcd_xyz[idx]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, alpha=0.03)

    if anchors_xyz.shape[0] > 0:
        ax.scatter(anchors_xyz[:, 0], anchors_xyz[:, 1], anchors_xyz[:, 2], s=18, alpha=0.9)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)

    # equal-ish aspect
    xyz = anchors_xyz if anchors_xyz.shape[0] > 0 else pcd_xyz
    if xyz is not None and xyz.shape[0] > 0:
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = np.max(maxs - mins) / 2.0 + 1e-6
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)


# =========================
# Per-scene processing
# =========================
def process_scene_interior(
    scene_id: int,
    base_root: str,
    N_POINTS: int = 256,
    RESOLUTION: float = 0.25,
    K_NEIGHBORS: int = 50,
    # outlier knobs
    OUTLIER_NB: int = 20,
    OUTLIER_STD: float = 2.0,
    save_fig: bool = True,
):
    data_root = os.path.join(base_root, str(scene_id))
    ply_path = os.path.join(data_root, "points3D.ply")
    out_pkl_path = os.path.join(data_root, f"{N_POINTS}_interior_points.pkl")
    out_fig_path = os.path.join(data_root, f"{N_POINTS}_interior_anchors_3d.png")

    if not os.path.exists(ply_path):
        print(f"[Scene {scene_id:02d}] skip (missing): {ply_path}")
        return False

    xyz, rgb, nrm = load_ply_xyz_rgb(ply_path)
    if xyz.shape[0] == 0:
        print(f"[Scene {scene_id:02d}] FAIL: empty point cloud")
        return False

    # 0) remove sparse outliers BEFORE grid/kmeans
    xyz_f, rgb_f, nrm_f = remove_sparse_outliers(
        xyz, rgb, nrm, nb_neighbors=OUTLIER_NB, std_ratio=OUTLIER_STD
    )

    if xyz_f.shape[0] < max(500, N_POINTS * 2):
        # avoid over-filtering in extremely sparse scenes
        xyz_f, rgb_f, nrm_f = xyz, rgb, nrm

    points, colors_feat, normals_feat = get_grid(
        positions=xyz_f,
        colors=rgb_f,
        normals=nrm_f,
        align_grids=None,
        n_clusters=N_POINTS,
        resolution=RESOLUTION,
        k=K_NEIGHBORS,
    )

    points_dict = {
        "points": points.astype(np.float32),          # (256,3)
        "colors": colors_feat.astype(np.float32),     # (256, 3*k)
        "normals": normals_feat.astype(np.float32),   # (256, 3*k)
    }
    with open(out_pkl_path, "wb") as f:
        pickle.dump(points_dict, f)

    if save_fig:
        save_anchors_3d_png(
            fig_path=out_fig_path,
            anchors_xyz=points,
            pcd_xyz=xyz_f,  # show filtered pcd context
            title=f"Scene {scene_id}: {N_POINTS} interior anchors (3D)"
        )

    print(f"[Scene {scene_id:02d}] saved -> {out_pkl_path} | fig={out_fig_path if save_fig else 'None'} "
          f"| pcd={xyz.shape[0]} -> {xyz_f.shape[0]} | resolution={RESOLUTION} | k={K_NEIGHBORS} "
          f"| outlier(nb={OUTLIER_NB}, std={OUTLIER_STD})")
    return True


# =========================
# Main
# =========================
def main():
    base_root = "../data_RWAVS"
    scene_ids = list(range(1, 14))  # 1..13

    N_POINTS = 256
    RESOLUTION = 0.25
    K_NEIGHBORS = 50

    # outlier knobs (tune if needed)
    OUTLIER_NB = 20
    OUTLIER_STD = 2.0

    ok = 0
    for sid in scene_ids:
        ok += int(process_scene_interior(
            scene_id=sid,
            base_root=base_root,
            N_POINTS=N_POINTS,
            RESOLUTION=RESOLUTION,
            K_NEIGHBORS=K_NEIGHBORS,
            OUTLIER_NB=OUTLIER_NB,
            OUTLIER_STD=OUTLIER_STD,
            save_fig=False,
        ))
    print(f"\nDone. success={ok}/{len(scene_ids)}")


if __name__ == "__main__":
    main()
