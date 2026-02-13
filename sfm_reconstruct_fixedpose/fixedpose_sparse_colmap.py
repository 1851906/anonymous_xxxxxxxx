"""
Minimal fixed-pose sparse point cloud reconstruction with COLMAP

- Use transforms_train.json (NeRF-style) as FIXED camera poses
- Optionally take a subset of frames (uniform/random) like your RWAVSDataset
- Run COLMAP:
  feature_extractor -> matcher -> point_triangulator (fixed poses) -> export PLY
- (NEW) Optionally copy the exported points3D.ply into a target scene folder.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np


def run(cmd, cwd=None):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def qvec_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix (world-to-camera) to COLMAP qvec = [qw, qx, qy, qz]."""
    R = R.astype(np.float64)
    q = np.empty(4, dtype=np.float64)

    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

    q /= (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q *= -1
    return q


def ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink() and (not dst.exists()):
        dst.unlink()

    if dst.exists() or dst.is_symlink():
        return

    os.symlink(str(src.resolve()), str(dst))


def ratio_to_dirname(r: float) -> str:
    s = f"{r:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def parse_img_idx(file_path: str) -> int:
    """Extract integer index from file name, e.g. frames/00012.png -> 12"""
    name = os.path.basename(file_path)
    stem = os.path.splitext(name)[0]
    digits = "".join([c for c in stem if c.isdigit()])
    return int(digits) if digits else 0


def subset_frames(frames: List[Dict], subset_ratio: float, subset_strategy: str, seed: int) -> List[Dict]:
    if not (0 < subset_ratio <= 1.0):
        raise ValueError("subset_ratio must be in (0, 1].")
    n_total = len(frames)
    if n_total == 0:
        raise RuntimeError("No frames in transforms json.")
    if subset_ratio >= 1.0:
        print(f"[subset] keep all: {n_total}/{n_total} (100%)")
        return frames

    n_keep = max(1, int(round(n_total * subset_ratio)))

    if subset_strategy == "uniform":
        times = np.array([parse_img_idx(fr["file_path"]) for fr in frames])
        order = np.argsort(times)
        pos = np.round(np.linspace(0, n_total - 1, n_keep)).astype(int)
        picked = order[pos].tolist()
    elif subset_strategy == "random":
        rng = np.random.default_rng(seed)
        picked = rng.choice(np.arange(n_total), size=n_keep, replace=False)
        picked = np.sort(picked).tolist()
    else:
        raise ValueError("subset_strategy must be 'uniform' or 'random'.")

    out = [frames[i] for i in picked]
    print(f"[subset] {subset_strategy}: keep {len(out)}/{n_total} ({subset_ratio*100:.1f}%)")
    return out


def build_subset_and_model(
    json_path: Path,
    images_root: Path,
    out_dir: Path,
    opengl_corr: bool,
    subset_ratio: float,
    subset_strategy: str,
    seed: int,
) -> Tuple[Path, Path]:
    """
    - Create subset image dir (symlinks) for selected frames
    - Create COLMAP text model (cameras.txt, images.txt, points3D.txt) using FIXED poses
    Returns: (subset_images_dir, model_txt_dir)
    """
    meta = json.loads(json_path.read_text())

    # intrinsics
    W = int(round(float(meta["w"])))
    H = int(round(float(meta["h"])))
    fx = float(meta["fl_x"])
    fy = float(meta.get("fl_y", fx))
    cx = float(meta.get("cx", W / 2.0))
    cy = float(meta.get("cy", H / 2.0))

    frames_all = meta["frames"]
    frames = subset_frames(frames_all, subset_ratio, subset_strategy, seed)

    subset_dir = out_dir / "images_train"
    model_txt = out_dir / "model_txt"
    subset_dir.mkdir(parents=True, exist_ok=True)
    model_txt.mkdir(parents=True, exist_ok=True)

    # clean subset_dir (avoid mixing old/new subsets)
    for p in subset_dir.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

    # link images
    missing = 0
    kept = []
    for fr in frames:
        fp = fr["file_path"]          # e.g. "frames/00001.png" or "00001.png"
        name = Path(fp).name          # "00001.png"
        src = images_root / name
        if not src.exists():
            src = images_root / fp
        if not src.exists():
            missing += 1
            continue
        dst = subset_dir / Path(src).name
        ensure_symlink(src, dst)
        kept.append((fr, Path(src).name))

    if missing > 0:
        print(f"[WARN] {missing} images listed in JSON were not found under {images_root}.")

    print(f"[OK] subset images: {subset_dir}  (count={len(kept)})")

    # cameras.txt
    with (model_txt / "cameras.txt").open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # OpenGL->OpenCV correction (optional)
    S = np.eye(4, dtype=np.float64)
    if opengl_corr:
        S[1, 1] = -1.0
        S[2, 2] = -1.0

    # images.txt
    with (model_txt / "images.txt").open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        img_id = 1
        for fr, name in kept:
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            if c2w.shape != (4, 4):
                raise RuntimeError(f"transform_matrix shape {c2w.shape} != (4,4) for {name}")

            c2w_corr = c2w @ S
            w2c = np.linalg.inv(c2w_corr)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = qvec_from_rotmat(R)

            f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
            img_id += 1

    # points3D.txt (empty)
    with (model_txt / "points3D.txt").open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")

    print(f"[OK] model txt: {model_txt}")
    return subset_dir, model_txt


def copy_ply_to_scene(ply_src: Path, scene_dir: Path, out_name: str, overwrite: bool):
    scene_dir.mkdir(parents=True, exist_ok=True)
    dst = scene_dir / out_name
    if dst.exists() and (not overwrite):
        print(f"[SKIP] copy target exists (no overwrite): {dst}")
        return
    shutil.copy2(ply_src, dst)
    print(f"[OK] copied PLY -> {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=str, help="path to transforms_train.json")
    parser.add_argument("--images-root", required=True, type=str, help="path to frames/ dir containing 00001.png ...")
    parser.add_argument("--out", required=True, type=str, help="output root dir, e.g. colmap_fixed/room7")

    # subset controls
    parser.add_argument("--subset-ratio", type=float, default=1.0, help="in (0,1], e.g. 0.1")
    parser.add_argument("--subset-strategy", type=str, default="uniform", choices=["uniform", "random"])
    parser.add_argument("--seed", type=int, default=42)

    # matcher
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--overlap", type=int, default=10, help="for sequential_matcher: match with neighbor frames")
    parser.add_argument("--use-gpu", type=int, default=1)

    # coords correction
    parser.add_argument("--no-opengl-corr", action="store_true", help="disable OpenGL->OpenCV axis flip")

    # (NEW) copy PLY into scene folder
    parser.add_argument("--copy-ply-to", type=str, default="", help="if set, copy exported PLY into this directory")
    parser.add_argument("--copy-ply-name", type=str, default="points3D_fixedpose.ply",
                        help="file name when copying to scene folder (default avoids overwrite)")
    parser.add_argument("--copy-overwrite", action="store_true", help="overwrite if target exists")

    args = parser.parse_args()

    json_path = Path(args.json)
    images_root = Path(args.images_root)
    out_root = Path(args.out)

    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not images_root.exists():
        raise FileNotFoundError(images_root)

    # output folder: out_root/<ratio>/
    ratio_dir = ratio_to_dirname(float(args.subset_ratio))
    out_dir = out_root / ratio_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "database.db"
    sparse_out = out_dir / "sparse_fixed"
    ply_out = out_dir / "points3D.ply"

    # clean per-ratio outputs
    if db_path.exists():
        db_path.unlink()
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    if ply_out.exists():
        ply_out.unlink()

    subset_dir, model_txt = build_subset_and_model(
        json_path=json_path,
        images_root=images_root,
        out_dir=out_dir,
        opengl_corr=(not args.no_opengl_corr),
        subset_ratio=float(args.subset_ratio),
        subset_strategy=args.subset_strategy,
        seed=int(args.seed),
    )

    # intrinsics again for feature_extractor
    meta = json.loads(json_path.read_text())
    W = int(round(float(meta["w"])))
    H = int(round(float(meta["h"])))
    fx = float(meta["fl_x"])
    fy = float(meta.get("fl_y", fx))
    cx = float(meta.get("cx", W / 2.0))
    cy = float(meta.get("cy", H / 2.0))
    intrinsics = f"{fx},{fy},{cx},{cy}"

    # feature extraction
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(subset_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", intrinsics,
        "--SiftExtraction.use_gpu", str(args.use_gpu),
    ])

    # matching
    if args.matcher == "exhaustive":
        run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", str(args.use_gpu),
        ])
    else:
        run([
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SequentialMatching.overlap", str(args.overlap),
            "--SiftMatching.use_gpu", str(args.use_gpu),
        ])

    # triangulate (fixed poses)
    sparse_out.mkdir(parents=True, exist_ok=True)
    run([
        "colmap", "point_triangulator",
        "--database_path", str(db_path),
        "--image_path", str(subset_dir),
        "--input_path", str(model_txt),
        "--output_path", str(sparse_out),
    ])

    # export PLY
    run([
        "colmap", "model_converter",
        "--input_path", str(sparse_out),
        "--output_path", str(ply_out),
        "--output_type", "PLY",
    ])

    print(f"\n[DONE] ratio={args.subset_ratio}  PLY: {ply_out}")
    print("[TIP] If axes look flipped/weird, try adding: --no-opengl-corr")

    # (NEW) copy to scene folder
    if args.copy_ply_to:
        scene_dir = Path(args.copy_ply_to)
        copy_ply_to_scene(ply_out, scene_dir, args.copy_ply_name, overwrite=args.copy_overwrite)


if __name__ == "__main__":
    main()
