import os
import math
import json
import random
import pickle
import einops
import librosa
import numpy as np
from tqdm import tqdm
from PIL import Image
import soundfile as sf

import torch
import torchvision.transforms as T

import open3d as o3d



def world_point_to_camera(c2w: np.ndarray, p_w: np.ndarray) -> np.ndarray:
    """
    c2w: (4,4) camera_to_world
    p_w: (3,) world point
    return: (3,) point in camera coords
    """
    R_cw = c2w[:3, :3]
    t_cw = c2w[:3, 3]
    p_c = R_cw.T @ (p_w - t_cw)
    return p_c


def stft(signal):
    spec = librosa.stft(signal, n_fft=512)  # hop 默认 128
    return spec


class RWAVSDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 sr=22050,
                 subset_ratio=1.0,
                 subset_strategy="uniform",
                 seed=42):
        super(RWAVSDataset, self).__init__()
        self.split = split
        self.sr = sr

        self.subset_ratio = float(subset_ratio)
        self.subset_strategy = subset_strategy
        self.seed = seed

        clip_len = 0.5  # second
        wav_len = int(2 * clip_len * sr)


        # =====boundary.pkl=====
        pkl_path = os.path.join(data_root, "32_boundary_points.pkl")
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        self.wall_xy_32 = np.asarray(d["points"]).astype(np.float32)  # (32,3)
        self.wall_rgb_32 = np.asarray(d["colors"]).astype(np.float32)  # (32,3) placeholder

        # =====interior.pkl=====
        pkl_path = os.path.join(data_root, "256_interior_points.pkl")
        with open(pkl_path, "rb") as f:
            points_dict = pickle.load(f)
        self.pcd_xyz_256 = np.asarray(points_dict['points']).astype(np.float32)    # (256,3)
        self.pcd_rgb_256 = np.asarray(points_dict['colors']).astype(np.float32)    # (256,3*50)


        # ===== 0) load point cloud ONCE (scene-level) =====
        ply_path = os.path.join(data_root, "points3D.ply")
        pcd = o3d.io.read_point_cloud(ply_path)
        self.pcd_xyz = np.asarray(pcd.points).astype(np.float32)            # (N,3)
        if pcd.has_colors():
            self.pcd_rgb = np.asarray(pcd.colors).astype(np.float32)        # (N,3) in [0,1]
        else:
            self.pcd_rgb = np.zeros_like(self.pcd_xyz, dtype=np.float32)    # (N,3)
        print(f"[PCD] {ply_path}  xyz: {self.pcd_xyz.shape}, rgb: {self.pcd_rgb.shape}")

        # === sound source (NOW: 3D) ===
        position_json = json.loads(
            open(os.path.join(os.path.dirname(data_root[:-1]), "position.json"), "r").read()
        )
        sp = position_json[data_root.split('/')[-2]]["source_position"]
        if len(sp) >= 3:
            source_pos_w = np.array(sp[:3], dtype=np.float32)
        else:
            source_pos_w = np.array([sp[0], sp[1], 0.0], dtype=np.float32)
        print(f"Split: {split}, sound source (world xyz): {source_pos_w}, wav_len: {wav_len}")

        # audio
        if os.path.exists(os.path.join(data_root, "binaural_syn_re.wav")):
            audio_bi, _ = librosa.load(os.path.join(data_root, "binaural_syn_re.wav"), sr=sr, mono=False)
        else:
            print("Unavilable, re-process binaural...")
            audio_bi_path = os.path.join(data_root, "binaural_syn.wav")
            audio_bi, _ = librosa.load(audio_bi_path, sr=sr, mono=False)
            audio_bi = audio_bi / np.abs(audio_bi).max()
            sf.write(os.path.join(data_root, "binaural_syn_re.wav"), audio_bi.T, sr, 'PCM_16')

        if os.path.exists(os.path.join(data_root, "source_syn_re.wav")):
            audio_sc, _ = librosa.load(os.path.join(data_root, "source_syn_re.wav"), sr=sr, mono=True)
        else:
            print("Unavilable, re-process source...")
            audio_sc_path = os.path.join(data_root, "source_syn.wav")
            audio_sc, _ = librosa.load(audio_sc_path, sr=sr, mono=True)
            audio_sc = audio_sc / np.abs(audio_sc).max()
            sf.write(os.path.join(data_root, "source_syn_re.wav"), audio_sc.T, sr, 'PCM_16')

        # ===== 1) poses used by training (scale) =====
        transforms_path = os.path.join(data_root, f"transforms_scale_{split}.json")
        transforms = json.loads(open(transforms_path, "r").read())

        # ===== 2) raw poses (non-scale) for point cloud fusion =====
        transforms_raw_path = os.path.join(data_root, f"transforms_{split}.json")
        transforms_raw = json.loads(open(transforms_raw_path, "r").read())

        # build map: "00001.png" -> c2w_raw
        raw_pose_map = {}
        for fr in transforms_raw["frames"]:
            name = fr["file_path"].split('/')[-1]
            raw_pose_map[name] = np.array(fr["transform_matrix"], dtype=np.float32).reshape(4, 4)

        data_list = []
        for item_idx, item in enumerate(transforms["camera_path"]):
            data = {}

            # compute source position in camera coords
            c2w = np.array(item["camera_to_world"], dtype=np.float32).reshape(4, 4)
            source_pos_c = world_point_to_camera(c2w, source_pos_w)
            data["source_pos_c"] = source_pos_c

            # extract key frames at 1 fps
            img_name = item["file_path"].split('/')[-1]              # "00001.png"
            time = int(img_name.split('.')[0])                       # 1
            data["img_idx"] = time

            # raw pose aligned by image name
            data["c2w_raw"] = raw_pose_map[img_name]                 # (4,4) np.float32

            st_idx = max(0, int(sr * (time - clip_len)))
            ed_idx = min(audio_bi.shape[1] - 1, int(sr * (time + clip_len)))
            if ed_idx - st_idx < int(clip_len * sr):
                continue

            audio_bi_clip = audio_bi[:, st_idx:ed_idx]
            audio_sc_clip = audio_sc[st_idx:ed_idx]

            # padding/cutting to wav_len
            if (ed_idx - st_idx) < wav_len:
                pad_len = wav_len - (ed_idx - st_idx)
                audio_bi_clip = np.concatenate((audio_bi_clip, np.zeros((2, pad_len), dtype=np.float32)), axis=1)
                audio_sc_clip = np.concatenate((audio_sc_clip, np.zeros((pad_len,), dtype=np.float32)), axis=0)
            elif (ed_idx - st_idx) > wav_len:
                audio_bi_clip = audio_bi_clip[:, :wav_len]
                audio_sc_clip = audio_sc_clip[:wav_len]

            # binaural
            spec_bi = stft(audio_bi_clip)
            mag_bi = np.abs(spec_bi)
            phase_bi = np.angle(spec_bi)
            data["mag_bi"] = mag_bi
            data["phase_bi"] = phase_bi
            data["wav_bi"] = audio_bi_clip

            # source
            spec_sc = stft(audio_sc_clip)
            mag_sc = np.abs(spec_sc)
            phase_sc = np.angle(spec_sc)
            data["mag_sc"] = mag_sc
            data["phase_sc"] = phase_sc
            data["wav_sc"] = audio_sc_clip

            data_list.append(data)

        # subset downsample
        if not (0 < self.subset_ratio <= 1.0):
            raise ValueError("subset_ratio must be in (0, 1].")
        if self.subset_ratio < 1.0 and len(data_list) > 0:
            n_total = len(data_list)
            n_keep = max(1, int(round(n_total * self.subset_ratio)))
            if self.subset_strategy == 'uniform':
                times = np.array([d["img_idx"] for d in data_list])
                order = np.argsort(times)
                pos = np.round(np.linspace(0, n_total - 1, n_keep)).astype(int)
                picked_idx = order[pos].tolist()
            elif self.subset_strategy == 'random':
                rng = np.random.default_rng(self.seed)
                picked_idx = rng.choice(np.arange(n_total), size=n_keep, replace=False)
                picked_idx = np.sort(picked_idx).tolist()
            else:
                raise ValueError("subset_strategy must be 'uniform' or 'random'.")
            data_list = [data_list[i] for i in picked_idx]
            print(f"[{self.split}] subset {self.subset_strategy}: keep {n_keep}/{n_total} ({self.subset_ratio * 100:.1f}%)")

        self.data_list = data_list

    def get_point_cloud(self):
        return self.pcd_xyz, self.pcd_rgb

    def get_interior_anchors(self):
        return self.pcd_xyz_256, self.pcd_rgb_256

    def get_boundary_anchors(self):
        return self.wall_xy_32, self.wall_rgb_32

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = RWAVSDataset(data_root="./data_RWAVS/1/", split="train")
    print(f"Dataset length: {len(dataset)}")
    xyz, rgb = dataset.get_point_cloud()
    print("PCD:", xyz.shape, rgb.shape)
    pcd_xyz_256, pcd_rgb_256 = dataset.pcd_xyz_256, dataset.pcd_rgb_256
    print("PCD 256:", pcd_xyz_256.shape, pcd_rgb_256.shape)
    for i in range(3):
        data = dataset[i]
        print(f"Data {i}: source pos in camera coords {data['source_pos_c']}, "
              f"c2w_raw {data['c2w_raw'].shape}, "
              f"wav_sc {data['wav_sc'].shape},"
              f"wav_bi {data['wav_bi'].shape}")
