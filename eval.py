from __future__ import division, print_function, with_statement

import os
import json
import argparse
import pickle
import time
from typing import Dict, Any, Tuple, List



import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from scipy.signal import hilbert


from data import RWAVSDataset
from models.model import S2A_NVAS

# -------------------------
# Metrics
# -------------------------
def Envelope_distance(predicted_binaural: np.ndarray, gt_binaural: np.ndarray) -> float:
    # predicted_binaural / gt_binaural: [2, L]
    pred_env_l = np.abs(hilbert(predicted_binaural[0, :]))
    gt_env_l = np.abs(hilbert(gt_binaural[0, :]))
    d_l = np.sqrt(np.mean((gt_env_l - pred_env_l) ** 2))

    pred_env_r = np.abs(hilbert(predicted_binaural[1, :]))
    gt_env_r = np.abs(hilbert(gt_binaural[1, :]))
    d_r = np.sqrt(np.mean((gt_env_r - pred_env_r) ** 2))

    return float(d_l + d_r)


def eval_mag(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: [B, L] (single channel batch)
    return: magnitude spectrogram [B, F, T]
    """
    stft = torch.stft(
        wav,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=torch.hamming_window(400, device=wav.device),
        pad_mode="constant",
        return_complex=True,
    )
    return stft.abs()


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Evaluate a trained checkpoint on RWAVS val split (MAG/LRE/ENV).")
    p.add_argument("--data-root", type=str, required=True, help="e.g. ./data_RWAVS/1/")
    p.add_argument("--ckpt", type=str, required=True, help="e.g. logs/subset_1.0/1/99.pth")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    # dataset
    p.add_argument('--no-position', action="store_true")
    p.add_argument('--no-orientation', action="store_true")
    p.add_argument('--room_number', type=str, default="1")
    # model
    p.add_argument('--use_ori', action="store_true")
    p.add_argument('--use_boundary_token', action="store_true")
    p.add_argument('--boundary_r_scale', type=float, default="10.0")
    p.add_argument('--boundary_k_numfreqs', type=int, default="6")
    # test
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", type=str, default=None, help="default: <ckpt_dir>/eval_<epoch>")
    p.add_argument("--save-wav", action="store_true", help="save predicted/gt wavs for the first N samples")
    p.add_argument("--save-wav-n", type=int, default=20, help="how many samples to save")
    p.add_argument("--sr", type=int, default=22050, help="sampling rate")
    return p.parse_args()


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle DataParallel checkpoints with 'module.' prefix."""
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def magphase_to_wav(mag_2tf: np.ndarray, phase_tf: np.ndarray, length: int, n_fft: int = 512) -> np.ndarray:
    """
    mag_2tf: [2, T, F]
    phase_tf: [T, F]  (mono phase broadcast to both ears)
    returns wav: [2, length]
    """
    spec = mag_2tf * np.exp(1j * phase_tf[np.newaxis, :, :])   # [2,T,F]
    wav = librosa.istft(spec.transpose(0, 2, 1), n_fft=n_fft, length=length)  # [2,L]
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)
    return wav


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: str,
    save_wav: bool = False,
    save_wav_n: int = 20,
    sr: int = 22050,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:

    model.eval()

    stats = {"mag": [],"lre": [], "env": []}
    per_sample: List[Dict[str, Any]] = []

    wav_pred_dir = os.path.join(out_dir, "wavs_pred")
    wav_gt_dir = os.path.join(out_dir, "wavs_gt")
    if save_wav:
        os.makedirs(wav_pred_dir, exist_ok=True)
        os.makedirs(wav_gt_dir, exist_ok=True)

    saved = 0
    infer_times = []

    t = tqdm(total=len(loader), desc="[EVAL]", leave=False)
    for batch in loader:
        # move batch to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                if k == "img_idx":
                    batch[k] = v.to(device)
                else:
                    batch[k] = v.float().to(device)

        # forward (predict magnitude spectrogram)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        ret = model(batch)  # ret["reconstr"]: [B,2,T,F]
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_times.append(time.time() - t0)

        B = batch["wav_bi"].shape[0]

        # reconstruct wavs (per-sample; keep logic minimal & explicit)
        pred_wavs = []
        gt_wavs = []
        img_idxs = []

        for b in range(B):
            wav_prd = ret["out_pred_wav"][b].detach().cpu().numpy()  # [2,L]
            # ---compute MAG metric
            wav_gt = batch["wav_bi"][b].detach().cpu().numpy()  # [2,L]
            min_len = min(wav_prd.shape[-1], wav_gt.shape[-1])
            wav_prd = wav_prd[..., :min_len]
            wav_gt = wav_gt[..., :min_len]

            pred_wavs.append(wav_prd)
            gt_wavs.append(wav_gt)
            img_idx = int(batch["img_idx"][b].detach().cpu().item()) if "img_idx" in batch else -1
            img_idxs.append(img_idx)

        pred_wavs_np = np.stack(pred_wavs, axis=0)  # [B,2,L]
        gt_wavs_np = np.stack(gt_wavs, axis=0)      # [B,2,L]


        pred_wav_t = torch.from_numpy(pred_wavs_np).float().to(device)  # [B,2,L]
        gt_wav_t = torch.from_numpy(gt_wavs_np).float().to(device)      # [B,2,L]

        # ---- compute MAG & LRE in torch (batch) ----
        pred_spec_l, tgt_spec_l = eval_mag(pred_wav_t[:, 0]), eval_mag(gt_wav_t[:, 0])
        pred_spec_r, tgt_spec_r = eval_mag(pred_wav_t[:, 1]), eval_mag(gt_wav_t[:, 1])
        mag_batch = ((pred_spec_l - tgt_spec_l).pow(2).mean((1, 2)) +
                                (pred_spec_r - tgt_spec_r).pow(2).mean((1, 2)))  # [B]

        # LRE: left-right energy ratio error (dB)
        pred_lr_ratio = 10 * torch.log10((pred_wav_t[:, 0].pow(2).sum(-1) + 1e-5) /
                                         (pred_wav_t[:, 1].pow(2).sum(-1) + 1e-5))
        tgt_lr_ratio = 10 * torch.log10((gt_wav_t[:, 0].pow(2).sum(-1) + 1e-5) /
                                        (gt_wav_t[:, 1].pow(2).sum(-1) + 1e-5))
        lre_batch = (pred_lr_ratio - tgt_lr_ratio).abs()  # [B]

        # ---- ENV: per-sample in numpy (Hilbert envelope) ----
        for b in range(B):
            # skip silent gt
            if float((gt_wav_t[b, 0].pow(2).sum(-1)).detach().cpu().item()) == 0.0:
                continue

            env_val = Envelope_distance(pred_wavs_np[b], gt_wavs_np[b])

            stats["mag"].append(float(mag_batch[b].detach().cpu().item()))
            stats["lre"].append(float(lre_batch[b].detach().cpu().item()))
            stats["env"].append(float(env_val))


            per_sample.append({
                "img_idx": int(img_idxs[b]),
                "mag": float(mag_batch[b].detach().cpu().item()),
                "lre": float(lre_batch[b].detach().cpu().item()),
                "env": float(env_val),
            })

            if save_wav and saved < save_wav_n:
                sf.write(os.path.join(wav_pred_dir, f"{saved:04d}_img{img_idxs[b]}_pred.wav"),
                         pred_wavs_np[b].T, sr)
                sf.write(os.path.join(wav_gt_dir, f"{saved:04d}_img{img_idxs[b]}_gt.wav"),
                         gt_wavs_np[b].T, sr)
                saved += 1

        t.update(1)

    t.close()

    metrics = {
        "mag": float(np.mean(stats["mag"])) if len(stats["mag"]) else 0.0,
        "lre": float(np.mean(stats["lre"])) if len(stats["lre"]) else 0.0,
        "env": float(np.mean(stats["env"])) if len(stats["env"]) else 0.0,
        "infer_time": float(np.mean(infer_times)) if len(infer_times) else 0.0,
        "num_samples": int(len(stats["mag"])),
    }
    return metrics, per_sample


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # --- dataset/loader ---
    val_dataset = RWAVSDataset(args.data_root, "val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # --- model ---
    # model = ANeRF(conv=args.conv).to(device)
    pcd_xyz_256, pcd_rgb_256 = val_dataset.get_interior_anchors()
    boundary_xy_32,_ = val_dataset.get_boundary_anchors()
    configs = {
        "room_number": args.room_number,
        "use_ori": args.use_ori,
        "use_boundary_token": args.use_boundary_token,
        "boundary_r_scale": args.boundary_r_scale,
        "boundary_k_numfreqs": args.boundary_k_numfreqs,
    }
    model = S2A_NVAS(configs=configs, pcd_xyz=pcd_xyz_256, pcd_rgb=pcd_rgb_256,boundary_xy=boundary_xy_32).to(device)
    # --- load checkpoint ---
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None

    # --- out dir ---
    if args.out_dir is None:
        ckpt_dir = os.path.dirname(args.ckpt)
        tag = f"eval_{epoch}" if epoch is not None else "eval"
        out_dir = os.path.join(ckpt_dir, tag)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- run eval ---
    metrics, per_sample = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        out_dir=out_dir,
        save_wav=args.save_wav,
        save_wav_n=args.save_wav_n,
        sr=args.sr,
    )

    # --- save results ---
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "per_sample.pkl"), "wb") as f:
        pickle.dump(per_sample, f)

    print("===================================")
    print("Checkpoint:", args.ckpt)
    print("Output dir:", out_dir)
    print("MAG:", metrics["mag"])
    print("LRE:", metrics["lre"])
    print("ENV:", metrics["env"])
    print("Infer time (s):", metrics["infer_time"])
    print("Num samples:", metrics["num_samples"])
    print("===================================")


if __name__ == "__main__":
    main()
