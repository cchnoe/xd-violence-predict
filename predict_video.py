"""
predict_video.py
===================
"""
import sys
sys.path.append("pytorch-i3d/")
import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

import argparse
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # workaround para conflicto de OpenMP
os.environ["OMP_NUM_THREADS"] = "1"          # opcional: limita hilos y reduce choques
from typing import List, Tuple
import shutil
import os
import subprocess
import tempfile
import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from model import Model
from pytorch_i3d import InceptionI3d
from torchvggish import vggish, vggish_input, vggish_params



def read_video_frames_short256(video_path):
    import cv2, numpy as np
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No puedo abrir: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = src_fps / 24.0
    frames = []
    idx = 0.0
    grab_i = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if grab_i >= int(round(idx)):
            ok, frame = cap.retrieve()
            if not ok: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_shorter_side(frame, short_side=256)
            frames.append(frame)
            idx += step
        grab_i += 1
    cap.release()
    if not frames:
        raise RuntimeError("No se leyeron frames.")
    return frames  # lista de frames con lado corto=256


def extract_i3d_rgb_features(
    frames: np.ndarray,
    weight_path: str,
    device: "torch.device",
    window_size: int = 16,
    stride: int = 8,
) -> np.ndarray:

    import math
    import numpy as np
    import torch
    from pytorch_i3d import InceptionI3d

    assert window_size > 0 and stride > 0
    T = frames.shape[0]
    if T < window_size:
        # Pad para al menos una ventana
        pad = np.zeros((window_size - T,) + frames.shape[1:], dtype=frames.dtype)
        frames = np.concatenate([frames, pad], axis=0)
        T = frames.shape[0]

    # Modelo
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items() if not k.replace("module.","").startswith("logits.")}
    i3d.load_state_dict(cleaned, strict=False)
    i3d.to(device).eval()

    # Indices de inicio de cada ventana
    starts = list(range(0, T - window_size + 1, stride))
    # Si el último trozo queda corto, añadimos una ventana final alineada al final
    if starts[-1] != T - window_size:
        starts.append(T - window_size)

    feats = []
    with torch.no_grad():
        for s in starts:
            e = s + window_size
            clip = frames[s:e]  # (window_size, 224, 224, 3)

            clip_t = torch.from_numpy(clip).to(device).float()
            clip_t = clip_t.permute(3, 0, 1, 2).unsqueeze(0)   # (1,3,T,H,W)
            # Normalización esperada por este port de I3D
            clip_t = clip_t * 2.0 - 1.0

            if hasattr(i3d, "extract_features"):
                f_map = i3d.extract_features(clip_t)           # (1,1024,T',7,7)
            else:
                raise RuntimeError("El I3D cargado no expone 'extract_features'.")

            f = f_map.mean(dim=[2, 3, 4]).squeeze(0)           # (1024,)
            feats.append(f)

    feats_t = torch.stack(feats, dim=0)                        # (N,1024)
    return feats_t.detach().cpu().numpy().astype(np.float32)



def extract_vggish_audio_features(
    video_path: str, device: "torch.device"
) -> "np.ndarray":

    target_sr = vggish_params.SAMPLE_RATE  # 16000

    # 1) Intento directo con librosa
    wav = None
    try:
        wav, sr = librosa.load(video_path, sr=target_sr, mono=True)
    except Exception:
        wav = None

    # 2) Fallback: usar FFmpeg para extraer WAV 16k mono
    if wav is None or len(wav) == 0:
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe is None:
            raise RuntimeError(
                "Librosa no pudo leer el MP4 y no encuentro FFmpeg en PATH.\n"
                "Instala FFmpeg o usa imageio-ffmpeg."
            )
        with tempfile.TemporaryDirectory() as tmpd:
            wav_path = os.path.join(tmpd, "audio_16k_mono.wav")
            cmd = [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-vn", "-ac", "1", "-ar", str(target_sr),
                "-f", "wav", wav_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            wav, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    if wav is None or len(wav) == 0:
        return np.zeros((0, 128), dtype=np.float32)

    # 3) VGGish: 0.96 s -> (N, 96, 64)
    examples = vggish_input.waveform_to_examples(wav, sr)  # puede ser np.ndarray o torch.Tensor
    # Manejo robusto del tipo devuelto
    if isinstance(examples, torch.Tensor):
        ex_t = examples
    else:
        ex_t = torch.from_numpy(np.asarray(examples))

    if ex_t.ndim != 3:
        # Esperamos (N, 96, 64). Si llega algo raro, devolvemos vacío para evitar crashear.
        return np.zeros((0, 128), dtype=np.float32)

    input_tensor = ex_t.unsqueeze(1).float().to(device)  # N x 1 x 96 x 64
    if input_tensor.shape[0] == 0:
        return np.zeros((0, 128), dtype=np.float32)

    model = vggish().to(device).eval()
    with torch.no_grad():
        emb = model(input_tensor)  # N x 128

    return emb.detach().cpu().numpy().astype(np.float32)




def align_audio_to_rgb(
    rgb_features: np.ndarray, audio_features: np.ndarray
) -> np.ndarray:

    T_rgb = rgb_features.shape[0]
    T_aud = audio_features.shape[0]
    # If there is no audio, return zeros for the audio part.
    if T_aud == 0:
        audio_aligned = np.zeros((T_rgb, audio_features.shape[1]), dtype=np.float32)
    else:
        # Resample audio features to length T_rgb using linear interpolation.
        # Compute fractional indices in the source audio feature sequence.
        src_indices = np.linspace(0, T_aud - 1, num=T_rgb)
        idx0 = np.floor(src_indices).astype(int)
        idx1 = np.clip(idx0 + 1, 0, T_aud - 1)
        lam = (src_indices - idx0).reshape(-1, 1)
        audio_aligned = (1 - lam) * audio_features[idx0] + lam * audio_features[idx1]
    fused = np.concatenate([rgb_features, audio_aligned], axis=1).astype(np.float32)
    return fused


def build_args_for_model(feature_dim, num_classes):
    import option  # viene de tu repo
    # no queremos leer la CLI real, así que parseamos “vacío” y sobreescribimos
    args = option.parser.parse_args([])
    args.feature_size = feature_dim
    args.num_classes = num_classes
    return args

def load_model(ckpt_path, feature_dim, device, num_classes=1):
    import torch
    args = build_args_for_model(feature_dim, num_classes)
    model = Model(args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.",""): v for k,v in state.items()}
    # igual que infer.py (estricto)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def resize_shorter_side(frame, short_side=256):
    import cv2
    h, w = frame.shape[:2]
    if h < w:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    else:
        new_w = short_side
        new_h = int(round(h * short_side / w))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def five_crops(frames_hw3_short256, crop=224):
    # frames_hw3_short256: lista/np.array de frames RGB con lado corto=256
    crops = []
    for pos in ("tl","tr","bl","br","center"):
        seq = []
        for f in frames_hw3_short256:
            H, W = f.shape[:2]
            if pos == "tl":
                y0, x0 = 0, 0
            elif pos == "tr":
                y0, x0 = 0, W - crop
            elif pos == "bl":
                y0, x0 = H - crop, 0
            elif pos == "br":
                y0, x0 = H - crop, W - crop
            else:  # center
                y0 = (H - crop) // 2
                x0 = (W - crop) // 2
            seq.append(f[y0:y0+crop, x0:x0+crop, :])
        crops.append(np.asarray(seq, dtype=np.float32) / 255.0)
    return crops  # lista de 5 arrays: (T,224,224,3)

def smooth_scores(scores: np.ndarray, k: int = 3) -> np.ndarray:
    """Media móvil (k impar). k=1 => sin suavizado."""
    if k <= 1:
        return scores
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    left = np.repeat(scores[:1], pad)
    right = np.repeat(scores[-1:], pad)
    x = np.concatenate([left, scores, right])
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="valid")


def aggregate_video_prob(scores: np.ndarray, method: str = "topk", topk_ratio: float = 0.15) -> float:
    """
    Agrega scores por snippet a probabilidad de video.
    - method: 'topk', 'max' o 'mean'
    - topk_ratio: fracción de snippets a promediar (solo 'topk')
    """
    if scores.size == 0:
        return 0.0
    m = method.lower()
    if m == "max":
        return float(scores.max())
    if m == "topk":
        k = max(1, int(np.ceil(topk_ratio * len(scores))))
        idx = np.argpartition(scores, -k)[-k:]
        return float(scores[idx].mean())
    # por defecto: media simple
    return float(scores.mean())


def predict_video(
    video_path: str,
    ckpt_path: str,
    i3d_rgb_weights: str,
    threshold: float = 0.5,
    device: "torch.device" = torch.device("cpu"),
    stride_frames: int = 8,
    agg: str = "topk",          # <--- NUEVO: 'topk' | 'max' | 'mean'
    topk_ratio: float = 0.15,   # <--- NUEVO: fracción para top-k (≈ setup de entrenamiento)
    smooth_k: int = 3,          # <--- NUEVO: media móvil (k impar); 1=sin suavizado
) -> "Tuple[float, List[Tuple[float, float]]]":
    """
    WSANet con fusión I3D-RGB (1024D) + VGGish (128D) = 1152D.
    5-crop TTA + stride temporal; agrega prob de video con 'agg' (top-k por defecto).
    """
    import numpy as np
    import torch

    window_size = 16
    seg_len_seconds = window_size / 24.0
    step_seconds    = stride_frames / 24.0

    # (1) Frames y 5-crops
    frames256 = read_video_frames_short256(video_path)
    crops_224 = five_crops(frames256, crop=224)

    # (2) Audio (una vez)
    audio_feats = extract_vggish_audio_features(video_path, device=device)

    # (3) Primer crop para dimensionar
    rgb_feats_0 = extract_i3d_rgb_features(
        crops_224[0], i3d_rgb_weights, device=device, window_size=window_size, stride=stride_frames
    )
    fused_0 = align_audio_to_rgb(rgb_feats_0, audio_feats)
    feature_dim = fused_0.shape[1]

    # (4) Modelo
    model = load_model(ckpt_path, feature_dim=feature_dim, device=device, num_classes=1)

    # (5) 5 crops -> scores por ventana y promedio entre crops
    per_crop_scores = []
    for crop_arr in crops_224:
        rgb_feats = extract_i3d_rgb_features(
            crop_arr, i3d_rgb_weights, device=device, window_size=window_size, stride=stride_frames
        )
        fused = align_audio_to_rgb(rgb_feats, audio_feats)
        x = torch.from_numpy(fused).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_seq, _ = model(x, seq_len=None)
            s = torch.sigmoid(logits_seq.squeeze(0).squeeze(-1)).cpu().numpy()
        per_crop_scores.append(s)

    # Alinea longitudes y promedia entre crops
    min_T = min(len(s) for s in per_crop_scores)
    if min_T == 0:
        return 0.0, []
    S = np.stack([s[:min_T] for s in per_crop_scores], axis=0)  # (5,T)
    mean_scores = S.mean(axis=0)                                 # (T,)

    # (6) Suavizado opcional y agregación robusta
    mean_scores = smooth_scores(mean_scores, k=smooth_k)
    video_prob = aggregate_video_prob(mean_scores, method=agg, topk_ratio=topk_ratio)

    # (7) Segmentos por umbral
    segments: "List[Tuple[float, float]]" = []
    active = mean_scores >= threshold
    start_idx = None
    for i, flag in enumerate(active):
        last = (i == len(active) - 1)
        if flag and start_idx is None:
            start_idx = i
        if (not flag and start_idx is not None) or (flag and last):
            end_idx = i if (flag and last) else i - 1
            start_time = start_idx * step_seconds
            end_time   = end_idx * step_seconds + seg_len_seconds
            segments.append((start_time, end_time))
            start_idx = None

    return float(video_prob), segments



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict violence in a video using WSANet and feature fusion."
    )
    # --- inferencia WSANet ---
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video (e.g. test_videos/video_2.mp4)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the trained WSANet checkpoint (e.g. ckpt/wsanodet_mix2.pkl)")
    parser.add_argument("--i3d_rgb_weights", type=str, required=True,
                        help="Path to the I3D RGB weights file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for classifying a snippet as violent")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device (cuda|cpu)")
    parser.add_argument("--stride_frames", type=int, default=8,
                        help="Temporal stride between I3D windows. 16=no overlap, 8=50%, 4=75%")
    parser.add_argument("--agg", type=str, default="topk", choices=["topk", "max", "mean"],
                        help="Aggregation of snippet scores into video probability")
    parser.add_argument("--topk_ratio", type=float, default=0.15,
                        help="Fraction of snippets used when agg=topk")
    parser.add_argument("--smooth_k", type=int, default=3,
                        help="Moving-average window (odd). 1=off")

    # --- visualización / tracking externo ---
    parser.add_argument("--visualize", action="store_true",
                        help="If violent, call external tracking (person re-ID + weapons).")
    parser.add_argument("--out", type=str, default="outputs/annotated.mp4",
                        help="Output annotated MP4 path (for tracking)")
    parser.add_argument("--det_conf", type=float, default=0.6,
                        help="YOLO min confidence for person detection")
    parser.add_argument("--weapon_conf", type=float, default=0.45,
                        help="YOLO min confidence for weapon/violent objects")
    parser.add_argument("--weapon_classes", type=str, default="knife,gun,pistol",
                        help='Weapon classes to highlight (comma-separated)')
    parser.add_argument("--weapon_model", type=str, default="",
                        help="(Optional) separate YOLO model for weapons (pt/onnx). Empty = reuse same model")
    parser.add_argument("--embedder_device", type=str, default=None,
                        help="Device for DeepSORT embedder (None => same as --device)")
    parser.add_argument("--show", action="store_true",
                        help="Show a playback window for tracking output")

    args = parser.parse_args()
    device = torch.device(args.device)

    prob, segments = predict_video(
        video_path=args.video,
        ckpt_path=args.ckpt,
        i3d_rgb_weights=args.i3d_rgb_weights,
        threshold=args.threshold,
        device=device,
        stride_frames=args.stride_frames,
        agg=args.agg,
        topk_ratio=args.topk_ratio,
        smooth_k=args.smooth_k,
    )

    label = "VIOLENT" if prob >= args.threshold else "NON-VIOLENT"
    print(f"Video: {args.video}")
    print(f"Mean violence probability: {prob:.3f} -> {label}")
    if segments:
        print("Violent segments (seconds):")
        for start, end in segments:
            print(f"  {start:.2f} – {end:.2f}")
    else:
        print("No segments exceeded the threshold.")

    # Llamada condicional al script externo (person re-ID + armas)
    if args.visualize and segments:
        import sys as _sys, subprocess as _sp
        seg_str = ";".join(f"{s:.3f}-{e:.3f}" for (s, e) in segments)
        embedder_dev = args.embedder_device if args.embedder_device else args.device

        cmd = [
            _sys.executable, "track_reid.py",
            "--video", args.video,
            "--segments", seg_str,
            "--out", args.out,
            "--device", args.device,
            "--embedder_device", embedder_dev,
            "--det_conf", str(args.det_conf),
            "--weapon_conf", str(args.weapon_conf),
            "--weapon_classes", args.weapon_classes,
        ]
        if args.weapon_model:
            cmd += ["--weapon_model", args.weapon_model]
        if args.show:
            cmd += ["--show"]

        print("[predict_video] Lanzando tracking + re-ID...")
        _sp.run(cmd, check=True)



if __name__ == "__main__":
    main()