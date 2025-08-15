# XD-Violence — Predicción + Tracking (WSANet + I3D/VGGish + YOLO/DeepSORT)

Este proyecto **predice violencia** en un video y (opcionalmente) **anota** los segmentos violentos con **detección/seguimiento de personas** y **resaltado de armas**.

El pipeline:
1. `predict_video.py`  
   - Re-muestrea el video a **24 fps**.  
   - Extrae **RGB features 1024-D** con **I3D** (por ventanas de 16 frames).  
   - Extrae **audio features 128-D** con **VGGish** (parches de 0.96 s).  
   - Fusiona (1024+128=**1152-D**), infiere con **WSANet** y produce **scores por snippet** y **segmentos violentos**.
2. (Opcional) Lanza `track_reid.py` para **detectar y seguir personas** (YOLOv8 + DeepSORT) y **marcar armas** en los **segmentos**.

Ejecución:
```bash
python predict_video.py --video "test_videos/video_13.mp4" --ckpt "ckpt/wsanodet_mix2.pkl" --i3d_rgb_weights "./pytorch-i3d/models/rgb_charades.pt" --threshold 0.5 --stride_frames 4 --agg topk --topk_ratio 0.2 --smooth_k 1 --visualize --out outputs/video_4_tracked.mp4 --det_conf 0.8 --weapon_conf 0.5 --weapon_classes "knife,gun,pistol" --device cuda --embedder_device cuda --show
```
### Dependencias
```bash
# Crea y activa un entorno (opcional)
conda env create -n xdv_env -f environment.yml
conda activate xdv_env

# Resto
pip install ultralytics deep-sort-realtime opencv-python librosa torchvggish
pip install "git+https://github.com/piergiaj/pytorch-i3d"