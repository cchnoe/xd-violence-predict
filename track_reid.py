# track_reid.py
import argparse
import os
import colorsys
from typing import List, Tuple, Dict
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ============ utilidades ============

def parse_segments(s: str) -> List[Tuple[float, float]]:
    segs = []
    if not s:
        return segs
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        segs.append((float(a), float(b)))
    return segs

def time_in_segments(t: float, segments: List[Tuple[float, float]]) -> bool:
    """
    Devuelve True si el tiempo t está dentro de alguno de los segmentos de violencia.
    """
    for s, e in segments:
        if s <= t <= e:
            return True
    return False

def color_from_index(idx: int) -> Tuple[int, int, int]:
    hue = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(255 * b), int(255 * g), int(255 * r)  # BGR

def color_from_label(label: str) -> Tuple[int, int, int]:
    h = (hash(label) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 1.0)
    return int(255 * b), int(255 * g), int(255 * r)  # BGR

def draw_box_with_label(img, x1, y1, x2, y2, color, label: str, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    th = th + base
    y_text = y1 - 6
    y_text = y_text if y_text - th > 0 else (y1 + th + 6)
    x_text = x1
    cv2.rectangle(img, (x_text, y_text - th), (x_text + tw + 6, y_text + 4), (0, 0, 0), -1)
    cv2.putText(img, label, (x_text + 3, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def build_name_set(result, wanted_names: List[str]) -> set:
    names_map = result.names if hasattr(result, "names") else {}
    existing = {v.lower() for v in names_map.values()}
    out = set()
    for w in wanted_names:
        w = w.strip().lower()
        if not w:
            continue
        if w in existing:
            out.add(w)
        else:
            continue
    return out

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = ua + ub - inter + 1e-6
    return inter / union

def center_dist(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = 0.5 * (ax1 + ax2), 0.5 * (ay1 + ay2)
    bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
    dx, dy = acx - bcx, acy - bcy
    return float((dx * dx + dy * dy) ** 0.5)

def safe_crop(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def hsv_hist_feat(crop_bgr):
    """Histograma HSV simple como 'huella' de apariencia (48 dims)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    feat = np.concatenate([h, s, v], axis=0).astype(np.float32)
    feat = feat / (np.sum(feat) + 1e-6)
    return feat.reshape(-1)

def hist_similarity(fa, fb):
    """Similaridad simple (correlación). 1=igual, 0~distinto."""
    if fa is None or fb is None:
        return 0.0
    # correlación de Pearson aproximada
    a = fa - fa.mean()
    b = fb - fb.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / denom)

# ============ main ============

def main():
    ap = argparse.ArgumentParser("Person re-ID + armas en segmentos violentos")
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--segments", required=True, type=str, help='"ini-fin;ini-fin;..." (seg)')
    ap.add_argument("--out", type=str, default="outputs/annotated.mp4")
    ap.add_argument("--device", type=str, default="cuda", help="cuda|cpu para YOLO")
    ap.add_argument("--embedder_device", type=str, default=None,
                    help="cpu|cuda para el re-ID (DeepSORT). Por defecto igual a --device")
    ap.add_argument("--det_conf", type=float, default=0.35, help="Confianza mínima para persona")
    ap.add_argument("--imgsz", type=int, default=960, help="Tamaño de entrada YOLO (640/960/1280)")
    ap.add_argument("--weapon_model", type=str, default="", help="(opcional) YOLO extra para armas")
    ap.add_argument("--weapon_conf", type=float, default=0.40, help="Confianza mínima armas/objetos")
    ap.add_argument("--weapon_classes", type=str, default="knife,gun,pistol",
                    help="Clases de armas (coma-separado)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--label_with_tid", action="store_true",
                    help="Etiquetar como ID <tid> en vez de Persona <sid>")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    segments = parse_segments(args.segments)
    if not segments:
        print("[track_reid] No hay segmentos; nada que hacer.")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"No puedo abrir: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # YOLO: usa un modelo un poco mayor para mejor recall
    person_model = YOLO("yolov8x.pt")  # cambia a "yolov8m.pt" si tu GPU lo permite
    weapon_model = YOLO(args.weapon_model) if args.weapon_model else person_model

    use_gpu_for_reid = (args.embedder_device or args.device).lower().startswith("cuda")
    max_age_frames = max(45, int(fps * 2))

    tracker = DeepSort(
        max_age=1000,  # El tracker mantendrá el objeto durante 100 frames sin detección
        n_init=3,  # El tracker necesitará 3 detecciones consecutivas para confirmar el seguimiento
        nn_budget=100,  # La memoria para las características es de 100 objetos
        max_cosine_distance=0.6,  # Permitirá una distancia moderada entre características para seguir el objeto
        embedder="mobilenet",  # Usará el modelo 'mobilenet' para extraer características
        embedder_gpu=use_gpu_for_reid  # Usará la GPU para la extracción de características si está disponible
    )

    # IDs estables (SID) + “costura” con apariencia
    tid2sid: Dict[int, int] = {}
    sid_memory: Dict[int, Dict] = {}
    next_sid = 1
    STITCH_IOU_MIN = 0.20
    STITCH_DIST_FRAC = 0.12      # 12% del ancho
    STITCH_APPEAR_MIN = 0.5     # similitud mínima hist para aceptar
    STITCH_TOL_FRAMES = int(fps * 3)  # ventana temporal ~3s
    SID_CREATE_MIN_HITS = 2      # un TID debe durar >= 2 frames antes de crear SID nuevo

    tid_hits: Dict[int, int] = {}  # cuántos frames seguidos lleva vivo cada TID

    MIN_BOX_FRAC = 0.0015  # 0.15% del frame (filtra person minúsculas)
    DRAW_EVERYWHERE = True  # trackear SIEMPRE; dibujar SOLO en segmentos

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t_sec = frame_idx / fps
        in_seg = time_in_segments(t_sec, segments)
        frame_idx += 1

        # 1) Detectar personas SIEMPRE (para continuidad del tracker)
        rp = person_model.predict(
            source=frame_bgr, conf=args.det_conf, classes=[0],
            imgsz=args.imgsz, device=args.device, verbose=False
        )
        dets = []
        if rp:
            res = rp[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                for bb, sc in zip(xyxy, conf):
                    x1, y1, x2, y2 = bb.astype(int)
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area < MIN_BOX_FRAC * (W * H):
                        continue
                    dets.append(([x1, y1, x2 - x1, y2 - y1], float(sc), "person"))

        # 2) Actualizar tracker
        tracks = tracker.update_tracks(dets, frame=frame_bgr)

        # 3) Armas SOLO cuando vayamos a dibujar
        weapon_boxes = []
        if in_seg:
            if weapon_model is person_model:
                if rp:
                    res = rp[0]
                    wset = build_name_set(res, [w for w in args.weapon_classes.split(",") if w.strip()])
                    if res.boxes is not None and len(res.boxes) > 0 and len(wset) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        conf = res.boxes.conf.cpu().numpy()
                        cls  = res.boxes.cls.cpu().numpy().astype(int)
                        for bb, sc, c in zip(xyxy, conf, cls):
                            cname = res.names.get(int(c), str(c)).lower()
                            if cname in wset and sc >= args.weapon_conf:
                                x1, y1, x2, y2 = bb.astype(int)
                                weapon_boxes.append((x1, y1, x2, y2, cname, float(sc)))
            else:
                rw = weapon_model.predict(
                    source=frame_bgr, conf=args.weapon_conf, imgsz=args.imgsz,
                    device=args.device, verbose=False
                )
                if rw:
                    resw = rw[0]
                    names_w = resw.names
                    wset = {w.strip().lower() for w in args.weapon_classes.split(",") if w.strip()}
                    if resw.boxes is not None and len(resw.boxes) > 0 and len(wset) > 0:
                        xyxy = resw.boxes.xyxy.cpu().numpy()
                        conf = resw.boxes.conf.cpu().numpy()
                        cls  = resw.boxes.cls.cpu().numpy().astype(int)
                        for bb, sc, c in zip(xyxy, conf, cls):
                            cname = names_w.get(int(c), str(c)).lower()
                            if cname in wset and sc >= args.weapon_conf:
                                x1, y1, x2, y2 = bb.astype(int)
                                weapon_boxes.append((x1, y1, x2, y2, cname, float(sc)))

        # 4) Costura SID (ID estable)
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            tid = trk.track_id
            l, t, r, b = map(int, trk.to_ltrb())
            tid_hits[tid] = tid_hits.get(tid, 0) + 1

            # Si ya tiene SID, refrescamos memoria
            if tid in tid2sid:
                sid = tid2sid[tid]
                crop = safe_crop(frame_bgr, l, t, r, b)
                feat = hsv_hist_feat(crop)
                if sid not in sid_memory:
                    sid_memory[sid] = {"bbox": (l, t, r, b), "last_seen": frame_idx, "hists": deque(maxlen=10)}
                sid_memory[sid]["bbox"] = (l, t, r, b)
                sid_memory[sid]["last_seen"] = frame_idx
                if feat is not None:
                    sid_memory[sid]["hists"].append(feat)
                continue

            # Intento pegar a un SID reciente (oclusiones)
            best_sid, best_score = None, -1e9
            cand_feat = hsv_hist_feat(safe_crop(frame_bgr, l, t, r, b))
            for sid, info in sid_memory.items():
                if frame_idx - info["last_seen"] > STITCH_TOL_FRAMES:
                    continue
                i = iou((l, t, r, b), info["bbox"])
                d = center_dist((l, t, r, b), info["bbox"]) / max(1.0, W)
                app = 0.0
                if len(info["hists"]) > 0 and cand_feat is not None:
                    app = hist_similarity(cand_feat, np.mean(np.stack(info["hists"], 0), 0))
                # score: preferimos IoU y apariencia alta, y distancia baja
                score = 0.6 * i + 0.3 * app - 0.1 * d
                if score > best_score:
                    best_score = score
                    best_sid = sid

            accept = False
            if best_sid is not None:
                # reglas de aceptación
                i = iou((l, t, r, b), sid_memory[best_sid]["bbox"])
                d = center_dist((l, t, r, b), sid_memory[best_sid]["bbox"]) / max(1.0, W)
                app = 0.0
                if len(sid_memory[best_sid]["hists"]) > 0 and cand_feat is not None:
                    app = hist_similarity(cand_feat, np.mean(np.stack(sid_memory[best_sid]["hists"], 0), 0))
                if (i >= STITCH_IOU_MIN) or (app >= STITCH_APPEAR_MIN and d <= STITCH_DIST_FRAC):
                    accept = True

            if accept:
                sid = best_sid
            else:
                # solo creamos SID nuevo si el TID ha vivido >= N frames
                if tid_hits[tid] < SID_CREATE_MIN_HITS:
                    # aún no asignamos SID; saltamos este frame
                    continue
                sid = next_sid
                next_sid += 1

            tid2sid[tid] = sid
            if sid not in sid_memory:
                sid_memory[sid] = {"bbox": (l, t, r, b), "last_seen": frame_idx, "hists": deque(maxlen=10)}
            else:
                sid_memory[sid]["bbox"] = (l, t, r, b)
                sid_memory[sid]["last_seen"] = frame_idx
            if cand_feat is not None:
                sid_memory[sid]["hists"].append(cand_feat)

        # 5) Dibujar SOLO en segmentos
        draw_now = in_seg
        if draw_now:
            # personas
            for trk in tracks:
                if not trk.is_confirmed():
                    continue
                tid = trk.track_id
                l, t, r, b = map(int, trk.to_ltrb())
                sid = tid2sid.get(tid, None)
                if sid is None:
                    # aún no confirmamos SID; evita “explosión” de IDs dibujando solo cuando haya SID
                    continue
                label = f"ID {tid}" if args.label_with_tid else f"Persona {sid}"
                draw_box_with_label(frame_bgr, l, t, r, b, color_from_index(sid), label, thickness=2)

            # armas
            for (x1, y1, x2, y2, wname, wconf) in weapon_boxes:
                draw_box_with_label(frame_bgr, x1, y1, x2, y2, color_from_label(wname),
                                    f"{wname} {wconf:.2f}", thickness=2)

        writer.write(frame_bgr)
        if args.show:
            cv2.imshow("Violent persons + weapons", frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"[track_reid] Guardado: {args.out}")

if __name__ == "__main__":
    main()
