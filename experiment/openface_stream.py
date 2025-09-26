import os
import sys
import time
import csv
from datetime import datetime
from typing import List, Tuple, Optional

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def _resolve_openface_root() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    of_root = os.path.abspath(os.path.join(project_root, 'OpenFace-3.0'))
    return of_root


def _ensure_openface_path():
    of_root = _resolve_openface_root()
    if of_root not in sys.path:
        sys.path.append(of_root)
    # Also ensure nested libs are importable
    nested = [
        os.path.join(of_root, 'Pytorch_Retinaface'),
        os.path.join(of_root, 'STAR'),
        os.path.join(of_root, 'model'),
    ]
    for p in nested:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


_ensure_openface_path()

# Imports from OpenFace-3.0
from model.MLT import MLT  # type: ignore  # noqa: E402
from Pytorch_Retinaface.models.retinaface import RetinaFace  # type: ignore  # noqa: E402
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox  # type: ignore  # noqa: E402
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm  # type: ignore  # noqa: E402
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms  # type: ignore  # noqa: E402
from Pytorch_Retinaface.data import cfg_mnet  # type: ignore  # noqa: E402
from Pytorch_Retinaface.detect import load_model  # type: ignore  # noqa: E402


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _preprocess_frame(frame: np.ndarray, retinaface_model, device, resize: float = 1.0,
                      confidence_threshold: float = 0.5, nms_threshold: float = 0.4) -> np.ndarray:
    img = np.float32(frame)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loc, conf, landms = retinaface_model(img)

    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([
        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
        img.shape[3], img.shape[2]
    ]).to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    inds = np.where(scores > confidence_threshold)[0]
    if inds.size == 0:
        return np.zeros((0, 15), dtype=np.float32)
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    dets = np.concatenate((dets, landms), axis=1)
    return dets


class OpenFaceStreamer:
    def __init__(self, camera_index: int = 0, device: Optional[torch.device] = None):
        self.camera_index = camera_index
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.retinaface = None

    def load_models(self):
        of_root = _resolve_openface_root()
        weights_dir = os.path.join(of_root, 'weights')

        # Load MLT multitask model
        mlt = MLT()
        mlt_weights = os.path.join(weights_dir, 'stage2_epoch_7_loss_1.1606_acc_0.5589.pth')
        mlt.load_state_dict(torch.load(mlt_weights, map_location=self.device))
        self.model = mlt.to(self.device).eval()

        # Load RetinaFace for face detection
        # RetinaFace internally loads a pretrain tar via a relative path "./weights/..."
        # Ensure cwd is OpenFace-3.0 so that relative load succeeds
        _cwd = os.getcwd()
        try:
            os.chdir(of_root)
            retina = RetinaFace(cfg=cfg_mnet, phase='test')
        finally:
            os.chdir(_cwd)
        retina = load_model(retina, os.path.join(weights_dir, 'mobilenet0.25_Final.pth'), self.device.type == 'cpu')
        self.retinaface = retina.to(self.device).eval()

    def _ensure_models(self):
        if self.model is None or self.retinaface is None:
            self.load_models()

    def stream(self, duration_seconds: float, session_dir: str, conf_threshold: float = 0.7):
        self._ensure_models()
        os.makedirs(session_dir, exist_ok=True)
        out_csv_path = os.path.join(session_dir, 'openface_stream.csv')

        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f'Failed to open camera index {self.camera_index}')
            return out_csv_path

        start_time = time.time()
        with open(out_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            header = [
                'timestamp_iso', 't_rel_s', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'conf',
            ] + [f'emotion_{i}' for i in range(8)] + [f'gaze_{i}' for i in range(2)] + [f'au_{i}' for i in range(8)]
            writer.writerow(header)

            while time.time() - start_time < duration_seconds:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                dets = _preprocess_frame(frame, self.retinaface, self.device, confidence_threshold=conf_threshold)
                if dets.shape[0] == 0:
                    # No face detected; still advance time
                    ts_iso = datetime.utcnow().isoformat()
                    t_rel = time.time() - start_time
                    writer.writerow([ts_iso, f'{t_rel:.3f}', '', '', '', '', '', *([''] * (8 + 2 + 8))])
                    continue

                # Take top detection
                x1, y1, x2, y2, conf = dets[0][:5]
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                face = frame[y1i:y2i, x1i:x2i]
                if face.size == 0:
                    ts_iso = datetime.utcnow().isoformat()
                    t_rel = time.time() - start_time
                    writer.writerow([ts_iso, f'{t_rel:.3f}', '', '', '', '', '', *([''] * (8 + 2 + 8))])
                    continue

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                tensor = _transform(face_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    emotion_out, gaze_out, au_out = self.model(tensor)

                # Convert tensors to flat lists
                def _to_list(t):
                    try:
                        return t.detach().cpu().numpy().reshape(-1).tolist()
                    except Exception:
                        return []

                e_list = _to_list(emotion_out)
                g_list = _to_list(gaze_out)
                a_list = _to_list(au_out)

                # Pad/truncate to expected sizes for stable CSV schema
                e_list = (e_list + [None] * 8)[:8]
                g_list = (g_list + [None] * 2)[:2]
                a_list = (a_list + [None] * 8)[:8]

                ts_iso = datetime.utcnow().isoformat()
                t_rel = time.time() - start_time

                writer.writerow([
                    ts_iso, f'{t_rel:.3f}', x1i, y1i, x2i, y2i, float(conf),
                    *e_list, *g_list, *a_list
                ])

                # Also print simple lines for debugging
                print(f'Emotion Output: {e_list}')
                print(f'Gaze Output: {g_list}')
                print(f'AU Output: {a_list}')

        cap.release()
        return out_csv_path


