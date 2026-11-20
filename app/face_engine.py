"""
Face Engine — InsightFace (ArcFace + RetinaFace).

Models:
- buffalo_l  : ArcFace ViT-B/16, 99.4% LFW accuracy (~326 MB) ← default
- buffalo_sc : ArcFace MobileNet, faster (~30 MB)
- buffalo_s  : ArcFace smallest (~14 MB)

ArcFace cosine similarity thresholds:
- >= 0.35 : same person (verified)
- 0.20–0.34 : uncertain
- < 0.20  : different person
"""
import cv2
import numpy as np
import base64
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False
    logger.error("InsightFace tidak terinstall. Jalankan: pip install insightface onnxruntime")


class FaceEngine:
    SIMILARITY_THRESHOLD = 0.30

    def __init__(self, model_name: str = "buffalo_l"):
        if not _INSIGHTFACE_OK:
            raise RuntimeError("InsightFace tidak terinstall.\nJalankan: pip install insightface onnxruntime")
        self.model_name = model_name
        self._app = None
        self._load()

    def _load(self):
        logger.info("Memuat InsightFace model: %s ...", self.model_name)
        self._app = FaceAnalysis(
            name=self.model_name,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        logger.info("Model berhasil dimuat.")

    @staticmethod
    def decode(data) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, str):
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            data = base64.b64decode(data)
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Gagal mendecode gambar")
            return img
        raise TypeError(f"Tipe tidak didukung: {type(data)}")

    def detect(self, img: np.ndarray) -> list:
        return self._app.get(img)

    def extract(self, data) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[str]]:
        """Ekstrak face embedding. Returns: (embedding, face_info, error_message)."""
        try:
            img = self.decode(data)
            h, w = img.shape[:2]
            if h < 80 or w < 80:
                return None, None, "Gambar terlalu kecil (min 80x80 piksel)"

            # FIX: detect once — reuse result for both largest_face and num_faces
            faces = self.detect(img)
            if not faces:
                return None, None, "Wajah tidak terdeteksi. Pastikan pencahayaan cukup."

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            if face.embedding is None:
                return None, None, "Gagal mengekstrak fitur wajah."

            det_score = float(getattr(face, "det_score", 1.0))
            if det_score < 0.45:
                return None, None, f"Kepercayaan deteksi wajah rendah ({det_score:.2f}). Perbaiki pencahayaan."

            bbox = face.bbox.astype(int).tolist()
            fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if fw < 50 or fh < 50:
                return None, None, "Wajah terlalu jauh. Dekatkan wajah ke kamera."

            info = {
                "bbox":       bbox,
                "det_score":  round(det_score, 3),
                "face_size":  [fw, fh],
                "image_size": [w, h],
                "num_faces":  len(faces),  # reuse cached faces, no second detect()
                "pose":       face.pose.tolist() if hasattr(face, "pose") and face.pose is not None else None,
            }
            return face.embedding.astype(np.float32).copy(), info, None

        except Exception as exc:
            logger.exception("Error ekstrak embedding")
            return None, None, f"Error pemrosesan: {exc}"

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, bool]:
        n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0, False
        sim = float(np.dot(emb1, emb2) / (n1 * n2))
        return sim, sim >= self.SIMILARITY_THRESHOLD

    @staticmethod
    def confidence_label(sim: float) -> str:
        if sim >= 0.60: return "TINGGI"
        if sim >= 0.40: return "SEDANG"
        if sim >= 0.30: return "RENDAH"
        return "TIDAK_COCOK"

    def check_motion(self, frame1: np.ndarray, frame2: np.ndarray,
                     bbox: list = None) -> Tuple[bool, float]:
        def crop_face(img, b):
            if b is None: return img
            x1, y1, x2, y2 = b
            pad = 20
            h, w = img.shape[:2]
            return img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
        try:
            roi1 = crop_face(frame1, bbox)
            roi2 = crop_face(frame2, bbox)
            size = (128, 128)
            g1 = cv2.cvtColor(cv2.resize(roi1, size), cv2.COLOR_BGR2GRAY).astype(float)
            g2 = cv2.cvtColor(cv2.resize(roi2, size), cv2.COLOR_BGR2GRAY).astype(float)
            score = float(np.abs(g1 - g2).mean())
            return score >= 8.0, round(score, 2)
        except Exception:
            return False, 0.0
