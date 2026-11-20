"""
Passive Anti-Spoofing / Liveness Detection.

Teknik yang digunakan (tidak butuh model ML tambahan):
1. Moire Pattern Detection (FFT)  — deteksi pola layar/screen
2. LBP Texture Analysis            — tekstur wajah asli vs foto/layar
3. HSV Color Distribution          — warna kulit alami vs print/screen
4. Image Quality Metrics           — kualitas foto alami
5. Gradient Distribution           — kedalaman 3D wajah asli

Kombinasi teknik ini efektif mendeteksi:
- Foto yang dicetak (printed)
- Foto dari layar/screen
- Video yang diputar di layar

Untuk level keamanan perbankan, tambahkan:
- Silent-Face Anti-Spoofing (MinivisionAI) — model deep learning khusus
- FaceForensics++ model — untuk deepfake detection
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

try:
    from skimage.feature import local_binary_pattern
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

try:
    from scipy.fft import fft2, fftshift
    _SCIPY = True
except ImportError:
    _SCIPY = False
    import numpy.fft as npfft


class PassiveAntiSpoof:
    """
    Multi-technique passive liveness detection.
    Score akhir 0.0–1.0, threshold default 0.55.
    """

    THRESHOLD = 0.55

    WEIGHTS = {
        "moire":    0.25,
        "texture":  0.25,
        "color":    0.20,
        "quality":  0.15,
        "gradient": 0.15,
    }

    # -----------------------------------------------------------------------
    # Individual checks
    # -----------------------------------------------------------------------

    def _check_moire(self, roi: np.ndarray) -> Tuple[float, str]:
        """FFT: deteksi pola periodik dari layar/screen."""
        try:
            gray = cv2.cvtColor(cv2.resize(roi, (128, 128)), cv2.COLOR_BGR2GRAY).astype(float)

            if _SCIPY:
                mag = 20 * np.log(np.abs(fftshift(fft2(gray))) + 1)
            else:
                mag = 20 * np.log(np.abs(npfft.fftshift(npfft.fft2(gray))) + 1)

            h, w = mag.shape
            mag[h//2-5:h//2+5, w//2-5:w//2+5] = 0  # buang DC component

            mag_norm = mag / (mag.max() + 1e-7)
            strong = int(np.sum(mag_norm > 0.70))

            if strong < 5:
                return 1.0, "Distribusi frekuensi alami"
            if strong < 15:
                return 0.70, "Anomali frekuensi ringan"
            if strong < 30:
                return 0.30, "Kemungkinan pola layar/print"
            return 0.10, "Pola moire terdeteksi — kemungkinan foto layar"
        except Exception as e:
            logger.debug("Moire check error: %s", e)
            return 0.50, "Check gagal"

    def _check_texture(self, roi: np.ndarray) -> Tuple[float, str]:
        """LBP: wajah asli punya tekstur lebih kaya."""
        try:
            gray = cv2.cvtColor(cv2.resize(roi, (128, 128)), cv2.COLOR_BGR2GRAY)

            if _SKIMAGE:
                n_pts = 8
                lbp = local_binary_pattern(gray, n_pts, 1, method="uniform")
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_pts + 3), range=(0, n_pts + 2))
                hist = hist.astype(float) / (hist.sum() + 1e-7)
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                score_raw = entropy / np.log2(n_pts + 2)
            else:
                # Fallback: variasi gradient
                sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gm = np.sqrt(sobx**2 + soby**2)
                score_raw = min(gm.std() / (gm.mean() + 1e-7) / 2.0, 1.0)

            if score_raw > 0.75:
                return 1.0, "Tekstur wajah kaya dan alami"
            if score_raw > 0.55:
                return 0.70, "Tekstur cukup baik"
            if score_raw > 0.40:
                return 0.40, "Tekstur kurang bervariasi"
            return 0.10, "Tekstur datar — kemungkinan foto dicetak/layar"
        except Exception as e:
            logger.debug("Texture check error: %s", e)
            return 0.50, "Check gagal"

    def _check_color(self, roi: np.ndarray) -> Tuple[float, str]:
        """HSV: warna kulit wajah asli vs print/layar."""
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Rentang warna kulit (mencakup berbagai tone kulit)
            masks = []
            for lo, hi in [
                ([0, 15, 60],  [25, 200, 255]),   # skin terang
                ([0, 10, 50],  [30, 220, 255]),    # skin gelap
            ]:
                masks.append(cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))

            skin = cv2.bitwise_or(masks[0], masks[1])
            skin_ratio = np.sum(skin > 0) / (roi.shape[0] * roi.shape[1])
            sat_mean = float(hsv[:, :, 1].mean())

            if 0.12 <= skin_ratio <= 0.85 and sat_mean < 185:
                return 1.0, f"Warna kulit alami ({skin_ratio:.0%})"
            if 0.08 <= skin_ratio <= 0.92:
                return 0.65, f"Warna cukup ({skin_ratio:.0%})"
            if skin_ratio < 0.05:
                return 0.20, "Warna kulit sangat sedikit"
            return 0.30, "Distribusi warna tidak wajar"
        except Exception as e:
            logger.debug("Color check error: %s", e)
            return 0.50, "Check gagal"

    def _check_quality(self, roi: np.ndarray) -> Tuple[float, str]:
        """
        Kualitas gambar: foto asli punya variasi blur dan noise alami.
        Foto dicetak: terlalu datar. Layar: terlalu tajam.
        """
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_std = float((gray.astype(float) - blurred.astype(float)).std())

            focus_ok = 15 <= lap_var <= 9000
            noise_ok = 0.8 <= noise_std <= 30.0

            if focus_ok and noise_ok:
                return 1.0, f"Kualitas gambar natural (sharpness={lap_var:.0f})"
            if focus_ok or noise_ok:
                return 0.65, "Kualitas gambar cukup"
            if lap_var < 10:
                return 0.20, "Gambar terlalu datar — kemungkinan foto dicetak"
            if lap_var > 10000:
                return 0.20, "Gambar terlalu tajam — kemungkinan tampilan layar"
            return 0.40, "Kualitas gambar tidak biasa"
        except Exception as e:
            logger.debug("Quality check error: %s", e)
            return 0.50, "Check gagal"

    def _check_gradient(self, roi: np.ndarray) -> Tuple[float, str]:
        """Distribusi gradient: wajah 3D punya pola gradient alami."""
        try:
            gray = cv2.cvtColor(cv2.resize(roi, (64, 64)), cv2.COLOR_BGR2GRAY)
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            mag = np.sqrt(sx**2 + sy**2)
            direction = np.arctan2(sy, sx)

            dh, _ = np.histogram(direction, bins=16, range=(-np.pi, np.pi))
            dh = dh.astype(float) / (dh.sum() + 1e-7)
            dir_entropy = -np.sum(dh * np.log2(dh + 1e-10)) / np.log2(16)

            mag_cv = mag.std() / (mag.mean() + 1e-7)
            combined = dir_entropy * 0.6 + min(mag_cv / 3.0, 1.0) * 0.4

            if combined > 0.65:
                return 1.0, "Distribusi gradient alami"
            if combined > 0.45:
                return 0.70, "Distribusi gradient cukup"
            return 0.30, "Pola gradient tidak wajar"
        except Exception as e:
            logger.debug("Gradient check error: %s", e)
            return 0.50, "Check gagal"

    # -----------------------------------------------------------------------
    # Main analysis
    # -----------------------------------------------------------------------

    def analyze(self, image: np.ndarray, face_bbox: list = None) -> Tuple[bool, Dict]:
        """
        Jalankan semua pemeriksaan anti-spoofing.
        Returns: (is_live, result_dict)
        """
        # Crop ke area wajah
        if face_bbox:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
            h, w = image.shape[:2]
            px = int((x2 - x1) * 0.20)
            py = int((y2 - y1) * 0.20)
            roi = image[max(0, y1-py):min(h, y2+py), max(0, x1-px):min(w, x2+px)]
        else:
            roi = image

        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return False, {"score": 0.0, "verdict": "SPOOF", "reason": "Area wajah tidak valid", "checks": {}}

        # Resize ke ukuran standar
        mh, mw = roi.shape[:2]
        if max(mh, mw) > 300:
            scale = 300 / max(mh, mw)
            roi = cv2.resize(roi, (int(mw * scale), int(mh * scale)))

        # Jalankan semua check
        raw = {
            "moire":    self._check_moire(roi),
            "texture":  self._check_texture(roi),
            "color":    self._check_color(roi),
            "quality":  self._check_quality(roi),
            "gradient": self._check_gradient(roi),
        }

        # Weighted score
        total = sum(raw[k][0] * self.WEIGHTS[k] for k in raw)

        is_live = total >= self.THRESHOLD

        if total >= 0.80:
            confidence = "TINGGI"
        elif total >= 0.65:
            confidence = "SEDANG"
        else:
            confidence = "RENDAH"

        weakest = min(raw.items(), key=lambda x: x[1][0])[0]

        return is_live, {
            "score":        round(total, 3),
            "verdict":      "ASLI" if is_live else "PALSU",
            "is_live":      is_live,
            "confidence":   confidence,
            "threshold":    self.THRESHOLD,
            "weakest_check": weakest,
            "checks": {
                name: {"score": round(score, 3), "detail": detail}
                for name, (score, detail) in raw.items()
            },
        }
