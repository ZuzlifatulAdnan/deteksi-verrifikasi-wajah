"""
Face Verification System — FastAPI Backend
Endpoints: register, verify (1:1), identify (1:N), presensi, admin
"""
import os
import uuid
import base64
import logging
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, APIRouter, File, UploadFile, Form, Request, HTTPException, Depends, Security
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from . import database as db
from .face_engine import FaceEngine
from .anti_spoof import PassiveAntiSpoof

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Admin authentication — semua endpoint admin wajib kirim header X-Admin-Key
# ---------------------------------------------------------------------------
_ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin(key: str = Security(_admin_key_header)):
    if not _ADMIN_SECRET:
        raise HTTPException(503, "ADMIN_SECRET belum dikonfigurasi di server")
    if not key or key != _ADMIN_SECRET:
        raise HTTPException(401, "Unauthorized: X-Admin-Key tidak valid")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR  = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Allowed origins for CORS — "*" is incompatible with allow_credentials=True
_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app = FastAPI(
    title="Face Verification System",
    description="Sistem deteksi, pendaftaran, verifikasi wajah, dan presensi",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,   # FIX: True + allow_origins=["*"] is rejected by browsers
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")

_engine: Optional[FaceEngine] = None
_spoof:  Optional[PassiveAntiSpoof] = None


def engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


def spoof() -> PassiveAntiSpoof:
    global _spoof
    if _spoof is None:
        _spoof = PassiveAntiSpoof()
    return _spoof


@app.on_event("startup")
async def startup():
    db.init_db()
    logger.info("Database siap.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

async def _img_from_upload(file: UploadFile):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Gagal decode file gambar")
    return img, data


def _img_from_b64(b64: str):
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Gagal decode base64 gambar")
    return img, raw


def _save_face(img: np.ndarray, user_id: str, bbox: list = None) -> Optional[str]:
    try:
        name = f"{user_id}_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(UPLOADS_DIR, name)
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad = 20
            h, w = img.shape[:2]
            crop = img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
            cv2.imwrite(path, crop)
        else:
            cv2.imwrite(path, img)
        return name
    except Exception as e:
        logger.warning("Gagal simpan gambar: %s", e)
        return None


def _err(status: int, msg: str, **extra) -> JSONResponse:
    """Helper: JSONResponse dengan status_code sebagai keyword."""
    return JSONResponse(content={"success": False, "error": msg, **extra}, status_code=status)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@api.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "model_loaded": _engine is not None,
        "users_registered": len(db.list_users()),
        "supabase": bool(db._supabase),
    }


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

@api.post("/register")
async def register(
    request: Request,
    user_id:      str = Form(...),
    name:         str = Form(...),
    email:        str = Form(None),
    phone:        str = Form(None),
    image:        UploadFile = File(None),
    image_base64: str = Form(None),
    skip_spoof:   bool = Form(False),
):
    try:
        if image:
            img, _ = await _img_from_upload(image)
        elif image_base64:
            img, _ = _img_from_b64(image_base64)
        else:
            raise HTTPException(400, "Tidak ada gambar yang dikirim")

        eng = engine()
        emb, face_info, err = eng.extract(img)
        if err:
            return _err(400, err)

        spoof_result = None
        if not skip_spoof:
            is_live, spoof_result = spoof().analyze(img, face_info.get("bbox"))
            # FIX: removed double condition — reject whenever is_live is False
            if not is_live:
                db.log_event(user_id, "register", is_live=False,
                             antispoof_score=spoof_result["score"],
                             ip=request.client.host,
                             notes="Ditolak: anti-spoofing gagal")
                return JSONResponse(
                    content={"success": False,
                             "error": "Gagal liveness check. Gunakan wajah asli, bukan foto/layar.",
                             "antispoof": spoof_result},
                    status_code=400
                )

        face_img = _save_face(img, user_id, face_info.get("bbox"))
        result = db.register_user(user_id, name, emb, email=email, phone=phone, face_image=face_img)
        db.log_event(user_id, "register", is_live=True,
                     antispoof_score=spoof_result["score"] if spoof_result else None,
                     ip=request.client.host)
        return {
            "success":   True,
            "message":   result["message"],
            "user_id":   user_id,
            "name":      name,
            "face_info": face_info,
            "antispoof": spoof_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Register error")
        return _err(500, str(e))


# ---------------------------------------------------------------------------
# Verify (1:1)
# ---------------------------------------------------------------------------

@api.post("/verify")
async def verify(
    request: Request,
    user_id:       str = Form(...),
    image:         UploadFile = File(None),
    image_base64:  str = Form(None),
    frame2_base64: str = Form(None),
    # FIX: removed liveness_passed — client-supplied bool allowed bypass of all liveness checks
):
    try:
        if image:
            img, _ = await _img_from_upload(image)
        elif image_base64:
            img, _ = _img_from_b64(image_base64)
        else:
            raise HTTPException(400, "Tidak ada gambar yang dikirim")

        user = db.get_user(user_id)
        if user is None:
            return JSONResponse(
                content={"success": False, "verified": False,
                         "error": f"User ID '{user_id}' tidak ditemukan"},
                status_code=404
            )

        eng = engine()
        emb, face_info, err = eng.extract(img)
        if err:
            return JSONResponse(content={"success": False, "verified": False, "error": err}, status_code=400)

        is_live_passive, spoof_result = spoof().analyze(img, face_info.get("bbox"))

        is_live_motion = False
        motion_score   = 0.0
        if frame2_base64:
            try:
                img2, _ = _img_from_b64(frame2_base64)
                is_live_motion, motion_score = eng.check_motion(img, img2, face_info.get("bbox"))
            except Exception as me:
                logger.debug("Motion check error: %s", me)

        # FIX: liveness is server-side only — passive OR motion, no client-supplied flag
        is_live  = is_live_passive or is_live_motion
        sim, is_match = eng.compare(emb, user["embedding"])
        verified = is_match and is_live

        db.log_event(user_id, "verify", similarity=sim, verified=verified,
                     is_live=is_live, antispoof_score=spoof_result["score"],
                     ip=request.client.host)
        return {
            "success":         True,
            "verified":        verified,
            "user_id":         user_id,
            "name":            user["name"],
            "similarity_pct":  round(sim * 100, 1),
            "similarity":      round(sim, 4),
            "confidence":      eng.confidence_label(sim),
            "is_live":         is_live,
            "is_live_passive": is_live_passive,
            "is_live_motion":  is_live_motion,
            "motion_score":    motion_score,
            "face_info":       face_info,
            "antispoof":       spoof_result,
            "verdict": ("TERVERIFIKASI ✓" if verified else
                        "LIVENESS GAGAL" if not is_live else "WAJAH TIDAK COCOK"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Verify error")
        return JSONResponse(content={"success": False, "verified": False, "error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Identify (1:N)
# ---------------------------------------------------------------------------

@api.post("/identify")
async def identify(
    request: Request,
    image:        UploadFile = File(None),
    image_base64: str = Form(None),
    threshold:    float = Form(0.35),
):
    try:
        if image:
            img, _ = await _img_from_upload(image)
        elif image_base64:
            img, _ = _img_from_b64(image_base64)
        else:
            raise HTTPException(400, "Tidak ada gambar yang dikirim")

        eng = engine()
        emb, face_info, err = eng.extract(img)
        if err:
            return _err(400, err)

        is_live, spoof_result = spoof().analyze(img, face_info.get("bbox"))

        # FIX: reject whenever not is_live — no secondary score threshold gap
        if not is_live:
            return JSONResponse(
                content={"success": False, "error": "Liveness check gagal",
                         "antispoof": spoof_result},
                status_code=400
            )

        match = db.find_best_match(emb, threshold=threshold)
        if match:
            db.log_event(match["user_id"], "identify", similarity=match["similarity"],
                         verified=True, is_live=is_live,
                         antispoof_score=spoof_result["score"], ip=request.client.host)
            return {
                "success":        True,
                "identified":     True,
                "user_id":        match["user_id"],
                "name":           match["name"],
                "similarity_pct": round(match["similarity"] * 100, 1),
                "similarity":     round(match["similarity"], 4),
                "confidence":     eng.confidence_label(match["similarity"]),
                "is_live":        is_live,
                "antispoof":      spoof_result,
                "face_info":      face_info,
            }
        else:
            db.log_event(None, "identify", verified=False, is_live=is_live,
                         antispoof_score=spoof_result["score"],
                         ip=request.client.host, notes="Tidak ditemukan")
            return {"success": True, "identified": False,
                    "message": "Wajah tidak ditemukan dalam database",
                    "is_live": is_live, "antispoof": spoof_result}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Identify error")
        return _err(500, str(e))


# ---------------------------------------------------------------------------
# Check liveness
# ---------------------------------------------------------------------------

@api.post("/check-liveness")
async def check_liveness(
    frame1_base64: str = Form(...),
    frame2_base64: str = Form(...),
    frame3_base64: str = Form(None),
):
    try:
        eng = engine()
        img1, _ = _img_from_b64(frame1_base64)
        img2, _ = _img_from_b64(frame2_base64)
        _, info1, _ = eng.extract(img1)
        bbox = info1.get("bbox") if info1 else None
        motion12, score12 = eng.check_motion(img1, img2, bbox)
        motion_score = score12
        if frame3_base64:
            img3, _ = _img_from_b64(frame3_base64)
            motion23, score23 = eng.check_motion(img2, img3, bbox)
            motion_score = max(score12, score23)
            is_live = motion12 or motion23
        else:
            is_live = motion12
        return {"success": True, "is_live": is_live, "motion_score": round(motion_score, 2),
                "message": "Liveness terdeteksi" if is_live else "Tidak ada gerakan terdeteksi"}
    except Exception as e:
        return _err(500, str(e))


# ---------------------------------------------------------------------------
# Presensi (Attendance)
# ---------------------------------------------------------------------------

@api.post("/presensi/masuk")
async def presensi_masuk(
    request:      Request,
    image_base64: str = Form(None),
    image:        UploadFile = File(None),
    device_id:    str = Form(None),
    location:     str = Form(None),
):
    """Clock-in: identifikasi wajah lalu catat presensi masuk."""
    try:
        if image:
            img, _ = await _img_from_upload(image)
        elif image_base64:
            img, _ = _img_from_b64(image_base64)
        else:
            raise HTTPException(400, "Tidak ada gambar yang dikirim")

        eng = engine()
        emb, face_info, err = eng.extract(img)
        if err:
            return _err(400, err)

        is_live, spoof_result = spoof().analyze(img, face_info.get("bbox"))
        if not is_live:
            return JSONResponse(
                content={"success": False, "error": "Liveness check gagal — gunakan wajah asli!",
                         "antispoof": spoof_result},
                status_code=400
            )

        match = db.find_best_match(emb, threshold=0.35)
        if not match:
            return _err(404, "Wajah tidak dikenali. Daftarkan diri terlebih dahulu.")

        rec = db.clock_attendance(
            user_id=match["user_id"], name=match["name"], tipe="masuk",
            similarity=match["similarity"], is_live=is_live,
            antispoof_score=spoof_result["score"],
            device_id=device_id, location=location,
            ip=request.client.host
        )
        db.log_event(match["user_id"], "presensi_masuk",
                     similarity=match["similarity"], verified=True, is_live=is_live,
                     antispoof_score=spoof_result["score"], ip=request.client.host)
        return {
            "success":        True,
            "tipe":           "masuk",
            "user_id":        match["user_id"],
            "name":           match["name"],
            "similarity_pct": round(match["similarity"] * 100, 1),
            "confidence":     eng.confidence_label(match["similarity"]),
            "waktu":          rec["waktu"],
            "is_live":        is_live,
            "antispoof":      spoof_result,
            "message":        f"✅ Selamat datang, {match['name']}! Presensi masuk tercatat.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Presensi masuk error")
        return _err(500, str(e))


@api.post("/presensi/keluar")
async def presensi_keluar(
    request:      Request,
    image_base64: str = Form(None),
    image:        UploadFile = File(None),
    device_id:    str = Form(None),
    location:     str = Form(None),
):
    """Clock-out: identifikasi wajah lalu catat presensi keluar."""
    try:
        if image:
            img, _ = await _img_from_upload(image)
        elif image_base64:
            img, _ = _img_from_b64(image_base64)
        else:
            raise HTTPException(400, "Tidak ada gambar yang dikirim")

        eng = engine()
        emb, face_info, err = eng.extract(img)
        if err:
            return _err(400, err)

        is_live, spoof_result = spoof().analyze(img, face_info.get("bbox"))
        if not is_live:
            return JSONResponse(
                content={"success": False, "error": "Liveness check gagal",
                         "antispoof": spoof_result},
                status_code=400
            )

        match = db.find_best_match(emb, threshold=0.35)
        if not match:
            return _err(404, "Wajah tidak dikenali.")

        rec = db.clock_attendance(
            user_id=match["user_id"], name=match["name"], tipe="keluar",
            similarity=match["similarity"], is_live=is_live,
            antispoof_score=spoof_result["score"],
            device_id=device_id, location=location,
            ip=request.client.host
        )
        db.log_event(match["user_id"], "presensi_keluar",
                     similarity=match["similarity"], verified=True, is_live=is_live,
                     antispoof_score=spoof_result["score"], ip=request.client.host)
        return {
            "success":        True,
            "tipe":           "keluar",
            "user_id":        match["user_id"],
            "name":           match["name"],
            "similarity_pct": round(match["similarity"] * 100, 1),
            "waktu":          rec["waktu"],
            "is_live":        is_live,
            "antispoof":      spoof_result,
            "message":        f"👋 Sampai jumpa, {match['name']}! Presensi keluar tercatat.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Presensi keluar error")
        return _err(500, str(e))


@api.get("/presensi/hari-ini")
async def presensi_hari_ini():
    return {
        "success": True,
        "tanggal": datetime.now().strftime("%Y-%m-%d"),
        "data":    db.get_today_attendance(),
        "stats":   db.get_attendance_stats(),
    }


@api.get("/presensi/riwayat/{user_id}")
async def presensi_riwayat(user_id: str, limit: int = 30):
    user = db.get_user(user_id)
    if user is None:
        return _err(404, f"User '{user_id}' tidak ditemukan")
    return {
        "success": True,
        "user_id": user_id,
        "name":    user["name"],
        "data":    db.get_attendance_history(user_id, limit),
    }


@api.get("/presensi/laporan")
async def presensi_laporan(
    dari: str = None,
    sampai: str = None,
    limit: int = 200,
):
    return {
        "success":  True,
        "dari":     dari,
        "sampai":   sampai,
        "data":     db.get_attendance_report(dari, sampai, limit),
        "total":    len(db.get_attendance_report(dari, sampai, limit)),
    }


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

@api.post("/admin/verify-key")
async def admin_verify_key(key: str = Security(_admin_key_header)):
    """Frontend pakai ini untuk cek apakah key valid sebelum tampilkan dashboard."""
    if not _ADMIN_SECRET:
        raise HTTPException(503, "ADMIN_SECRET belum dikonfigurasi")
    if not key or key != _ADMIN_SECRET:
        raise HTTPException(401, "Key tidak valid")
    return {"success": True}


@api.get("/users", dependencies=[Depends(require_admin)])
async def list_users():
    return {"success": True, "users": db.list_users()}


@api.delete("/users/{user_id}", dependencies=[Depends(require_admin)])
async def delete_user(user_id: str):
    if db.delete_user(user_id):
        return {"success": True, "message": f"User {user_id} dihapus"}
    return JSONResponse(content={"success": False, "error": "User tidak ditemukan"}, status_code=404)


@api.get("/logs", dependencies=[Depends(require_admin)])
async def get_logs(limit: int = 50):
    return {"success": True, "logs": db.get_logs(limit)}


# ---------------------------------------------------------------------------
# Headless endpoints — untuk integrasi Express.js / backend eksternal
# Database/penyimpanan dikelola sepenuhnya oleh backend pemanggil.
# ---------------------------------------------------------------------------

@api.post("/extract-embedding")
async def extract_embedding(
    image:        UploadFile = File(None),
    image_base64: str = Form(None),
    check_liveness: bool = Form(True),
):
    """
    Ekstrak 512-dim face embedding dari gambar.
    Backend eksternal menyimpan & membandingkan embedding sendiri.
    Response: { embedding: float[512], face_info, antispoof, is_live }
    """
    try:
        if image:
            img, _ = await _img_from_upload(image)
        elif image_base64:
            img, _ = _img_from_b64(image_base64)
        else:
            raise HTTPException(400, "Tidak ada gambar yang dikirim")

        eng = engine()
        emb, face_info, err = eng.extract(img)
        if err:
            return _err(400, err)

        is_live, spoof_result = None, None
        if check_liveness:
            is_live, spoof_result = spoof().analyze(img, face_info.get("bbox"))

        return {
            "success":   True,
            "embedding": emb.tolist(),          # 512 float — simpan di DB sendiri
            "face_info": face_info,
            "is_live":   is_live,
            "antispoof": spoof_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("extract-embedding error")
        return _err(500, str(e))


@api.post("/compare-embeddings")
async def compare_embeddings(
    embedding1: str = Form(...),   # JSON array [float, ...] 512-dim
    embedding2: str = Form(...),   # JSON array [float, ...] 512-dim
    threshold:  float = Form(0.30),
):
    """
    Bandingkan dua embedding (keduanya sudah disimpan di backend pemanggil).
    Response: { similarity, is_match, confidence }
    """
    try:
        import json
        e1 = np.array(json.loads(embedding1), dtype=np.float32)
        e2 = np.array(json.loads(embedding2), dtype=np.float32)
        if e1.shape != (512,) or e2.shape != (512,):
            return _err(400, "Embedding harus 512-dimensi")
        eng = engine()
        eng.SIMILARITY_THRESHOLD = threshold
        sim, is_match = eng.compare(e1, e2)
        return {
            "success":    True,
            "similarity": round(sim, 4),
            "similarity_pct": round(sim * 100, 1),
            "is_match":   is_match,
            "confidence": eng.confidence_label(sim),
            "threshold":  threshold,
        }
    except Exception as e:
        return _err(500, str(e))


@api.get("/face-image/{filename}")
async def face_image(filename: str):
    filename = os.path.basename(filename)   # strip any path components
    path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(404, "Gambar tidak ditemukan")


# ---------------------------------------------------------------------------
# Register router then serve static files
# ---------------------------------------------------------------------------

app.include_router(api)


@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    if not full_path:
        full_path = "index.html"

    # FIX: prevent path traversal — resolve and confirm within STATIC_DIR
    safe = os.path.normpath(full_path).lstrip(os.sep).replace("\\", "/")
    if ".." in safe.split("/"):
        raise HTTPException(400, "Path tidak valid")
    file_path = os.path.join(STATIC_DIR, safe)
    if not os.path.realpath(file_path).startswith(os.path.realpath(STATIC_DIR)):
        raise HTTPException(403, "Akses ditolak")

    if os.path.isfile(file_path):
        return FileResponse(file_path)
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    raise HTTPException(404, "File tidak ditemukan")
