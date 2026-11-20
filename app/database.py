"""
Database layer: SQLite (default) dengan opsional Supabase cloud sync.

Set env vars untuk aktifkan Supabase:
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_KEY=your-anon-key
"""
import os
import sqlite3
import base64
import logging
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "faces.db")

# ---------------------------------------------------------------------------
# Optional Supabase
# ---------------------------------------------------------------------------
_supabase = None
_SUPABASE_URL = os.getenv("SUPABASE_URL", "")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if _SUPABASE_URL and _SUPABASE_KEY:
    try:
        from supabase import create_client
        _supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        logger.info("Supabase terhubung: %s", _SUPABASE_URL)
    except Exception as e:
        logger.warning("Supabase gagal terhubung: %s — pakai SQLite saja.", e)


def _emb_to_b64(emb: np.ndarray) -> str:
    return base64.b64encode(emb.astype(np.float32).tobytes()).decode()

def _b64_to_emb(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).copy()

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# SQLite connection
# ---------------------------------------------------------------------------

def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = _conn()
    cur = c.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            email           TEXT,
            phone           TEXT,
            face_embedding  BLOB NOT NULL,
            face_image      TEXT,
            registered_at   TEXT DEFAULT (datetime('now','localtime')),
            updated_at      TEXT,
            is_active       INTEGER DEFAULT 1
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          TEXT,
            action           TEXT,
            similarity_score REAL,
            is_verified      INTEGER,
            is_live          INTEGER,
            antispoof_score  REAL,
            timestamp        TEXT DEFAULT (datetime('now','localtime')),
            ip_address       TEXT,
            notes            TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL,
            name            TEXT,
            tipe            TEXT NOT NULL,
            similarity      REAL,
            is_live         INTEGER,
            antispoof_score REAL,
            device_id       TEXT,
            location        TEXT,
            ip_address      TEXT,
            waktu           TEXT DEFAULT (datetime('now','localtime')),
            notes           TEXT
        )
    """)

    c.commit()
    c.close()
    logger.info("Database initialized at %s", DB_PATH)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def register_user(user_id: str, name: str, embedding: np.ndarray,
                  email: str = None, phone: str = None,
                  face_image: str = None) -> Dict:
    blob = embedding.astype(np.float32).tobytes()
    c = _conn()
    cur = c.cursor()
    action = "created"
    try:
        cur.execute(
            "INSERT INTO users (user_id,name,email,phone,face_embedding,face_image) VALUES (?,?,?,?,?,?)",
            (user_id, name, email, phone, blob, face_image)
        )
    except sqlite3.IntegrityError:
        cur.execute(
            """UPDATE users SET name=?,email=?,phone=?,face_embedding=?,
               face_image=COALESCE(?,face_image),updated_at=datetime('now','localtime')
               WHERE user_id=?""",
            (name, email, phone, blob, face_image, user_id)
        )
        action = "updated"
    c.commit()
    c.close()

    # Supabase sync
    if _supabase:
        try:
            payload = {"user_id": user_id, "name": name, "email": email,
                       "phone": phone, "face_embedding": _emb_to_b64(embedding),
                       "face_image": face_image}
            if action == "created":
                _supabase.table("users").insert(payload).execute()
            else:
                _supabase.table("users").update(payload).eq("user_id", user_id).execute()
        except Exception as e:
            logger.warning("Supabase sync gagal: %s", e)

    msg = "Wajah berhasil didaftarkan" if action == "created" else "Data wajah diperbarui"
    return {"action": action, "message": msg}


def get_user(user_id: str) -> Optional[Dict]:
    c = _conn()
    cur = c.cursor()
    cur.execute("SELECT * FROM users WHERE user_id=? AND is_active=1", (user_id,))
    row = cur.fetchone()
    c.close()
    if row is None:
        return None
    return {
        "user_id":       row["user_id"],
        "name":          row["name"],
        "email":         row["email"],
        "phone":         row["phone"],
        # FIX: .copy() so the array is writable and not backed by raw SQLite bytes
        "embedding":     np.frombuffer(row["face_embedding"], dtype=np.float32).copy(),
        "face_image":    row["face_image"],
        "registered_at": row["registered_at"],
    }


def list_users() -> List[Dict]:
    c = _conn()
    cur = c.cursor()
    cur.execute("SELECT user_id,name,email,phone,registered_at,face_image FROM users WHERE is_active=1 ORDER BY registered_at DESC")
    rows = cur.fetchall()
    c.close()
    return [dict(r) for r in rows]


def delete_user(user_id: str) -> bool:
    c = _conn()
    cur = c.cursor()
    cur.execute("UPDATE users SET is_active=0 WHERE user_id=?", (user_id,))
    affected = cur.rowcount
    c.commit()
    c.close()
    if affected and _supabase:
        try:
            _supabase.table("users").update({"is_active": False}).eq("user_id", user_id).execute()
        except Exception as e:
            logger.warning("Supabase delete sync gagal: %s", e)
    return affected > 0


# ---------------------------------------------------------------------------
# 1:N Identification
# ---------------------------------------------------------------------------

def find_best_match(query_emb: np.ndarray, threshold: float = 0.35) -> Optional[Dict]:
    c = _conn()
    cur = c.cursor()
    cur.execute("SELECT user_id,name,face_embedding FROM users WHERE is_active=1")
    rows = cur.fetchall()
    c.close()

    best, best_score = None, -1.0
    for row in rows:
        # FIX: .copy() — same fix as get_user
        stored = np.frombuffer(row["face_embedding"], dtype=np.float32).copy()
        n1, n2 = np.linalg.norm(query_emb), np.linalg.norm(stored)
        if n1 == 0 or n2 == 0:
            continue
        sim = float(np.dot(query_emb, stored) / (n1 * n2))
        if sim > best_score:
            best_score, best = sim, {"user_id": row["user_id"], "name": row["name"], "similarity": sim}

    if best and best_score >= threshold:
        return best
    return None


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def log_event(user_id: str, action: str, similarity: float = None,
              verified: bool = None, is_live: bool = None,
              antispoof_score: float = None, ip: str = None, notes: str = None):
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO logs (user_id,action,similarity_score,is_verified,is_live,antispoof_score,ip_address,notes) VALUES (?,?,?,?,?,?,?,?)",
        (user_id, action, similarity,
         1 if verified else (0 if verified is not None else None),
         1 if is_live else (0 if is_live is not None else None),
         antispoof_score, ip, notes)
    )
    c.commit()
    c.close()

    if _supabase:
        try:
            _supabase.table("logs").insert({
                "user_id": user_id, "action": action, "similarity_score": similarity,
                "is_verified": verified, "is_live": is_live,
                "antispoof_score": antispoof_score, "ip_address": ip, "notes": notes,
            }).execute()
        except Exception as e:
            logger.warning("Supabase log sync gagal: %s", e)


def get_logs(limit: int = 50) -> List[Dict]:
    c = _conn()
    cur = c.cursor()
    cur.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    c.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Attendance (Presensi)
# ---------------------------------------------------------------------------

def clock_attendance(user_id: str, name: str, tipe: str,
                     similarity: float = None, is_live: bool = None,
                     antispoof_score: float = None,
                     device_id: str = None, location: str = None,
                     ip: str = None, notes: str = None) -> Dict:
    """Catat presensi masuk atau keluar."""
    c = _conn()
    cur = c.cursor()
    waktu = _now()
    cur.execute(
        """INSERT INTO attendance
           (user_id,name,tipe,similarity,is_live,antispoof_score,device_id,location,ip_address,waktu,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, name, tipe, similarity,
         1 if is_live else 0, antispoof_score,
         device_id, location, ip, waktu, notes)
    )
    att_id = cur.lastrowid
    c.commit()
    c.close()

    if _supabase:
        try:
            _supabase.table("attendance").insert({
                "user_id": user_id, "name": name, "tipe": tipe,
                "similarity": similarity, "is_live": is_live,
                "antispoof_score": antispoof_score, "device_id": device_id,
                "location": location, "ip_address": ip, "waktu": waktu, "notes": notes,
            }).execute()
        except Exception as e:
            logger.warning("Supabase attendance sync gagal: %s", e)

    return {"id": att_id, "waktu": waktu}


def get_today_attendance() -> List[Dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "SELECT * FROM attendance WHERE waktu LIKE ? ORDER BY waktu DESC",
        (f"{today}%",)
    )
    rows = cur.fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_attendance_history(user_id: str, limit: int = 30) -> List[Dict]:
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "SELECT * FROM attendance WHERE user_id=? ORDER BY waktu DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cur.fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_attendance_report(date_from: str = None, date_to: str = None,
                          limit: int = 200) -> List[Dict]:
    c = _conn()
    cur = c.cursor()
    if date_from and date_to:
        cur.execute(
            "SELECT * FROM attendance WHERE waktu BETWEEN ? AND ? ORDER BY waktu DESC LIMIT ?",
            (f"{date_from} 00:00:00", f"{date_to} 23:59:59", limit)
        )
    elif date_from:
        cur.execute(
            "SELECT * FROM attendance WHERE waktu >= ? ORDER BY waktu DESC LIMIT ?",
            (f"{date_from} 00:00:00", limit)
        )
    else:
        cur.execute("SELECT * FROM attendance ORDER BY waktu DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_attendance_stats() -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    c = _conn()
    cur = c.cursor()
    cur.execute("SELECT COUNT(*) FROM attendance WHERE waktu LIKE ? AND tipe='masuk'", (f"{today}%",))
    masuk_today = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM attendance WHERE waktu LIKE ? AND tipe='keluar'", (f"{today}%",))
    keluar_today = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT user_id) FROM users WHERE is_active=1")
    total_users = cur.fetchone()[0]
    c.close()
    return {
        "masuk_hari_ini": masuk_today,
        "keluar_hari_ini": keluar_today,
        "total_karyawan": total_users,
        "tanggal": today,
    }
