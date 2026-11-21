"""
Database layer: MySQL via mysql-connector-python.

Set env vars di .env:
  MYSQL_HOST=localhost
  MYSQL_PORT=3306
  MYSQL_USER=root
  MYSQL_PASSWORD=your_password
  MYSQL_DATABASE=faceverify
"""
import os
import base64
import logging
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import mysql.connector
from mysql.connector import pooling, errors as mysql_errors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dari env
# ---------------------------------------------------------------------------
_MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST",     "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER",     "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "faceverify"),
    "charset":  "utf8mb4",
}

_pool: Optional[pooling.MySQLConnectionPool] = None

def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="faceverify_pool",
            pool_size=5,
            pool_reset_session=True,
            **_MYSQL_CONFIG,
        )
    return _pool

def _conn():
    return _get_pool().get_connection()


# ---------------------------------------------------------------------------
# Supabase (opsional, tetap didukung)
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
        logger.warning("Supabase gagal: %s", e)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _emb_to_b64(emb: np.ndarray) -> str:
    return base64.b64encode(emb.astype(np.float32).tobytes()).decode()

def _b64_to_emb(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).copy()

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Init DB — buat tabel jika belum ada
# ---------------------------------------------------------------------------
def init_db():
    # Pastikan database ada (buat jika belum)
    cfg_no_db = {k: v for k, v in _MYSQL_CONFIG.items() if k != "database"}
    tmp = mysql.connector.connect(**cfg_no_db)
    cur = tmp.cursor()
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{_MYSQL_CONFIG['database']}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    tmp.commit()
    cur.close()
    tmp.close()

    c = _conn()
    cur = c.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id              INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            user_id         VARCHAR(100) UNIQUE NOT NULL,
            name            VARCHAR(200) NOT NULL,
            email           VARCHAR(200),
            phone           VARCHAR(50),
            face_embedding  LONGBLOB NOT NULL,
            face_image      LONGTEXT,
            registered_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at      DATETIME,
            is_active       TINYINT(1) DEFAULT 1
        ) ENGINE=InnoDB
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id               INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            user_id          VARCHAR(100),
            action           VARCHAR(50),
            similarity_score DOUBLE,
            is_verified      TINYINT(1),
            is_live          TINYINT(1),
            antispoof_score  DOUBLE,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address       VARCHAR(45),
            notes            TEXT
        ) ENGINE=InnoDB
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id              INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            user_id         VARCHAR(100) NOT NULL,
            name            VARCHAR(200),
            tipe            VARCHAR(20) NOT NULL,
            similarity      DOUBLE,
            is_live         TINYINT(1),
            antispoof_score DOUBLE,
            device_id       VARCHAR(100),
            location        VARCHAR(200),
            ip_address      VARCHAR(45),
            waktu           DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes           TEXT
        ) ENGINE=InnoDB
    """)

    c.commit()
    cur.close()
    c.close()
    logger.info("MySQL database '%s' initialized.", _MYSQL_CONFIG["database"])


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def register_user(user_id: str, name: str, embedding: np.ndarray,
                  email: str = None, phone: str = None,
                  face_image: str = None) -> Dict:
    blob = embedding.astype(np.float32).tobytes()
    action = "created"
    c = _conn()
    cur = c.cursor()
    try:
        cur.execute(
            "INSERT INTO users (user_id,name,email,phone,face_embedding,face_image) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (user_id, name, email, phone, blob, face_image)
        )
    except mysql_errors.IntegrityError:
        cur.execute(
            """UPDATE users SET name=%s, email=%s, phone=%s,
               face_embedding=%s, face_image=COALESCE(%s,face_image),
               updated_at=NOW()
               WHERE user_id=%s""",
            (name, email, phone, blob, face_image, user_id)
        )
        action = "updated"
    c.commit()
    cur.close()
    c.close()

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
    cur = c.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE user_id=%s AND is_active=1", (user_id,))
    row = cur.fetchone()
    cur.close()
    c.close()
    if row is None:
        return None
    return {
        "user_id":       row["user_id"],
        "name":          row["name"],
        "email":         row["email"],
        "phone":         row["phone"],
        "embedding":     np.frombuffer(bytes(row["face_embedding"]), dtype=np.float32).copy(),
        "face_image":    row["face_image"],
        "registered_at": str(row["registered_at"]),
    }


def list_users() -> List[Dict]:
    c = _conn()
    cur = c.cursor(dictionary=True)
    cur.execute(
        "SELECT user_id,name,email,phone,registered_at,face_image "
        "FROM users WHERE is_active=1 ORDER BY registered_at DESC"
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    return [{**r, "registered_at": str(r["registered_at"])} for r in rows]


def delete_user(user_id: str) -> bool:
    c = _conn()
    cur = c.cursor()
    cur.execute("UPDATE users SET is_active=0 WHERE user_id=%s", (user_id,))
    affected = cur.rowcount
    c.commit()
    cur.close()
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
    cur = c.cursor(dictionary=True)
    cur.execute("SELECT user_id,name,face_embedding FROM users WHERE is_active=1")
    rows = cur.fetchall()
    cur.close()
    c.close()

    best, best_score = None, -1.0
    for row in rows:
        stored = np.frombuffer(bytes(row["face_embedding"]), dtype=np.float32).copy()
        n1, n2 = np.linalg.norm(query_emb), np.linalg.norm(stored)
        if n1 == 0 or n2 == 0:
            continue
        sim = float(np.dot(query_emb, stored) / (n1 * n2))
        if sim > best_score:
            best_score, best = sim, {
                "user_id": row["user_id"],
                "name":    row["name"],
                "similarity": sim,
            }

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
        "INSERT INTO logs "
        "(user_id,action,similarity_score,is_verified,is_live,antispoof_score,ip_address,notes) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (user_id, action, similarity,
         1 if verified else (0 if verified is not None else None),
         1 if is_live  else (0 if is_live  is not None else None),
         antispoof_score, ip, notes)
    )
    c.commit()
    cur.close()
    c.close()

    if _supabase:
        try:
            _supabase.table("logs").insert({
                "user_id": user_id, "action": action,
                "similarity_score": similarity, "is_verified": verified,
                "is_live": is_live, "antispoof_score": antispoof_score,
                "ip_address": ip, "notes": notes,
            }).execute()
        except Exception as e:
            logger.warning("Supabase log sync gagal: %s", e)


def get_logs(limit: int = 50) -> List[Dict]:
    c = _conn()
    cur = c.cursor(dictionary=True)
    cur.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    c.close()
    return [{**r, "timestamp": str(r["timestamp"])} for r in rows]


# ---------------------------------------------------------------------------
# Attendance (Presensi)
# ---------------------------------------------------------------------------

def clock_attendance(user_id: str, name: str, tipe: str,
                     similarity: float = None, is_live: bool = None,
                     antispoof_score: float = None,
                     device_id: str = None, location: str = None,
                     ip: str = None, notes: str = None) -> Dict:
    waktu = _now()
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO attendance "
        "(user_id,name,tipe,similarity,is_live,antispoof_score,"
        " device_id,location,ip_address,waktu,notes) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (user_id, name, tipe, similarity,
         1 if is_live else 0, antispoof_score,
         device_id, location, ip, waktu, notes)
    )
    att_id = cur.lastrowid
    c.commit()
    cur.close()
    c.close()

    if _supabase:
        try:
            _supabase.table("attendance").insert({
                "user_id": user_id, "name": name, "tipe": tipe,
                "similarity": similarity, "is_live": is_live,
                "antispoof_score": antispoof_score, "device_id": device_id,
                "location": location, "ip_address": ip,
                "waktu": waktu, "notes": notes,
            }).execute()
        except Exception as e:
            logger.warning("Supabase attendance sync gagal: %s", e)

    return {"id": att_id, "waktu": waktu}


def get_today_attendance() -> List[Dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    c = _conn()
    cur = c.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM attendance WHERE DATE(waktu)=%s ORDER BY waktu DESC",
        (today,)
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    return [{**r, "waktu": str(r["waktu"])} for r in rows]


def get_attendance_history(user_id: str, limit: int = 30) -> List[Dict]:
    c = _conn()
    cur = c.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM attendance WHERE user_id=%s ORDER BY waktu DESC LIMIT %s",
        (user_id, limit)
    )
    rows = cur.fetchall()
    cur.close()
    c.close()
    return [{**r, "waktu": str(r["waktu"])} for r in rows]


def get_attendance_report(date_from: str = None, date_to: str = None,
                          limit: int = 200) -> List[Dict]:
    c = _conn()
    cur = c.cursor(dictionary=True)
    if date_from and date_to:
        cur.execute(
            "SELECT * FROM attendance WHERE waktu BETWEEN %s AND %s "
            "ORDER BY waktu DESC LIMIT %s",
            (f"{date_from} 00:00:00", f"{date_to} 23:59:59", limit)
        )
    elif date_from:
        cur.execute(
            "SELECT * FROM attendance WHERE waktu >= %s ORDER BY waktu DESC LIMIT %s",
            (f"{date_from} 00:00:00", limit)
        )
    else:
        cur.execute("SELECT * FROM attendance ORDER BY waktu DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    c.close()
    return [{**r, "waktu": str(r["waktu"])} for r in rows]


def get_attendance_stats() -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    c = _conn()
    cur = c.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM attendance WHERE DATE(waktu)=%s AND tipe='masuk'", (today,)
    )
    masuk_today = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM attendance WHERE DATE(waktu)=%s AND tipe='keluar'", (today,)
    )
    keluar_today = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT user_id) FROM users WHERE is_active=1")
    total_users = cur.fetchone()[0]
    cur.close()
    c.close()
    return {
        "masuk_hari_ini":  masuk_today,
        "keluar_hari_ini": keluar_today,
        "total_karyawan":  total_users,
        "tanggal":         today,
    }
