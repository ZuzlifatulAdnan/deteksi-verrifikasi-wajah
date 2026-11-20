# FaceVerify — Sistem Verifikasi Wajah

Sistem verifikasi wajah production-grade berbasis **InsightFace ArcFace** dengan anti-spoofing multi-layer, backend FastAPI, dan database Supabase/SQLite. Cocok untuk presensi karyawan, KYC, akses kontrol, dan integrasi aplikasi Android/iOS.

---

## Fitur Utama

| Fitur | Keterangan |
|---|---|
| **Pendaftaran Wajah** | Daftarkan wajah baru dengan webcam, simpan 512-dim ArcFace embedding |
| **Verifikasi 1:1** | Cocokkan wajah dengan pengguna tertentu berdasarkan ID |
| **Identifikasi 1:N** | Cari identitas dari seluruh database secara otomatis |
| **Anti-Spoofing** | 5 lapisan deteksi passive liveness (FFT, LBP, HSV, Quality, Gradient) |
| **Presensi / Absensi** | Clock-in dan clock-out berbasis wajah dengan timestamp otomatis |
| **Admin Dashboard** | Statistik, riwayat log, manajemen pengguna |
| **REST API** | Siap diintegrasikan dengan web, Android, dan iOS |

---

## Tech Stack

| Komponen | Teknologi | Keterangan |
|---|---|---|
| **Face Detection** | RetinaFace (InsightFace) | 97.8% accuracy, 5-point landmarks |
| **Face Recognition** | ArcFace buffalo_l (ViT-B/16) | 99.4% LFW benchmark, 512-dim embedding |
| **Anti-Spoofing** | Custom 5-layer passive | FFT · LBP · HSV · Quality · Gradient |
| **Backend** | FastAPI + Uvicorn | Async, OpenAPI docs otomatis |
| **Database** | Supabase (PostgreSQL) / SQLite | Cloud atau lokal |
| **Image Processing** | OpenCV, NumPy | Decode, resize, color conversion |
| **Texture Analysis** | scikit-image | LBP untuk anti-spoofing |
| **Frequency Analysis** | SciPy FFT | Deteksi pola Moire layar/cetak |

> Model buffalo_l (~326 MB) diunduh otomatis saat server pertama kali dijalankan.

---

## Cara Instalasi

### 1. Clone & masuk ke direktori

```bash
git clone <repo-url>
cd deteksi-wajah-verifikasi
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan server

```bash
python run.py
```

Buka browser: **http://localhost:8080**

### (Opsional) Setup via script Windows

```bat
setup.bat
```

---

## Konfigurasi Supabase (Cloud Database)

Tanpa konfigurasi, sistem menggunakan **SQLite lokal**. Untuk mode cloud (multi-device, mobile sync), gunakan Supabase.

### 1. Buat project di [supabase.com](https://supabase.com)

### 2. Set environment variables

```bash
# Windows
set SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
set SUPABASE_KEY=your-anon-or-service-role-key

# Linux/macOS
export SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
export SUPABASE_KEY=your-anon-or-service-role-key
```

Atau buat file `.env` di root project:

```env
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_KEY=your-service-role-key
```

### 3. Buat tabel di Supabase SQL Editor

```sql
-- Tabel pengguna
CREATE TABLE users (
  id           SERIAL PRIMARY KEY,
  user_id      TEXT UNIQUE NOT NULL,
  name         TEXT NOT NULL,
  email        TEXT,
  phone        TEXT,
  face_embedding BYTEA,
  face_image   TEXT,
  is_active    BOOLEAN DEFAULT true,
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now()
);

-- Tabel log verifikasi
CREATE TABLE logs (
  id             SERIAL PRIMARY KEY,
  user_id        TEXT,
  action         TEXT,
  similarity     FLOAT,
  verified       BOOLEAN,
  is_live        BOOLEAN,
  antispoof_score FLOAT,
  ip             TEXT,
  device_id      TEXT,
  location       TEXT,
  notes          TEXT,
  created_at     TIMESTAMPTZ DEFAULT now()
);

-- Tabel presensi (absensi)
CREATE TABLE attendance (
  id          SERIAL PRIMARY KEY,
  user_id     TEXT NOT NULL,
  name        TEXT,
  type        TEXT NOT NULL,  -- 'masuk' atau 'keluar'
  similarity  FLOAT,
  is_live     BOOLEAN,
  device_id   TEXT,
  location    TEXT,
  notes       TEXT,
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- Index untuk performa query
CREATE INDEX idx_users_user_id      ON users(user_id);
CREATE INDEX idx_attendance_user_id ON attendance(user_id);
CREATE INDEX idx_attendance_date    ON attendance(created_at);
CREATE INDEX idx_logs_created_at    ON logs(created_at);
```

---

## REST API Reference

Base URL: `http://localhost:8080`  
Semua endpoint mengembalikan JSON. Untuk mobile, tambahkan header `Content-Type: multipart/form-data` pada request POST.

### Health Check

```
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "users_registered": 42
}
```

---

### Pendaftaran Wajah

```
POST /api/register
Content-Type: multipart/form-data
```

| Field | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `user_id` | string | ✅ | ID unik pengguna (NIK, NIP, dsb.) |
| `name` | string | ✅ | Nama lengkap |
| `email` | string | — | Email |
| `phone` | string | — | Nomor HP |
| `image_base64` | string | ✅* | Gambar wajah format base64 (data URI atau raw) |
| `image` | file | ✅* | Alternatif: upload file gambar langsung |
| `skip_spoof` | boolean | — | `false` (default). `true` melewati anti-spoof (dev only) |

*Salah satu dari `image_base64` atau `image` wajib diisi.

**Response sukses:**
```json
{
  "success": true,
  "message": "Wajah berhasil didaftarkan",
  "user_id": "12345",
  "name": "Budi Santoso",
  "face_info": { "det_score": 0.98, "bbox": [x1, y1, x2, y2] },
  "antispoof": { "is_live": true, "score": 0.78, "checks": { ... } }
}
```

---

### Verifikasi Wajah (1:1)

```
POST /api/verify
Content-Type: multipart/form-data
```

| Field | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `user_id` | string | ✅ | ID pengguna yang akan diverifikasi |
| `image_base64` | string | ✅ | Frame pertama (wajah utama) |
| `frame2_base64` | string | — | Frame kedua untuk motion liveness check |

**Response:**
```json
{
  "success": true,
  "verified": true,
  "user_id": "12345",
  "name": "Budi Santoso",
  "similarity_pct": 87.4,
  "similarity": 0.874,
  "confidence": "TINGGI",
  "is_live": true,
  "verdict": "TERVERIFIKASI ✓",
  "antispoof": { "is_live": true, "score": 0.81 }
}
```

---

### Identifikasi Wajah (1:N)

```
POST /api/identify
Content-Type: multipart/form-data
```

| Field | Tipe | Keterangan |
|---|---|---|
| `image_base64` | string | Gambar wajah |
| `threshold` | float | Ambang batas kemiripan minimum (default: 0.35) |

**Response:**
```json
{
  "success": true,
  "identified": true,
  "user_id": "12345",
  "name": "Budi Santoso",
  "similarity_pct": 84.2,
  "confidence": "TINGGI",
  "is_live": true
}
```

---

### Presensi — Clock In

```
POST /api/presensi/masuk
Content-Type: multipart/form-data
```

| Field | Tipe | Keterangan |
|---|---|---|
| `image_base64` | string | Selfie wajah saat masuk |
| `device_id` | string | ID perangkat (opsional) |
| `location` | string | Lokasi/nama kantor (opsional) |

**Response:**
```json
{
  "success": true,
  "type": "masuk",
  "user_id": "12345",
  "name": "Budi Santoso",
  "similarity_pct": 86.1,
  "timestamp": "2026-11-21T08:03:22",
  "is_live": true,
  "verdict": "PRESENSI MASUK BERHASIL ✓"
}
```

---

### Presensi — Clock Out

```
POST /api/presensi/keluar
Content-Type: multipart/form-data
```

Field sama dengan `/api/presensi/masuk`.

---

### Presensi Hari Ini

```
GET /api/presensi/hari-ini
```

**Response:**
```json
{
  "success": true,
  "date": "2026-11-21",
  "total": 25,
  "records": [
    {
      "user_id": "12345",
      "name": "Budi Santoso",
      "masuk": "08:03",
      "keluar": "17:12",
      "durasi": "9j 9m"
    }
  ]
}
```

---

### Riwayat Presensi Pengguna

```
GET /api/presensi/riwayat/{user_id}?limit=30
```

---

### Laporan Presensi

```
GET /api/presensi/laporan?from=2026-11-01&to=2026-11-30
```

---

### Manajemen Pengguna

```
GET    /api/users              — Daftar semua pengguna aktif
DELETE /api/users/{user_id}    — Hapus pengguna (soft delete)
GET    /api/logs?limit=50      — Riwayat log verifikasi
GET    /api/face-image/{file}  — Ambil foto wajah tersimpan
```

---

### OpenAPI / Swagger Docs

Dokumentasi interaktif tersedia di:  
**http://localhost:8080/docs**

---

## Integrasi Android (Kotlin + Retrofit)

### 1. Tambah dependensi di `build.gradle`

```kotlin
implementation("com.squareup.retrofit2:retrofit:2.9.0")
implementation("com.squareup.retrofit2:converter-gson:2.9.0")
implementation("com.squareup.okhttp3:okhttp:4.12.0")
```

### 2. Definisikan interface

```kotlin
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Call
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

data class VerifyResponse(
    val success: Boolean,
    val verified: Boolean,
    val name: String?,
    val similarity_pct: Float,
    val confidence: String,
    val is_live: Boolean,
    val verdict: String
)

interface FaceVerifyApi {
    @Multipart
    @POST("/api/verify")
    fun verify(
        @Part("user_id") userId: RequestBody,
        @Part image: MultipartBody.Part
    ): Call<VerifyResponse>

    @Multipart
    @POST("/api/presensi/masuk")
    fun presensiMasuk(
        @Part image: MultipartBody.Part,
        @Part("device_id") deviceId: RequestBody,
        @Part("location") location: RequestBody
    ): Call<VerifyResponse>
}
```

### 3. Panggil API

```kotlin
val retrofit = Retrofit.Builder()
    .baseUrl("http://192.168.1.100:8080/")  // IP server
    .addConverterFactory(GsonConverterFactory.create())
    .build()

val api = retrofit.create(FaceVerifyApi::class.java)

// Konversi bitmap ke RequestBody
val bitmapBytes = ByteArrayOutputStream()
    .also { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
    .toByteArray()
val imagePart = MultipartBody.Part.createFormData(
    "image", "face.jpg",
    bitmapBytes.toRequestBody("image/jpeg".toMediaType())
)
val userIdBody = userId.toRequestBody("text/plain".toMediaType())

api.verify(userIdBody, imagePart).enqueue(object : Callback<VerifyResponse> {
    override fun onResponse(call: Call<VerifyResponse>, response: Response<VerifyResponse>) {
        val result = response.body()
        if (result?.verified == true) {
            // Verifikasi berhasil
            Toast.makeText(context, "✅ ${result.name}", Toast.LENGTH_SHORT).show()
        }
    }
    override fun onFailure(call: Call<VerifyResponse>, t: Throwable) {
        Log.e("FaceVerify", "Error: ${t.message}")
    }
})
```

---

## Integrasi iOS (Swift + URLSession)

```swift
import Foundation
import UIKit

struct VerifyResponse: Codable {
    let success: Bool
    let verified: Bool
    let name: String?
    let similarity_pct: Float
    let confidence: String
    let is_live: Bool
    let verdict: String
}

func verifyFace(userId: String, image: UIImage, completion: @escaping (VerifyResponse?) -> Void) {
    guard let url = URL(string: "http://192.168.1.100:8080/api/verify"),
          let imageData = image.jpegData(compressionQuality: 0.9) else { return }

    let boundary = UUID().uuidString
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

    var body = Data()
    // user_id field
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"user_id\"\r\n\r\n".data(using: .utf8)!)
    body.append("\(userId)\r\n".data(using: .utf8)!)
    // image file
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"image\"; filename=\"face.jpg\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    body.append(imageData)
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

    request.httpBody = body

    URLSession.shared.dataTask(with: request) { data, _, error in
        guard let data = data, error == nil else { completion(nil); return }
        let result = try? JSONDecoder().decode(VerifyResponse.self, from: data)
        DispatchQueue.main.async { completion(result) }
    }.resume()
}

// Penggunaan
verifyFace(userId: "12345", image: capturedImage) { result in
    if result?.verified == true {
        print("✅ Terverifikasi: \(result?.name ?? "")")
    } else {
        print("❌ \(result?.verdict ?? "Gagal")")
    }
}
```

---

## Anti-Spoofing — 5 Lapisan Keamanan

Sistem menggunakan pendekatan **passive liveness detection** tanpa model deep learning tambahan, sehingga ringan dan tidak memerlukan unduhan model ekstra.

| Lapisan | Bobot | Target | Teknik |
|---|---|---|---|
| **FFT Moire Pattern** | 25% | Layar HP/laptop/monitor | Deteksi pola frekuensi periodik dari piksel LED/LCD |
| **LBP Texture Analysis** | 25% | Foto dicetak / kertas | Analisis tekstur mikro — kulit asli lebih kaya variasi |
| **HSV Color Distribution** | 20% | Semua media palsu | Rentang warna kulit manusia yang valid (hue, saturation) |
| **Image Quality Metrics** | 15% | Foto re-capture | Laplacian variance + estimasi noise — gambar asli lebih tajam |
| **Gradient Distribution** | 15% | Wajah 2D (foto/video) | Wajah 3D punya distribusi gradien yang khas |

**Threshold gabungan:** skor ≥ 0.55 = LIVE. Setiap lapisan menghasilkan skor 0–1 dan dikombinasikan secara tertimbang.

---

## Struktur Direktori

```
deteksi-wajah-verifikasi/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI endpoints
│   ├── face_engine.py   # InsightFace wrapper (detect + embedding + compare)
│   ├── anti_spoof.py    # Passive anti-spoofing 5 lapisan
│   └── database.py      # SQLite / Supabase interface
├── static/
│   ├── index.html       # Landing page
│   ├── register.html    # UI pendaftaran wajah (webcam, hold-and-capture)
│   ├── verify.html      # UI verifikasi/identifikasi
│   ├── presensi.html    # UI presensi/absensi
│   ├── admin.html       # Dashboard admin
│   └── test.html        # Halaman test sederhana
├── data/
│   └── faces.db         # SQLite database (auto-dibuat)
├── uploads/             # Foto wajah tersimpan (auto-dibuat)
├── run.py               # Entry point
├── setup.bat            # Windows setup script
├── requirements.txt     # Python dependencies
├── .env.example         # Contoh konfigurasi environment
├── README.md
└── LICENSE
```

---

## Environment Variables

| Variable | Keterangan | Default |
|---|---|---|
| `SUPABASE_URL` | URL project Supabase | — (gunakan SQLite) |
| `SUPABASE_KEY` | API key Supabase (service role) | — |
| `PORT` | Port server | `8080` |
| `HOST` | Host server | `0.0.0.0` |

---

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- RAM minimal 2 GB (model buffalo_l ~326 MB)
- Webcam (untuk UI web)
- Koneksi internet (untuk unduh model pertama kali)

---

## Lisensi

MIT License — lihat [LICENSE](LICENSE).

---

## Kontribusi

Pull request dan issue sangat disambut. Pastikan kode melewati linting sebelum submit.

---

*Dibangun dengan InsightFace, FastAPI, dan OpenCV.*
