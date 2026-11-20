"""
Jalankan Face Verification System.
Usage: python run.py
"""
import sys
import os

# Load .env jika ada (python-dotenv opsional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def check_deps():
    missing = []
    for pkg, imp in [
        ('fastapi',    'fastapi'),
        ('uvicorn',    'uvicorn'),
        ('insightface','insightface'),
        ('onnxruntime','onnxruntime'),
        ('cv2',        'cv2'),
        ('numpy',      'numpy'),
        ('PIL',        'PIL'),
        ('scipy',      'scipy'),
        ('skimage',    'skimage'),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    return missing

if __name__ == '__main__':
    print("=" * 55)
    print(" FaceVerify — Sistem Verifikasi Wajah")
    print("=" * 55)

    missing = check_deps()
    if missing:
        print(f"\n❌ Dependensi belum terinstall: {', '.join(missing)}")
        print("\nJalankan perintah ini dulu:")
        print("  pip install -r requirements.txt")
        print()
        sys.exit(1)

    print("\n✅ Semua dependensi tersedia")
    print("\n📋 Model yang digunakan:")
    print("   - InsightFace buffalo_l (ArcFace + RetinaFace)")
    print("     Akan diunduh otomatis saat pertama dijalankan (~326MB)")
    print("\n🔗 Akses di browser:")
    print("   http://localhost:8080                 -- Halaman utama")
    print("   http://localhost:8080/register.html   -- Daftarkan wajah")
    print("   http://localhost:8080/verify.html     -- Verifikasi wajah")
    print("   http://localhost:8080/presensi.html   -- Terminal presensi")
    print("   http://localhost:8080/test.html       -- Halaman uji coba")
    print("   http://localhost:8080/docs            -- API documentation")
    print("\n   Tekan Ctrl+C untuk menghentikan server")
    print("-" * 55)

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
