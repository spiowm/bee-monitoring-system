#!/usr/bin/env python3
"""
Bee Monitoring System — one-command launcher for Kaggle / Google Colab.

Two cells in the notebook:
  Cell 1:  !git clone https://github.com/spiowm/bee-monitoring-system /kaggle/working/bms
  Cell 2:  !python /kaggle/working/bms/scripts/launch.py

That's it. You get a browser URL at the end.

MongoDB options (edit at the top of this file):
  USE_LOCAL_DB = True   → installs MongoDB locally (data resets on session end)
  USE_LOCAL_DB = False  → uses Atlas via MONGO_URI Kaggle/Colab secret
"""

import os, sys, re, time, signal, threading, subprocess, urllib.request
from pathlib import Path

# ── CONFIG (edit here if needed) ───────────────────────────────────────────
USE_LOCAL_DB  = True   # True = local MongoDB (zero-setup); False = Atlas from secret
BACKEND_PORT  = 8000
FRONTEND_PORT = 4173
# ───────────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parent.parent
BACKEND_DIR  = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"

def _env():
    if Path("/kaggle").exists(): return "kaggle"
    if Path("/content").exists(): return "colab"
    return "local"

ENV = _env()
OUTPUT_DIR = {
    "kaggle": Path("/kaggle/working/processed"),
    "colab":  Path("/content/processed"),
    "local":  BACKEND_DIR / "data" / "videos" / "processed",
}[ENV]

_procs: list[subprocess.Popen] = []


# ── HELPERS ────────────────────────────────────────────────────────────────

def sh(cmd, *, check=True, quiet=True, **kw):
    r = subprocess.run(cmd, capture_output=quiet, **kw)
    if check and r.returncode != 0:
        out = (r.stdout or b"").decode()[-2000:]
        err = (r.stderr or b"").decode()[-2000:]
        print(f"❌  {' '.join(str(c) for c in cmd)}\n{out}\n{err}")
        sys.exit(1)
    return r


def available(cmd):
    return subprocess.run(["which", cmd], capture_output=True).returncode == 0


def wait_http(url, timeout=40, label="service"):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    print(f"❌  {label} не відповідає на {url}")
    return False


def get_secret(name):
    """Read secret from Kaggle / Colab / env."""
    if ENV == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass
    if ENV == "colab":
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            pass
    return os.environ.get(name, "")


def cf_tunnel(port, label):
    """Start cloudflared quick tunnel, return public HTTPS URL."""
    print(f"   🌍  cloudflare tunnel → :{port}  ({label})")
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    _procs.append(proc)

    found = [None]

    def _read():
        for line in proc.stdout:
            m = re.search(r"https://[\w-]+\.trycloudflare\.com", line)
            if m:
                found[0] = m.group(0)
                return

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=90)

    if not found[0]:
        print(f"❌  Не вдалось отримати URL для {label} (timeout)")
        sys.exit(1)

    return found[0]


# ── STEP 1: system deps ────────────────────────────────────────────────────

def install_system_deps():
    print("📦  Системні залежності…")

    if not available("ffmpeg"):
        sh(["apt-get", "install", "-y", "ffmpeg", "-q"])

    if not available("cloudflared"):
        sh(["curl", "-fsSL", "-o", "/tmp/cf.deb",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"])
        sh(["dpkg", "-i", "/tmp/cf.deb"])

    if not available("uv"):
        sh([sys.executable, "-m", "pip", "install", "uv", "-q"])

    print("✅  Deps OK")


# ── STEP 2: MongoDB ────────────────────────────────────────────────────────

def setup_mongo():
    if not USE_LOCAL_DB:
        uri = get_secret("MONGO_URI") or os.environ.get("MONGO_URI", "")
        if not uri:
            print("⚠️  MONGO_URI secret не знайдено — перемикаємо на локальний MongoDB")
        else:
            print(f"✅  MongoDB Atlas")
            return uri

    print("🗄️   Локальний MongoDB…")
    # Try simple apt-get (works on some images)
    r = sh(["apt-get", "install", "-y", "mongodb", "-q"], check=False)
    if r.returncode != 0:
        # Ubuntu 22.04+ needs official repo
        sh(["bash", "-c",
            "curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg 2>/dev/null && "
            "echo 'deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse' > /etc/apt/sources.list.d/mongodb-org-7.0.list && "
            "apt-get update -q && apt-get install -y mongodb-org -q"], quiet=False, check=False)

    Path("/tmp/mongodb").mkdir(exist_ok=True)
    subprocess.Popen(
        ["mongod", "--fork", "--logpath", "/tmp/mongod.log",
         "--dbpath", "/tmp/mongodb", "--bind_ip", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    print("✅  Local MongoDB started (дані зникнуть після сесії)")
    return "mongodb://localhost:27017"


# ── STEP 3: backend ────────────────────────────────────────────────────────

def _find_model(name):
    """Search mounted Kaggle datasets, then repo fallback."""
    for p in Path("/kaggle/input").glob(f"*/{name}/best.pt"):
        return str(p)
    # flat layout in dataset
    for p in Path("/kaggle/input").glob(f"*/bee_pose*.pt"):
        if name == "bee_pose": return str(p)
    for p in Path("/kaggle/input").glob(f"*/ramp*.pt"):
        if name == "ramp_detector": return str(p)
    return str(BACKEND_DIR / "data" / "models" / name / "best.pt")


def setup_backend(mongo_uri):
    print("🔧  Backend…")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "data" / "videos" / "uploads").mkdir(parents=True, exist_ok=True)
    (BACKEND_DIR / "data" / "videos" / "test").mkdir(parents=True, exist_ok=True)

    model_path = _find_model("bee_pose")
    ramp_path  = _find_model("ramp_detector")

    if not Path(model_path).exists():
        print(f"⚠️   Модель не знайдена: {model_path}")
        print("    → Додай датасет 'bee-monitoring-models' до ноутбука")
        print(f"    → АБО завантаж: backend/data/models/bee_pose/best.pt")

    (BACKEND_DIR / ".env").write_text(f"""\
MONGO_URI={mongo_uri}
DB_NAME=buzz_buzz_buzz
MODEL_PATH={model_path}
RAMP_MODEL_PATH={ramp_path}
OUTPUT_DIR={OUTPUT_DIR}
CORS_ORIGINS=*
MAX_VIDEO_SIZE_MB=500
RAMP_DETECT_INTERVAL=30
""")

    sh(["uv", "sync"], cwd=BACKEND_DIR)
    print("✅  Backend налаштований")


def start_backend():
    print(f"🚀  Backend :{ BACKEND_PORT}…")
    log = open("/tmp/backend.log", "w")
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
        cwd=BACKEND_DIR, stdout=log, stderr=log,
    )
    _procs.append(proc)

    if not wait_http(f"http://localhost:{BACKEND_PORT}/jobs", label="Backend"):
        print("--- backend.log (останні рядки) ---")
        print(open("/tmp/backend.log").read()[-3000:])
        sys.exit(1)

    print(f"✅  Backend запущений (PID {proc.pid})")
    return proc


# ── STEP 4: frontend ───────────────────────────────────────────────────────

def build_frontend(api_url):
    print(f"⚛️   Frontend build  (VITE_API_URL={api_url})")
    print("    npm install…")

    if not (FRONTEND_DIR / "node_modules").exists():
        sh(["npm", "install"], cwd=FRONTEND_DIR)

    env = {**os.environ, "VITE_API_URL": api_url}
    r = sh(["npm", "run", "build"], cwd=FRONTEND_DIR, env=env, check=False, quiet=False)
    if r.returncode != 0:
        sys.exit(1)
    print("✅  Frontend збудований")


def start_frontend():
    print(f"🌐  Frontend :{ FRONTEND_PORT}…")
    log = open("/tmp/frontend.log", "w")
    # vite preview is in node_modules/.bin
    proc = subprocess.Popen(
        ["npx", "vite", "preview",
         "--port", str(FRONTEND_PORT), "--host", "0.0.0.0"],
        cwd=FRONTEND_DIR, stdout=log, stderr=log,
    )
    _procs.append(proc)
    time.sleep(3)
    print(f"✅  Frontend сервер (PID {proc.pid})")
    return proc


# ── STEP 5: keep alive ─────────────────────────────────────────────────────

def keep_alive(frontend_url, backend_url):
    border = "=" * 60
    print(f"\n{border}")
    print(f"  🚀  ВІДКРИЙ У БРАУЗЕРІ:\n")
    print(f"     ➡   {frontend_url}\n")
    print(f"  📊  Backend API:  {backend_url}")
    print(f"  📚  Swagger docs: {backend_url}/docs")
    print(f"{border}\n")
    print("Ctrl+C / interrupt cell → зупинити все\n")

    minutes = 0
    while True:
        time.sleep(60)
        minutes += 1
        try:
            urllib.request.urlopen(
                f"http://localhost:{BACKEND_PORT}/jobs", timeout=3
            )
        except Exception:
            pass
        if minutes % 15 == 0:
            print(f"🟢  Uptime: {minutes} хв  |  {frontend_url}")


# ── CLEANUP ────────────────────────────────────────────────────────────────

def cleanup(*_):
    print("\n🛑  Зупиняємо всі процеси…")
    for p in _procs:
        try: p.terminate()
        except Exception: pass
    sys.exit(0)


# ── MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("🐝  Bee Monitoring System — Launcher")
    print(f"    env={ENV}  repo={REPO_ROOT}  mongodb={'local' if USE_LOCAL_DB else 'atlas'}")
    print("=" * 55 + "\n")

    install_system_deps()
    mongo_uri = setup_mongo()
    setup_backend(mongo_uri)
    start_backend()

    print("\n── Тунелі (Cloudflare, без реєстрації) ──")
    backend_url = cf_tunnel(BACKEND_PORT, "Backend API")
    print(f"   Backend → {backend_url}")

    build_frontend(backend_url)
    start_frontend()

    frontend_url = cf_tunnel(FRONTEND_PORT, "Frontend UI")
    print(f"   Frontend → {frontend_url}")

    keep_alive(frontend_url, backend_url)
