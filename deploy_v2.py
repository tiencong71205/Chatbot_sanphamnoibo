"""
Full deployment script (Windows-compatible, SCP-only, no rsync)
Upload vhomenex_hybrid_rag_v2 to Ubuntu server, stop old containers,
build new images, start services, and optionally ingest documents.

Usage:
  python deploy_v2.py              # Full deploy + ingest
  python deploy_v2.py --no-ingest  # Deploy without ingest
  python deploy_v2.py --ingest-only  # Only trigger ingest (backend already running)
  python deploy_v2.py --verify-only  # Just verify health/products
  python deploy_v2.py --logs         # Show backend logs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
SSH_HOST = "192.168.1.226"
SSH_USER = "vconnex"
REMOTE_DIR = "/home/vconnex/vhomenex_hybrid_rag_v2"
LOCAL_PROJECT = Path(__file__).parent
LOCAL_DATASET = LOCAL_PROJECT / "data" / "source_docs"

ADMIN_TOKEN = "vhomenex_admin_2024"
BACKEND_URL = f"http://{SSH_HOST}:8000"

RAG_FILES = [
    "01_Cam_bien_hien_dien_RAG_v3.docx",
    "03_Cam_bien_chuyen_dong_anh_sang_RAG_v3.docx",
    "04_Bo_dieu_khien_hong_ngoai_RAG_v3.docx",
    "05_Dong_co_rem_RAG_v3.docx",
    "06_Cong_to_giam_sat_dien_RAG_v3.docx",
    "07_Cong_tac_chong_giat_BNN_RAG_v3.docx",
    "08_Cong_tac_cua_cuon_cua_cong_V2_RAG_v3.docx",
    "09_Gateway_RAG_v3.docx",
    "10_Khoa_dien_tu_RAG_v3.docx",
    "11_Cong_tac_dimmer_RAG_v3.docx",
    "12_Cong_tac_thong_minh_V2_RAG_v3.docx",
]

# Files to upload as individual items (relative to LOCAL_PROJECT)
INDIVIDUAL_FILES = [
    "docker-compose.yml",
    "Dockerfile.backend",
    "Dockerfile.frontend",
    ".env.docker",
    "requirements.txt",
    "data/product_catalog.json",
]

# Directories to upload (recursively via scp -r)
DIRECTORIES = [
    ("app", f"{REMOTE_DIR}/app"),
    ("ui", f"{REMOTE_DIR}/ui"),
    ("scripts", f"{REMOTE_DIR}/scripts"),
]

SSH_TARGET = f"{SSH_USER}@{SSH_HOST}"


def run(cmd: list[str] | str, check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess command with output visible to user."""
    if isinstance(cmd, str):
        print(f"\n$ {cmd}")
        result = subprocess.run(cmd, shell=True, text=True)
    else:
        print(f"\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd, shell=False, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def ssh(remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command on the remote Ubuntu server via SSH."""
    cmd = ["ssh", f"{SSH_TARGET}", remote_cmd]
    print(f"\n$ ssh {SSH_TARGET} '{remote_cmd}'")
    result = subprocess.run(cmd, shell=False, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] SSH command failed: {remote_cmd}")
        sys.exit(result.returncode)
    return result


def scp_file(local_path: str | Path, remote_path: str) -> None:
    """Copy a single file to the remote server."""
    cmd = ["scp", str(local_path), f"{SSH_TARGET}:{remote_path}"]
    print(f"\n$ scp {local_path} -> {remote_path}")
    subprocess.run(cmd, shell=False, text=True, check=True)


def scp_dir(local_path: str | Path, remote_parent: str) -> None:
    """Recursively copy a local directory into remote_parent/."""
    cmd = ["scp", "-r", str(local_path), f"{SSH_TARGET}:{remote_parent}"]
    print(f"\n$ scp -r {local_path} -> {remote_parent}/")
    subprocess.run(cmd, shell=False, text=True, check=True)


# ─── STEPS ───────────────────────────────────────────────────────────────────

def step_stop_containers() -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Stopping old containers")
    print("=" * 60)

    ssh(
        f"cd {REMOTE_DIR} 2>/dev/null && "
        "docker compose down --remove-orphans 2>/dev/null || true",
        check=False,
    )
    for name in ["vhomenex-backend-v2", "vhomenex-frontend-v2"]:
        ssh(f"docker stop {name} 2>/dev/null && docker rm {name} 2>/dev/null || true", check=False)
    print("✓ Old containers stopped")


def step_backup() -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Backing up old storage")
    print("=" * 60)

    ts = int(time.time())
    backup_dir = f"/home/{SSH_USER}/backups/rag_v2_{ts}"
    ssh(
        f"mkdir -p {backup_dir} && "
        f"cp -r {REMOTE_DIR}/storage {backup_dir}/ 2>/dev/null || true && "
        f"echo 'Backed up to {backup_dir}'",
        check=False,
    )
    print("✓ Storage backed up")


def step_prepare_remote_dirs() -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Preparing remote directory structure")
    print("=" * 60)

    ssh(
        f"mkdir -p "
        f"{REMOTE_DIR}/app "
        f"{REMOTE_DIR}/ui "
        f"{REMOTE_DIR}/scripts "
        f"{REMOTE_DIR}/data/source_docs "
        f"{REMOTE_DIR}/data/processed "
        f"{REMOTE_DIR}/storage/bm25 "
        f"{REMOTE_DIR}/logs"
    )
    print("✓ Remote directories prepared")


def step_upload_project() -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Uploading project source files")
    print("=" * 60)

    # Upload individual config files
    print("\n→ Config files:")
    for fname in INDIVIDUAL_FILES:
        local = LOCAL_PROJECT / fname
        if local.exists():
            remote = f"{REMOTE_DIR}/{fname}"
            scp_file(local, remote)
        else:
            print(f"  [SKIP] Not found: {local}")

    # Upload directories using scp -r
    # We put them directly inside REMOTE_DIR so scp -r copies contents correctly
    print("\n→ Source directories (app, ui, scripts):")
    for local_rel, remote_path in DIRECTORIES:
        local_abs = LOCAL_PROJECT / local_rel
        if local_abs.exists():
            # scp -r copies the DIRECTORY itself into the parent, so target parent
            remote_parent = remote_path.rsplit("/", 1)[0]  # e.g. REMOTE_DIR
            # First remove old dir so we get a clean copy
            dir_name = local_rel  # e.g. "app"
            ssh(f"rm -rf {REMOTE_DIR}/{dir_name}", check=False)
            scp_dir(local_abs, REMOTE_DIR)
        else:
            print(f"  [SKIP] Not found: {local_abs}")

    print("✓ Project files uploaded")


def step_upload_documents() -> None:
    print("\n" + "=" * 60)
    print("STEP 5: Uploading DOCX documents (11 files)")
    print("=" * 60)

    remote_docs = f"{REMOTE_DIR}/data/source_docs"
    found = 0
    missing = []

    for fname in RAG_FILES:
        local_file = LOCAL_DATASET / fname
        if local_file.exists():
            scp_file(local_file, f"{remote_docs}/{fname}")
            found += 1
        else:
            missing.append(fname)
            print(f"  [WARN] Not found locally: {fname}")

    if missing:
        print(f"\n[WARN] {len(missing)} files missing: {missing}")
        if LOCAL_DATASET.exists():
            print(f"  Available in {LOCAL_DATASET}:")
            for f in sorted(LOCAL_DATASET.glob("*.docx")):
                print(f"    {f.name}")

    print(f"\n✓ {found}/{len(RAG_FILES)} documents uploaded")
    ssh(f"ls -lh {remote_docs}/ | grep -c .docx || true", check=False)


def step_build_containers() -> None:
    print("\n" + "=" * 60)
    print("STEP 6: Building Docker images")
    print("=" * 60)

    ssh(f"cd {REMOTE_DIR} && docker compose build 2>&1 | tail -30")
    print("✓ Docker images built")


def step_start_containers() -> None:
    print("\n" + "=" * 60)
    print("STEP 7: Starting containers")
    print("=" * 60)

    ssh(f"cd {REMOTE_DIR} && docker compose up -d")
    print("\nWaiting 25 seconds for services to initialize...")
    time.sleep(25)
    ssh(f"cd {REMOTE_DIR} && docker compose ps")
    print("✓ Containers started")


def step_wait_backend(max_seconds: int = 120) -> bool:
    print("\n" + "=" * 60)
    print("STEP 8: Waiting for backend health check")
    print("=" * 60)

    for i in range(max_seconds // 10):
        result = subprocess.run(
            ["ssh", f"{SSH_TARGET}", f"curl -sf {BACKEND_URL}/ready"],
            shell=False, capture_output=True, text=True,
        )
        if result.returncode == 0 and ("ready" in result.stdout or "ok" in result.stdout):
            print(f"✓ Backend is ready! ({(i+1)*10}s)")
            return True
        print(f"  Waiting... ({(i+1)*10}s) - {result.stdout.strip()[:60]}")
        time.sleep(10)

    print("[WARN] Backend readiness timed out, proceeding anyway")
    return False


def step_ingest(recreate: bool = True) -> None:
    print("\n" + "=" * 60)
    print("STEP 9: Running document ingestion")
    print("=" * 60)

    endpoint = "/api/reindex" if recreate else "/api/ingest"
    payload = f'{{"admin_token": "{ADMIN_TOKEN}", "recreate_collection": true, "dry_run": false}}'

    print(f"\n→ POST {BACKEND_URL}{endpoint}")
    result = subprocess.run(
        [
            "ssh", f"{SSH_TARGET}",
            f"curl -s -X POST '{BACKEND_URL}{endpoint}' "
            f"-H 'Content-Type: application/json' "
            f"-d '{payload}'"
        ],
        shell=False, capture_output=True, text=True, timeout=600,
    )

    if result.returncode == 0 and result.stdout:
        print(f"Response: {result.stdout[:2000]}")
        print("✓ Ingestion triggered via API")
    else:
        print(f"[WARN] API ingest returned: {result.stderr}")
        print("→ Fallback: running ingestion directly inside container...")
        ingest_cmd = (
            "docker exec vhomenex-backend-v2 "
            "python3 -c \""
            "import sys; sys.path.insert(0, '/app'); "
            "from app.config import get_settings; "
            "from app.database.qdrant_store import QdrantStore; "
            "from app.database.bm25_store import BM25Store; "
            "from app.embeddings.ollama_embedding import OllamaEmbedding; "
            "from app.ingestion.ingest_service import IngestService; "
            "s = get_settings(); q = QdrantStore(s); b = BM25Store(s); e = OllamaEmbedding(s); "
            "svc = IngestService(s, q, b, e); "
            "r = svc.ingest_all(recreate_collection=True); "
            "print(r)"
            "\""
        )
        ssh(ingest_cmd, check=False)


def step_verify() -> None:
    print("\n" + "=" * 60)
    print("STEP 10: End-to-end verification")
    print("=" * 60)

    print("\n-> Health check:")
    ssh(f"curl -s {BACKEND_URL}/health | python3 -m json.tool 2>/dev/null || echo 'health check failed'", check=False)

    print("\n-> Products list:")
    ssh(
        f"curl -s {BACKEND_URL}/api/products | python3 -c \""
        "import sys,json; d=json.load(sys.stdin); "
        "[print(f'  - {p[\\\"product_name\\\"]} ({p[\\\"chunk_count\\\"]} chunks)') for p in d]"
        "\" 2>/dev/null || echo 'products list failed'",
        check=False,
    )

    print("\n-> Search test (cai bien hien dien):")
    ssh(
        f"curl -s -X POST '{BACKEND_URL}/api/search' "
        f"-H 'Content-Type: application/json' "
        f"-d '{{\"query\": \"cam bien hien dien ket noi nhu the nao\", \"top_k\": 3}}' | "
        "python3 -c \""
        "import sys,json; d=json.load(sys.stdin); "
        "print(f'Found {len(d[\\\"results\\\"])} results, latency: {d[\\\"latency_ms\\\"]:.0f}ms')"
        "\" 2>/dev/null || echo 'search test failed'",
        check=False,
    )

    print("\n-> GPU status:")
    ssh("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU'", check=False)

    print("\n" + "=" * 60)
    print(f"Frontend:  http://{SSH_HOST}:8501")
    print(f"API:       http://{SSH_HOST}:8000")
    print(f"API Docs:  http://{SSH_HOST}:8000/docs")
    print("=" * 60)


def show_logs() -> None:
    print("\n→ Backend logs (last 80 lines):")
    ssh("docker logs vhomenex-backend-v2 --tail 80 2>&1 || echo 'container not found'", check=False)


# ─── MODES ───────────────────────────────────────────────────────────────────

def full_deploy(ingest: bool = True) -> None:
    step_stop_containers()
    step_backup()
    step_prepare_remote_dirs()
    step_upload_project()
    step_upload_documents()
    step_build_containers()
    step_start_containers()
    backend_up = step_wait_backend()
    if ingest:
        if backend_up:
            step_ingest(recreate=True)
        else:
            print("[WARN] Backend not ready — skipping ingest. Run with --ingest-only after backend starts.")
    step_verify()


def ingest_only() -> None:
    step_wait_backend(max_seconds=30)
    step_ingest(recreate=True)
    step_verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Vhomenex Hybrid RAG v2 to Ubuntu server")
    parser.add_argument("--no-ingest", action="store_true", help="Deploy without document ingestion")
    parser.add_argument("--ingest-only", action="store_true", help="Only run ingestion (backend must already be running)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify the running system")
    parser.add_argument("--logs", action="store_true", help="Show backend logs")
    args = parser.parse_args()

    print("=" * 60)
    print("Vhomenex Hybrid RAG v2 — Deployment Script (Windows/SCP)")
    print(f"Target: {SSH_TARGET}:{REMOTE_DIR}")
    print("=" * 60)

    if args.verify_only:
        step_verify()
    elif args.logs:
        show_logs()
    elif args.ingest_only:
        ingest_only()
    elif args.no_ingest:
        full_deploy(ingest=False)
    else:
        full_deploy(ingest=True)
