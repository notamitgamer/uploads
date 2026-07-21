import os
import re
import json
import random
import string
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# Fix for Vercel's read-only file system (os error 30)
os.environ["HF_HOME"] = "/tmp"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

app = FastAPI(docs_url=None, redoc_url=None)

DATASET_REPO = "notamitgamer/uploads"
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_UPLOAD_BYTES = int(4.4 * 1024 * 1024)  # stay under Vercel's 4.5MB body limit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_id(raw: str, ext: str) -> str:
    """Turn a user-supplied custom_id into a safe repo path segment.

    - strips anything that isn't alnum, dot, dash, underscore
    - removes leading dots/dashes (no path traversal, no dotfiles)
    - enforces a sane max length
    - re-appends the original extension if missing
    """
    raw = raw.strip()
    raw = raw.replace("/", "-").replace("\\", "-")
    raw = _SAFE_ID_RE.sub("", raw)
    raw = raw.lstrip(".-")
    if not raw:
        raise HTTPException(status_code=400, detail="custom_id is invalid after sanitization")
    raw = raw[:80]
    if ext and not raw.endswith(ext):
        raw += ext
    return raw


def random_id(ext: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_str = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"{timestamp}-{random_str}{ext}"


def get_api() -> HfApi:
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN environment variable is not set")
    return HfApi()


def ensure_repo(api: HfApi):
    api.create_repo(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        exist_ok=True,
        token=HF_TOKEN,
        private=False,
    )


def upload_bytes(api: HfApi, content: bytes, path_in_repo: str):
    api.upload_file(
        path_or_fileobj=content,
        path_in_repo=path_in_repo,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )


def write_meta(api: HfApi, item_id: str, expires_at: Optional[str]):
    if not expires_at:
        return
    meta = {"expires_at": expires_at}
    upload_bytes(api, json.dumps(meta).encode(), f"{item_id}.meta.json")


def raw_url(item_id: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{item_id}"


# ---------------------------------------------------------------------------
# Upload a single file (used by /api/upload for each item)
# ---------------------------------------------------------------------------

def handle_one_upload(
    api: HfApi,
    filename: str,
    content: bytes,
    custom_id: Optional[str],
    expires_in_minutes: Optional[int],
) -> dict:
    ext = os.path.splitext(filename)[1]
    item_id = sanitize_id(custom_id, ext) if custom_id else random_id(ext)

    expires_at = None
    if expires_in_minutes and expires_in_minutes > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)).isoformat()

    upload_bytes(api, content, item_id)
    write_meta(api, item_id, expires_at)

    link_path = f"/api/file/{item_id}" if expires_at else f"/{item_id}"
    return {
        "item_id": item_id,
        "link_path": link_path,
        "expires_at": expires_at,
        "size": len(content),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    custom_id: Optional[str] = Form(None),
    expires_in_minutes: Optional[int] = Form(None),
):
    api = get_api()
    ensure_repo(api)

    # custom_id only makes sense for a single file
    if custom_id and len(files) > 1:
        raise HTTPException(status_code=400, detail="custom_id can only be used when uploading a single file")

    results = []
    for f in files:
        content = await f.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"'{f.filename}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB request limit",
            )
        try:
            result = handle_one_upload(api, f.filename or "file", content, custom_id, expires_in_minutes)
            results.append(result)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload '{f.filename}': {e}")

    return {"files": results}


@app.post("/api/upload-url")
async def upload_from_url(
    url: str = Form(...),
    custom_id: Optional[str] = Form(None),
    expires_in_minutes: Optional[int] = Form(None),
):
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Only http/https URLs are supported")

    api = get_api()
    ensure_repo(api)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Remote file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
        )

    filename = url.split("/")[-1].split("?")[0] or "file"

    try:
        result = handle_one_upload(api, filename, content, custom_id, expires_in_minutes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"files": [result]}


@app.get("/api/file/{item_id:path}")
async def serve_expiring_file(item_id: str):
    """Dynamic redirect for links that were uploaded with an expiry."""
    try:
        meta_path = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=f"{item_id}.meta.json",
            token=HF_TOKEN,
        )
        with open(meta_path) as fh:
            meta = json.load(fh)
        expires_at = meta.get("expires_at")
        if expires_at and datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
            return JSONResponse(status_code=410, content={"detail": "This link has expired."})
    except EntryNotFoundError:
        pass  # no meta file -> treat as non-expiring
    except Exception:
        pass  # if metadata can't be checked, fail open rather than block a valid file

    return RedirectResponse(url=raw_url(item_id), status_code=302)


@app.get("/api/files")
async def list_files():
    api = get_api()
    try:
        entries = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    files = []
    for name in entries:
        if name.endswith(".meta.json") or name == ".gitattributes":
            continue
        files.append({"item_id": name, "url": raw_url(name)})

    files.sort(key=lambda x: x["item_id"], reverse=True)
    return {"files": files}
