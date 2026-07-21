import os
import re
import json
import random
import string
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
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


def random_batch_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{timestamp}-{random_str}"


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

    batch_path = None
    batch_expires_at = None
    if len(results) > 1:
        batch_id = random_batch_id()
        batch_expires_at = results[0].get("expires_at")  # all files in a batch share the same expiry
        manifest = {
            "item_ids": [r["item_id"] for r in results],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": batch_expires_at,
        }
        upload_bytes(api, json.dumps(manifest).encode(), f"_batches/{batch_id}.json")
        batch_path = f"/batch/{batch_id}"

    return {"files": results, "batch_path": batch_path, "batch_expires_at": batch_expires_at}


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


@app.get("/batch/{batch_id}", response_class=HTMLResponse)
async def serve_batch(batch_id: str):
    """Renders a simple page listing every file uploaded together in one batch."""
    try:
        manifest_path = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=f"_batches/{batch_id}.json",
            token=HF_TOKEN,
        )
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except EntryNotFoundError:
        return HTMLResponse(status_code=404, content=_batch_html([], not_found=True))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    expires_at = manifest.get("expires_at")
    if expires_at and datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
        return HTMLResponse(status_code=410, content=_batch_html([], expired=True))

    item_ids = manifest.get("item_ids", [])
    files = []
    for item_id in item_ids:
        # if a file has an expiry meta, route through the checked path; otherwise link straight to HF
        link_path = f"/api/file/{item_id}"
        files.append({"item_id": item_id, "link_path": link_path})

    return HTMLResponse(content=_batch_html(files))


def _batch_html(files: list, not_found: bool = False, expired: bool = False) -> str:
    if not_found:
        body = '<p class="empty">Batch not found.</p>'
    elif expired:
        body = '<p class="empty">This batch link has expired.</p>'
    elif not files:
        body = '<p class="empty">This batch is empty.</p>'
    else:
        rows = "\n".join(
            f'''<div class="row">
                <span class="name">{f["item_id"]}</span>
                <a class="open" href="{f["link_path"]}" target="_blank">Open</a>
            </div>'''
            for f in files
        )
        body = f'<div class="list">{rows}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Batch Upload</title>
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  body {{ font-family: 'Lexend', sans-serif; background: #FDFCFB; color: #1A1C19; margin: 0; padding: 2rem 1rem; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #1A1C19; color: #E3E3DC; }} }}
  .wrap {{ max-width: 520px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 1.5rem; }}
  .list {{ display: flex; flex-direction: column; gap: 0.5rem; }}
  .row {{ display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;
          background: #F1F1EA; padding: 0.9rem 1rem; border-radius: 12px; }}
  @media (prefers-color-scheme: dark) {{ .row {{ background: #2D2F2B; }} }}
  .name {{ font-size: 0.85rem; word-break: break-all; }}
  .open {{ flex-shrink: 0; background: #386A20; color: #fff; text-decoration: none;
           font-size: 0.85rem; font-weight: 600; padding: 0.45rem 0.9rem; border-radius: 8px; }}
  @media (prefers-color-scheme: dark) {{ .open {{ background: #9CD67D; color: #0C3900; }} }}
  .empty {{ font-size: 0.9rem; opacity: 0.7; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Batch upload &middot; {len(files)} file{'s' if len(files) != 1 else ''}</h1>
    {body}
  </div>
</body>
</html>"""
