import os
import re
import json
import random
import string
from datetime import datetime, timezone
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
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
    """Turn a user-supplied custom_id into a safe repo path segment."""
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

# ---------------------------------------------------------------------------
# Upload a single file
# ---------------------------------------------------------------------------

def handle_one_upload(
    api: HfApi,
    filename: str,
    content: bytes,
    custom_id: Optional[str]
) -> dict:
    ext = os.path.splitext(filename)[1]
    item_id = sanitize_id(custom_id, ext) if custom_id else random_id(ext)
    
    upload_bytes(api, content, item_id)
    
    # Since expiry is removed, all links just use the root static redirect
    link_path = f"/{item_id}"
    
    return {
        "item_id": item_id,
        "link_path": link_path,
        "size": len(content),
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    custom_id: Optional[str] = Form(None)
):
    api = get_api()
    ensure_repo(api)

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
            result = handle_one_upload(api, f.filename or "file", content, custom_id)
            results.append(result)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload '{f.filename}': {e}")

    batch_path = None
    if len(results) > 1:
        batch_id = random_batch_id()
        manifest = {
            "item_ids": [r["item_id"] for r in results],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        upload_bytes(api, json.dumps(manifest).encode(), f"_batches/{batch_id}.json")
        batch_path = f"/batch/{batch_id}"

    return {"files": results, "batch_path": batch_path}

@app.post("/api/upload-url")
async def upload_from_url(
    url: str = Form(...),
    custom_id: Optional[str] = Form(None)
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
        result = handle_one_upload(api, filename, content, custom_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"files": [result]}

@app.get("/batch/{batch_id}", response_class=HTMLResponse)
async def serve_batch(batch_id: str):
    """Renders a simple page listing every file uploaded together in one batch."""
    try:
        manifest_path = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=f"_batches/{batch_id}.json",
            token=HF_TOKEN,
            cache_dir="/tmp"  # Explicitly force /tmp to avoid Vercel Errno 30
        )
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except EntryNotFoundError:
        return HTMLResponse(status_code=404, content=_batch_html([], not_found=True))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    item_ids = manifest.get("item_ids", [])
    files = [{"item_id": i, "link_path": f"/{i}"} for i in item_ids]

    return HTMLResponse(content=_batch_html(files))

def _batch_html(files: list, not_found: bool = False) -> str:
    if not_found:
        body = '<p class="empty">Batch not found.</p>'
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
