import os
import random
import string
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from huggingface_hub import HfApi

# Fix for Vercel's read-only file system (os error 30)
os.environ["HF_HOME"] = "/tmp"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Create FastAPI app for Vercel
app = FastAPI(docs_url=None, redoc_url=None)

# Settings
DATASET_REPO = "notamitgamer/uploads" 
HF_TOKEN = os.getenv("HF_TOKEN")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), custom_id: str = Form(None)):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN environment variable is not set")

    try:
        ext = os.path.splitext(file.filename)[1]

        # Generate unique code if not provided
        if not custom_id:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
            item_id = f"{timestamp}-{random_str}{ext}"
        else:
            item_id = custom_id
            if ext and not item_id.endswith(ext):
                item_id += ext

        # Read the file directly into server RAM (bytes)
        content = await file.read()
        
        api = HfApi()
        
        # NEW: Automatically create the dataset on Hugging Face if it doesn't exist
        api.create_repo(
            repo_id=DATASET_REPO, 
            repo_type="dataset", 
            exist_ok=True, 
            token=HF_TOKEN,
            private=False # Ensure it is public so your links work instantly
        )

        # Push directly from RAM to Hugging Face Dataset (Bypassing disk entirely)
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=item_id,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        return {"item_id": item_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
