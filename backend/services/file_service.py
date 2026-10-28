import os
import shutil
from typing import Tuple, Any
from fastapi import HTTPException

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}
UPLOAD_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")

# Ensure uploads directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

def validate_and_save_file(file: Any, prefix_id: str = "") -> Tuple[str, str]:
    """
    Validates uploaded file against allowed formats and stores it securely in uploads/.
    Returns (relative_file_path, absolute_file_path).
    """
    filename = file.filename or "upload.png"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Fallback to content-type if extension missing
    if not ext and hasattr(file, "content_type") and file.content_type:
        mime = str(file.content_type).lower()
        if "jpeg" in mime or "jpg" in mime:
            ext = ".jpg"
        elif "png" in mime:
            ext = ".png"
        elif "pdf" in mime:
            ext = ".pdf"
        elif "webp" in mime:
            ext = ".webp"

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: JPG, JPEG, PNG, PDF."
        )

    # Ensure file pointer is at the beginning
    if hasattr(file, "file") and hasattr(file.file, "seek"):
        file.file.seek(0)

    # Sanitize and create distinct filename
    clean_name = os.path.basename(filename).replace(" ", "_")
    if not os.path.splitext(clean_name)[1]:
        clean_name += ext

    if prefix_id:
        target_filename = f"{prefix_id}_{clean_name}"
    else:
        target_filename = f"{int(os.path.getmtime(UPLOAD_DIRECTORY) if os.path.exists(UPLOAD_DIRECTORY) else 1)}_{clean_name}"

    dest_path = os.path.join(UPLOAD_DIRECTORY, target_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Also keep direct clean_name copy available in uploads/ for direct filename lookups
    direct_path = os.path.join(UPLOAD_DIRECTORY, clean_name)
    if dest_path != direct_path:
        try:
            shutil.copyfile(dest_path, direct_path)
        except Exception:
            pass

    relative_path = os.path.join("uploads", target_filename).replace("\\", "/")
    return relative_path, dest_path
