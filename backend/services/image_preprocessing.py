import io
from typing import Union, Tuple
import numpy as np
from PIL import Image
import cv2

def preprocess_image_for_cnn(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    target_min_dim: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Image preprocessing pipeline for handwriting recognition:
    Input image -> Grayscale -> Noise reduction -> Contrast enhancement -> Thresholding -> Clean Binary Image.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (enhanced_grayscale_image, binary_thresholded_image)
    """
    # 1. Load image to numpy array
    if isinstance(image_input, str):
        with Image.open(image_input) as pil_img:
            img_np = np.array(pil_img)
    elif isinstance(image_input, bytes):
        with Image.open(io.BytesIO(image_input)) as pil_img:
            img_np = np.array(pil_img)
    elif isinstance(image_input, Image.Image):
        img_np = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        img_np = image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # 2. Convert to Grayscale
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 4:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # 3. Resize / Upscale low-resolution images for fine stroke segmentation
    h, w = gray.shape[:2]
    if w < target_min_dim or h < target_min_dim:
        scale = max(1.5, float(target_min_dim) / max(w, h, 1))
        scale = min(scale, 2.5)
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 4. Noise Reduction: Bilateral filter preserves sharp ink edges while smoothing paper background
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Contrast Enhancement: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 6. Thresholding: Otsu binarization with inverted threshold (ink = 255, background = 0)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 7. Morphological noise clean
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

    return enhanced, binary


def format_line_for_crnn(
    line_crop: np.ndarray,
    target_height: int = 32,
    target_width: int = 384,
) -> np.ndarray:
    """
    Standardizes a segmented text line crop into a CNN+BiLSTM ready input tensor:
    - Automatically crops to active ink strokes
    - Scales height to target_height (32px) while preserving width aspect ratio
    - Centers/pads into normalized (32, target_width) float32 array in [0.0, 1.0].
    
    Returns:
        np.ndarray of shape (target_height, target_width) with float32 values in [0.0, 1.0].
    """
    if line_crop.size == 0 or np.sum(line_crop) == 0:
        return np.zeros((target_height, target_width), dtype=np.float32)

    # Auto-crop to active non-zero stroke pixels
    non_zeros = np.argwhere(line_crop > 20)
    if non_zeros.size > 0:
        y_min, x_min = non_zeros.min(axis=0)
        y_max, x_max = non_zeros.max(axis=0) + 1
        crop = line_crop[y_min:y_max, x_min:x_max]
    else:
        crop = line_crop

    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return np.zeros((target_height, target_width), dtype=np.float32)

    # Scale preserving height
    scale_h = float(target_height - 6) / max(ch, 1)
    new_w = min(int(round(cw * scale_h)), target_width - 8)
    new_h = int(round(ch * scale_h))

    resized = cv2.resize(crop, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA)

    patch = np.zeros((target_height, target_width), dtype=np.float32)
    start_y = (target_height - new_h) // 2
    start_x = 4
    patch[start_y:start_y+new_h, start_x:start_x+new_w] = resized.astype(np.float32) / 255.0

    return patch


def normalize_character_patch(
    canvas: np.ndarray,
    target_size: Tuple[int, int] = (28, 28)
) -> np.ndarray:
    """Standardizes a 2D character canvas into normalized target_size float32 array in [0.0, 1.0]."""
    if canvas.size == 0 or np.sum(canvas) == 0:
        return np.zeros(target_size, dtype=np.float32)
    non_zeros = np.argwhere(canvas > 20)
    if non_zeros.size > 0:
        y_min, x_min = non_zeros.min(axis=0)
        y_max, x_max = non_zeros.max(axis=0) + 1
        crop = canvas[y_min:y_max, x_min:x_max]
    else:
        crop = canvas
    ch, cw = crop.shape[:2]
    tw, th = target_size
    if ch == 0 or cw == 0:
        return np.zeros(target_size, dtype=np.float32)
    scale = float(min(tw, th) - 8) / max(ch, cw, 1)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    patch = np.zeros(target_size, dtype=np.float32)
    sy = (th - new_h) // 2
    sx = (tw - new_w) // 2
    patch[sy:sy+new_h, sx:sx+new_w] = resized.astype(np.float32) / 255.0
    return patch


# =====================================================================
# Advanced Image Preprocessing for Accurate Handwritten OCR
# =====================================================================

import os
from PIL import ImageOps, ImageEnhance

# Configure pytesseract path if available for OSD orientation
try:
    import pytesseract
    tesseract_env_path = os.getenv("TESSERACT_PATH")
    if tesseract_env_path and os.path.exists(tesseract_env_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_env_path
    else:
        for p in [
            r"C:\Users\rasmi\AppData\Local\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
except ImportError:
    pytesseract = None


def correct_orientation(pil_img: Image.Image) -> Image.Image:
    """
    Detects and corrects camera / phone photo rotation:
    1. Reads EXIF metadata (ImageOps.exif_transpose)
    2. Runs fast Orientation Script Detection (OSD) on downsampled preview
    3. Rotates 90, 180, or 270 degrees if text was captured sideways or inverted.
    """
    # 1. EXIF orientation
    transposed = ImageOps.exif_transpose(pil_img)
    if transposed.mode != 'RGB':
        transposed = transposed.convert('RGB')

    # 2. Fast OSD check using a low-res thumbnail (< 0.5s)
    if pytesseract is not None:
        try:
            w, h = transposed.size
            scale = 800.0 / max(w, h)
            preview = transposed.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            osd = pytesseract.image_to_osd(preview)
            rotate_angle = 0
            for line in osd.splitlines():
                if line.startswith("Rotate:"):
                    rotate_angle = int(line.split(":")[1].strip())
                    break

            if rotate_angle == 90:
                transposed = transposed.rotate(270, expand=True)
            elif rotate_angle == 180:
                transposed = transposed.rotate(180, expand=True)
            elif rotate_angle == 270:
                transposed = transposed.rotate(90, expand=True)
        except Exception:
            pass

    return transposed


def deskew_image(img_cv: np.ndarray) -> np.ndarray:
    """
    Detects subtle tilt angle (-15 to +15 deg) in handwritten lines and deskews to horizontal.
    Uses gentle affine rotation so ink strokes remain sharp.
    """
    if img_cv is None or img_cv.size == 0:
        return img_cv

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use horizontal morphology to connect letter clusters into line hulls
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue
        rect = cv2.minAreaRect(cnt)
        w_r, h_r = rect[1]
        if w_r < 40 or h_r < 10:
            continue
        angle = rect[2]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        if -15 < angle < 15:
            angles.append(angle)

    if len(angles) >= 2:
        median_angle = float(np.median(angles))
        if abs(median_angle) > 0.5:
            h, w = img_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            deskewed = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed

    return img_cv


def remove_outer_borders(img_cv: np.ndarray) -> np.ndarray:
    """
    Trims dark table shadows or non-paper boundaries around photographed notebook pages
    while preserving page margins and text content.
    """
    if img_cv is None or img_cv.size == 0:
        return img_cv

    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
    center_mean = np.mean(gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)])

    if center_mean > 120:
        mask = (gray > 65).astype(np.uint8) * 255
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            if bw > w * 0.75 and bh > h * 0.75:
                px = max(0, x - 15)
                py = max(0, y - 15)
                pw = min(w - px, bw + 30)
                ph = min(h - py, bh + 30)
                if pw < w or ph < h:
                    return img_cv[py:py+ph, px:px+pw]
    return img_cv


def enhance_handwriting_clarity(img_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produces two non-destructive, stroke-preserving variants for handwriting OCR:
    - Variant A (enhanced_rgb): LAB CLAHE illumination equalization (preserves color & fine ink)
    - Variant B (unsharp_gray): Subtle unsharp masking (sharpens light pencil/pen strokes)
    Avoids aggressive binarization that removes delicate curves of 'c', 'e', 'k', 'h'.
    """
    if len(img_cv.shape) == 3:
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_rgb = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv
        enhanced_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gray, 1.4, gaussian, -0.4, 0)
    enhanced_gray = cv2.cvtColor(unsharp, cv2.COLOR_GRAY2RGB)

    return enhanced_rgb, enhanced_gray

