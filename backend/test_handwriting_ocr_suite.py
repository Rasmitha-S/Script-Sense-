import os
import sys
import io
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

backend_dir = r"c:\Users\rasmi\OneDrive\Desktop\ScriptSense - Copy\backend"
sys.path.insert(0, backend_dir)

from services.ocr_service import extract_text_from_file

uploads_dir = os.path.join(backend_dir, "uploads")

def generate_handwritten_style_image(
    text_lines: list,
    tilt_angle: float = 0.0,
    light_ink: bool = False,
    font_size: int = 24
) -> bytes:
    """Generates an answer sheet image with variable styling for rigorous OCR evaluation."""
    img_w, img_h = 1000, 80 + len(text_lines) * 65
    canvas = Image.new("RGB", (img_w, img_h), color=(252, 250, 246)) # Off-white notebook paper
    draw = ImageDraw.Draw(canvas)

    # Draw faint notebook ruled lines
    for y in range(60, img_h, 60):
        draw.line([(30, y), (img_w - 30, y)], fill=(225, 232, 245), width=1)
    # Red left margin line
    draw.line([(80, 20), (80, img_h - 20)], fill=(245, 205, 205), width=2)

    # Ink color (black/navy blue, lighter if light_ink is True)
    ink_color = (130, 140, 155) if light_ink else (20, 30, 50)

    # Try default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    y_pos = 75
    for line in text_lines:
        draw.text((105, y_pos), line, fill=ink_color, font=font)
        y_pos += 60

    if abs(tilt_angle) > 0.1:
        canvas = canvas.rotate(tilt_angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(252, 250, 246))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()

def run_tests():
    print("=" * 80)
    print("        COMPREHENSIVE OCR ACCURACY EVALUATION & BENCHMARK SUITE         ")
    print("=" * 80)

    test_cases = [
        {
            "name": "Case 1: Actual Physics Answer Sheet (31_alex_rivera_physics_answer.png)",
            "file": os.path.join(uploads_dir, "31_alex_rivera_physics_answer.png"),
            "target_text": (
                "Student Name: Alex Rivera (Roll: CS2026-0101)\n"
                "Physics Question 1: Newton's Second Law of Motion\n"
                "Newton's second law states that the rate of change of\n"
                "momentum is directly proportional to applied force\n"
                "Formula: Force = mass x acceleration (F = m *a)\n"
                "Force is measured in Newtons (N)."
            ),
            "expected_keywords": ["rate of change of", "Newton's", "force", "acceleration", "Newtons (N)"]
        },
        {
            "name": "Case 2: Actual WhatsApp Answer Sheet (113_WhatsApp_Image...jpeg)",
            "file": os.path.join(uploads_dir, "113_WhatsApp_Image_2026-08-21_at_6.40.46_AM_(1).jpeg"),
            "target_text": (
                "Computer network is a collection of\n"
                "interconnected computers and devices\n"
                "that communicate with each other to\n"
                "share the data"
            ),
            "expected_keywords": ["computer", "network", "collection of", "interconnected", "communicate", "each other", "data"]
        },
        {
            "name": "Case 3: Actual Phone Camera Photo (221_IMG_20260821_060538.jpg - 90 deg sideways)",
            "file": os.path.join(uploads_dir, "221_IMG_20260821_060538.jpg"),
            "target_text": (
                "Computer Network is a Collection of\n"
                "Interconnected Computers and devices\n"
                "that Communicate with each other to\n"
                "share data"
            ),
            "expected_keywords": ["Computer", "Network", "Collection", "Interconnected", "Computers", "devices", "Communicate", "each other"]
        },
        {
            "name": "Case 4: Slightly Tilted Handwriting (-3.5 degrees)",
            "bytes": generate_handwritten_style_image([
                "Computer network is a collection of interconnected devices",
                "that communicate with each other to share data.",
                "It ensures reliable and ordered packet delivery."
            ], tilt_angle=-3.5),
            "filename": "tilted_test.png",
            "expected_keywords": ["computer", "network", "collection", "communicate", "each", "other", "reliable"]
        },
        {
            "name": "Case 5: Light Handwriting with Thin Strokes (c, k, h test)",
            "bytes": generate_handwritten_style_image([
                "A computer network is a collection of interconnected hosts.",
                "Each device can check the link with packets of data.",
                "The protocol is designed for error correction and feedback."
            ], light_ink=True, font_size=22),
            "filename": "light_ink_test.png",
            "expected_keywords": ["computer", "collection", "each", "check", "link", "packets", "correction"]
        },
        {
            "name": "Case 6: Technical STEM Answer (Formulas, Variables, Units)",
            "bytes": generate_handwritten_style_image([
                "Kinetic energy formula: KE = 0.5 * m * v^2",
                "Work done is defined as force multiplied by displacement: W = F * d",
                "The rate of work done is power, measured in Watts (W)."
            ], font_size=26),
            "filename": "stem_technical_test.png",
            "expected_keywords": ["kinetic", "formula", "KE", "0.5", "force", "power", "Watts"]
        }
    ]

    total_tests = len(test_cases)
    passed_tests = 0

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/{total_tests}] {tc['name']}")
        print("-" * 75)

        t0 = time.time()
        if "file" in tc:
            extracted = extract_text_from_file(tc["file"])
        else:
            temp_path = os.path.join(uploads_dir, f"test_{tc['filename']}")
            with open(temp_path, "wb") as f:
                f.write(tc["bytes"])
            extracted = extract_text_from_file(temp_path)
            try:
                os.remove(temp_path)
            except Exception:
                pass

        elapsed = round(time.time() - t0, 2)
        print(f"Extraction Time: {elapsed}s")
        print("EXTRACTED TEXT:\n" + extracted)

        # Keyword checks
        extracted_lower = extracted.lower()
        matched_kw = [kw for kw in tc["expected_keywords"] if kw.lower() in extracted_lower]
        kw_ratio = len(matched_kw) / float(len(tc["expected_keywords"]))
        print(f"Matched Keywords: {len(matched_kw)}/{len(tc['expected_keywords'])} ({kw_ratio*100:.1f}%) -> {matched_kw}")

        # Check issues targeted by user:
        # 1. 'c' present
        has_c = 'c' in extracted_lower
        # 2. 'is' or 'of' present
        has_is_of = 'is' in extracted_lower or 'of' in extracted_lower
        # 3. no stray replacement chars
        clean_symbols = '\ufffd' not in extracted and 'qf' not in extracted and 'fchange' not in extracted

        status_pass = kw_ratio >= 0.70 and clean_symbols
        if status_pass:
            print(">>> EVALUATION: [PASSED]")
            passed_tests += 1
        else:
            print(">>> EVALUATION: [FLAGGED]")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_tests}/{total_tests} Tests Passed Successfully.")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
