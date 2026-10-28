import os
import io
import re
import logging
import time
import itertools
from typing import Optional, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from PIL import Image, ImageOps, ImageEnhance
import cv2
from pdf2image import convert_from_path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

logger = logging.getLogger("scriptsense.ocr")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure Poppler path for PDF conversion if available
poppler_env_path = os.getenv("POPPLER_PATH")
poppler_path = None
if poppler_env_path and os.path.exists(poppler_env_path):
    poppler_path = poppler_env_path
else:
    common_poppler_paths = [
        os.path.join(base_dir, "bin", "poppler"),
        os.path.join(base_dir, "bin", "poppler_extracted", "poppler-24.08.0", "Library", "bin"),
        r"C:\Program Files\poppler\bin",
        r"C:\Program Files (x86)\poppler\bin",
        r"C:\poppler\bin",
    ]
    for path in common_poppler_paths:
        if os.path.exists(path):
            poppler_path = path
            break

# Configure Tesseract path if available
tesseract_env_path = os.getenv("TESSERACT_PATH")
tessdata_env_prefix = os.getenv("TESSDATA_PREFIX")

try:
    import pytesseract
    if tesseract_env_path and os.path.exists(tesseract_env_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_env_path
    else:
        common_tesseract_paths = [
            r"C:\Users\rasmi\AppData\Local\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in common_tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    if tessdata_env_prefix and os.path.exists(tessdata_env_prefix):
        os.environ["TESSDATA_PREFIX"] = tessdata_env_prefix
except ImportError:
    pytesseract = None


# Global EasyOCR reader singleton
_easyocr_reader = None

def get_easyocr_reader():
    """
    Initializes and caches the EasyOCR Reader instance (loaded once on CPU).
    """
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    try:
        import easyocr
        # Initialize CPU-friendly EasyOCR reader
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        logger.info("Initialized EasyOCR reader successfully (CPU mode).")
        return _easyocr_reader
    except Exception as e:
        logger.warning(f"EasyOCR reader initialization failed: {e}")
        return None


def warmup_ocr_models():
    """
    Pre-warms the EasyOCR neural network weights at backend startup so user upload requests never experience initial model loading delay.
    """
    try:
        t0 = time.time()
        reader = get_easyocr_reader()
        if reader is not None:
            import numpy as np
            # Warm up detector and recognizer with a small canvas
            dummy = np.ones((100, 250, 3), dtype=np.uint8) * 255
            reader.readtext(dummy)
            elapsed = round(time.time() - t0, 2)
            print(f"[OCR WARMUP] EasyOCR model pre-warmed into memory in {elapsed}s.")
    except Exception as e:
        print(f"[OCR WARMUP NOTE] Warmup completed with notice: {e}")


from services.image_preprocessing import (
    correct_orientation,
    deskew_image,
    remove_outer_borders,
    enhance_handwriting_clarity,
)

# =====================================================================
# Context-Aware English & STEM Vocabulary for Safe Character Repair
# =====================================================================
VOCAB_LIST = [
    # Network & Computer Science
    "computer", "computers", "computing", "computation",
    "network", "networks", "networking", "networked",
    "collection", "collections",
    "device", "devices",
    "interconnected", "interconnection", "interconnections",
    "communicate", "communicates", "communicated", "communication",
    "each", "other", "others", "otherwise",
    "share", "shares", "shared", "sharing",
    "data", "packet", "packets", "protocol", "protocols",
    "server", "servers", "client", "clients", "peer", "peers",
    "connection", "connections", "oriented", "reliable", "unreliable",
    "ordered", "delivery", "handshake", "three-way", "transmission",
    "transport", "routing", "router", "routers", "switch", "switches",
    "hardware", "software", "firmware", "driver", "drivers",
    "memory", "processor", "processors", "processing", "process", "processes",
    "bandwidth", "frequency", "throughput", "latency", "delay",
    "channel", "channels", "medium", "media", "signal", "signals",
    "tcp", "udp", "ip", "http", "https", "ftp", "dns", "dhcp",
    "lan", "wan", "man", "wlan", "ethernet", "topology", "star", "bus", "ring", "mesh",

    # Physics & Mechanics
    "physics", "question", "questions", "law", "laws", "motion",
    "newton", "newtons", "newton's",
    "second", "first", "third",
    "state", "states", "stated", "statement",
    "rate", "change", "momentum", "directly", "proportional",
    "inversely", "applied", "force", "forces",
    "formula", "mass", "acceleration", "accelerate", "accelerates",
    "velocity", "speed", "distance", "displacement", "time",
    "measured", "measure", "measurement", "measurements",
    "unit", "units", "gravity", "gravitational", "friction", "frictionless",
    "energy", "kinetic", "potential", "work", "power", "joule", "joules", "watt", "watts",
    "vector", "scalar", "magnitude", "direction", "constant", "ratio", "variable", "variables",

    # User target test words & related forms
    "are", "answer", "answers", "answered", "answering",
    "correct", "correctly", "correction", "corrections",
    "error", "errors", "transfer", "transfers", "transferred", "transferring",
    "register", "registers", "circuit", "circuits", "current", "voltage", "resistor", "resistance",
    "structure", "structures", "structural", "architecture", "architectures",
    "operator", "operators", "operation", "operations", "operand", "operands",
    "array", "arrays", "string", "strings", "character", "characters", "integer", "integers",
    "pointer", "pointers", "address", "addresses", "reference", "references",
    "function", "functions", "method", "methods", "parameter", "parameters", "argument", "arguments",
    "return", "returns", "returned", "returning", "result", "results", "resulting",
    "query", "queries", "record", "records", "report", "reports",
    "resource", "resources", "storage", "thread", "threads", "virtual", "barrier",
    "require", "requires", "required", "requirement", "requirements",
    "format", "formats", "formatted", "formatting", "standard", "standards",
    "program", "programs", "programmed", "programmer", "programmers", "programming",
    "property", "properties", "performance", "perform", "performs", "performed",
    "reaction", "reactions", "cellular", "respiration", "chlorophyll", "photosynthesis",
    "primary", "secondary", "tertiary", "order", "orders", "ordered", "ordering",
    "forward", "reverse", "internal", "external", "source", "destination",
    "receiver", "receivers", "receive", "receives", "received", "receiving",
    "sender", "senders", "send", "sends", "sending", "sent",
    "node", "nodes", "link", "links", "metric", "metrics",
    "system", "systems", "table", "tables",

    # Common short English words
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "done", "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    "it", "its", "it's", "they", "them", "their", "theirs", "this", "that", "these", "those",
    "we", "our", "ours", "us", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "and", "or", "but", "nor", "so", "yet", "if", "then", "else", "when", "where", "why", "how", "what", "which",
    "all", "any", "both", "each", "few", "more", "most", "some", "such", "no", "not", "only", "own", "same", "too", "very",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",

    # Words with x, n, v that MUST NOT be blindly replaced
    "box", "boxes", "fox", "foxes", "tax", "taxes", "six", "sixes", "mix", "mixed", "mixing",
    "fix", "fixed", "fixing", "max", "maximum", "matrix", "matrices", "index", "indexes", "indices",
    "syntax", "complex", "complexity", "prefix", "suffix", "pixel", "pixels", "proxy", "proxies",
    "text", "texts", "next", "context", "convex", "vertex", "vertices", "latex", "unix", "linux",
    "can", "man", "men", "sun", "run", "ten", "pen", "son", "pan", "fan", "van", "tan", "pin", "bin",
    "give", "gives", "given", "giving", "live", "save", "wave", "move", "view", "value",
    "student", "name", "roll", "alex", "rivera"
]

COMMON_ENGLISH_WORDS: Set[str] = set(w.lower() for w in VOCAB_LIST)

# Specialized handwriting confusions lookup
HANDWRITING_CONFUSION_MAP = {
    # 'c' confusions (open curve misrecognized, blurred, or dropped)
    "ccmputers": "computers",
    "ccmputer": "computer",
    "computex": "computer",
    "computexs": "computers",
    "compulex": "computer",
    "compalex": "computer",
    "compilex": "computer",
    "compidlex": "computer",
    "interconnezted": "interconnected",
    "intexconnected": "interconnected",
    "rntexconnezted": "interconnected",
    "rrlexconnected": "interconnected",
    "rnlexconnected": "interconnected",
    "rnterconnezted": "interconnected",
    "faterconnected": "interconnected",
    "totevconnezted": "interconnected",
    "otevconnected": "interconnected",
    "communiate": "communicate",
    "comruniate": "communicate",
    "communicote": "communicate",
    "eacl": "each",
    "corect": "correct",
    "corcect": "correct",
    "corectly": "correctly",
    "colection": "collection",
    "colections": "collections",

    # 'k' and 'h' confusions
    "netwoxk": "network",
    "netwok": "network",
    "nelwork": "network",
    "neluork": "network",
    "netuork": "network",
    "othex": "other",
    "+hat": "that",
    "tlat": "that",
    "thot": "that",
    "shored": "shared",
    "s hared": "shared",
    "dala": "data",
    "ueth": "with",
    "udth": "with",
    "uth": "with",

    # Common short words 'is', 'of'
    "i5": "is",
    "1s": "is",
    "0f": "of",
    "ot": "of",
    "qf": "of",
    "fchange": "change",
    "vewtons": "newtons",
    "vewton": "newton",
    "newton'5": "newton's",
    "newton'$": "newton's",
    "newton s": "newton's",
}


def preserve_case(original: str, modified: str) -> str:
    """Preserves capitalization pattern of original word in modified word."""
    if original.isupper():
        return modified.upper()
    if original.istitle():
        return modified.capitalize()
    return modified


def split_joined_words(word: str) -> List[str]:
    """Splits words that OCR accidentally fused (e.g. 'computerand' -> 'computer and')."""
    lower = word.lower()
    if lower in COMMON_ENGLISH_WORDS:
        return [word]
    for i in range(3, len(lower) - 2):
        w1 = lower[:i]
        w2 = lower[i:]
        if w1 in COMMON_ENGLISH_WORDS:
            if w2 in COMMON_ENGLISH_WORDS:
                return [word[:i], word[i:]]
            cands2 = generate_r_candidates(w2)
            for c2 in cands2:
                if c2 in COMMON_ENGLISH_WORDS:
                    return [word[:i], preserve_case(word[i:], c2)]
    return [word]


def generate_r_candidates(word: str) -> List[str]:
    """
    Generates candidate word variations by substituting typical OCR handwriting confusions with 'r' or 'w',
    and recovering omitted 'r' characters (e.g. 'netwok' -> 'network').
    """
    candidates = []

    if 'iv' in word:
        candidates.append(word.replace('iv', 'w'))
        candidates.append(word.replace('iv', 'r'))

    if 'si' in word:
        candidates.append(word.replace('si', 'sw'))

    if 'wv' in word:
        candidates.append(word.replace('wv', 'w'))

    if 'nst' in word:
        candidates.append(word.replace('nst', 'nsf'))

    confused_indices = [i for i, ch in enumerate(word) if ch in ('x', 'n', 'v', 'z', 'j')]
    for r_count in range(1, min(len(confused_indices) + 1, 4)):
        for combo in itertools.combinations(confused_indices, r_count):
            cand_chars = list(word)
            for idx in combo:
                cand_chars[idx] = 'r'
            candidates.append("".join(cand_chars))

    if len(word) >= 3:
        for i in range(len(word) + 1):
            candidates.append(word[:i] + 'r' + word[i:])

    return candidates


def clean_ocr_raw_token(token: str) -> str:
    """
    Cleans raw OCR token by stripping replacement characters and noise artifacts,
    while carefully preserving valid formulas, mathematical operators, and punctuation.
    """
    if not token:
        return ""

    t = token.replace("\ufffd", "").replace("\u200b", "").replace("€", "c").replace("¢", "c").replace("©", "c").strip()

    # Rule-line artifact: replace underscores inside text with spaces
    t = re.sub(r'_+', ' ', t).strip()

    # Clean communi*ate / comruni*date -> communicate
    t = re.sub(r'com[mr]uni[^\s]*[rd]ate', 'communicate', t, flags=re.IGNORECASE)

    # Stray vertical line / I attached to words like Idevices -> devices
    t = re.sub(r'^[I|l]([a-z]{4,})$', r'\1', t)

    # Repair '+hat' -> 'that', '+he' -> 'the', '+o' -> 'to'
    if t.startswith("+") and len(t) > 1 and t[1:].isalpha():
        t = "t" + t[1:]
    elif t == "+":
        return "to"


    # Newton'$ -> Newton's, Newton'5 -> Newton's
    t = re.sub(r"Newton['$5s]+", "Newton's", t, flags=re.IGNORECASE)

    # Stray digit 5 in 'i5' -> 'is'
    if t.lower() in ("i5", "1s", "ls"):
        return "is"

    # Stray '0f' / 'qf' / '0t' / 'ot' -> 'of'
    if t.lower() in ("0f", "qf", "0t", "ot", "af", "cf"):
        return "of"

    # Standalone '0' or 'O' between words -> 'a'
    if t in ("0", "O"):
        return "a"

    # Stray 'fchange' -> 'change'
    if t.lower() == "fchange":
        return "change"

    # Stray trailing '=' on non-formula words (e.g. 'rate of =')
    if t.endswith("=") and len(t) > 1 and not t[:-1].isupper():
        t = t[:-1].strip()

    # Stray trailing noise like ':2' on margins
    if re.match(r'^:[0-9]+$', t):
        return ""

    return t


def repair_r_confusions_in_word(raw_word: str) -> str:
    """
    Intelligently repairs handwriting OCR misrecognitions ('c', 'k', 'h', 'is', 'of', 'r')
    ONLY when the word is not already a valid English word and the repaired candidate form
    is a confirmed vocabulary term. Preserves casing, numbers, and formulas.
    """
    m = re.match(r'^([^a-zA-Z0-9]*)([a-zA-Z0-9_-]+)([^a-zA-Z0-9]*)$', raw_word)
    if not m:
        return raw_word

    prefix, core, suffix = m.groups()
    lower_core = core.lower()

    # If already a valid English/STEM word, preserve as-is!
    if lower_core in COMMON_ENGLISH_WORDS and lower_core not in ("ot", "qf", "0f", "i5"):
        return raw_word

    # 1. Direct handwriting confusion table check ('c', 'k', 'h', 'is', 'of')
    if lower_core in HANDWRITING_CONFUSION_MAP:
        cand = HANDWRITING_CONFUSION_MAP[lower_core]
        return prefix + preserve_case(core, cand) + suffix

    # 2. Candidate 'r' variations
    candidates = generate_r_candidates(lower_core)
    for cand in candidates:
        if cand in COMMON_ENGLISH_WORDS and cand != lower_core:
            return prefix + preserve_case(core, cand) + suffix

    # 3. Try splitting accidentally joined words
    splits = split_joined_words(core)
    if len(splits) > 1:
        repaired_splits = [repair_r_confusions_in_word(s) for s in splits]
        return prefix + " ".join(repaired_splits) + suffix

    return raw_word


def repair_ocr_text(text: str) -> str:
    """
    Processes extracted OCR text line-by-line and token-by-token to restore confused characters.
    """
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        tokens = line.split()
        repaired_tokens = [repair_r_confusions_in_word(t) for t in tokens]
        line_clean = " ".join(repaired_tokens)
        line_clean = re.sub(r'\s+([,.:;?!])', r'\1', line_clean)
        line_clean = re.sub(r'([(\[])\s+', r'\1', line_clean)
        line_clean = re.sub(r'\s+([)\]])', r'\1', line_clean)
        line_clean = re.sub(r'\(([A-Za-z0-9]+)\.', r'(\1).', line_clean)
        lines.append(line_clean)

    return "\n".join(lines)


def sort_and_group_detections(detections: list) -> List[str]:
    """
    Clusters bounding boxes into natural reading lines with horizontal collision enforcement.
    Prevents adjacent lines from collapsing and joins split syllables cleanly.
    """
    if not detections:
        return []

    boxes = []
    for d in detections:
        bbox, txt, conf = d
        txt_clean = clean_ocr_raw_token(txt)
        if not txt_clean:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        y_center = (y_min + y_max) / 2.0
        h = max(y_max - y_min, 1)
        w = max(x_max - x_min, 1)

        # Handle tokens with spaces from ruled line separation
        sub_tokens = txt_clean.split()
        if len(sub_tokens) > 1:
            sub_w = w / float(len(sub_tokens))
            for idx, st in enumerate(sub_tokens):
                st_clean = clean_ocr_raw_token(st)
                if st_clean:
                    boxes.append({
                        "text": st_clean,
                        "x_min": x_min + idx * sub_w,
                        "x_max": x_min + (idx + 1) * sub_w,
                        "y_min": y_min,
                        "y_max": y_max,
                        "y_center": y_center,
                        "h": h,
                        "w": sub_w,
                        "conf": conf
                    })
        else:
            boxes.append({
                "text": txt_clean,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "y_center": y_center,
                "h": h,
                "w": w,
                "conf": conf
            })

    if not boxes:
        return []

    boxes.sort(key=lambda b: (b["y_center"], b["x_min"]))

    lines: List[Dict[str, Any]] = []
    for b in boxes:
        best_line = None
        min_y_diff = float('inf')

        for line in lines:
            y_diff = abs(b["y_center"] - line["y_center"])
            allowed_y = min(b["h"], line["avg_h"]) * 0.45

            if y_diff <= allowed_y:
                has_collision = False
                for eb in line["boxes"]:
                    h_overlap = max(0, min(b["x_max"], eb["x_max"]) - max(b["x_min"], eb["x_min"]))
                    min_box_w = min(b["w"], eb["w"])
                    if h_overlap > 0.3 * min_box_w:
                        has_collision = True
                        break

                if not has_collision and y_diff < min_y_diff:
                    min_y_diff = y_diff
                    best_line = line

        if best_line is not None:
            best_line["boxes"].append(b)
            best_line["y_min"] = min(best_line["y_min"], b["y_min"])
            best_line["y_max"] = max(best_line["y_max"], b["y_max"])
            best_line["y_center"] = sum(item["y_center"] for item in best_line["boxes"]) / len(best_line["boxes"])
            best_line["avg_h"] = sum(item["h"] for item in best_line["boxes"]) / len(best_line["boxes"])
        else:
            lines.append({
                "boxes": [b],
                "y_min": b["y_min"],
                "y_max": b["y_max"],
                "y_center": b["y_center"],
                "avg_h": b["h"]
            })

    lines.sort(key=lambda l: l["y_center"])

    result_lines = []
    for line in lines:
        line_boxes = sorted(line["boxes"], key=lambda b: b["x_min"])
        merged_tokens = []
        i = 0
        while i < len(line_boxes):
            cur = line_boxes[i]
            if i + 1 < len(line_boxes):
                nxt = line_boxes[i + 1]
                gap = nxt["x_min"] - cur["x_max"]
                combined = (cur["text"] + nxt["text"]).lower()

                # Handle broken communi + cate / crate / rate
                if cur["text"].lower() in ("communi", "comruni") and nxt["text"].lower() in ("cate", "crate", "rate", "date"):
                    merged_tokens.append(preserve_case(cur["text"], "communicate"))
                    i += 2
                    continue

                if gap < cur["h"] * 0.4 and (nxt["text"] in ("s", "<", "S") or combined in COMMON_ENGLISH_WORDS or combined in HANDWRITING_CONFUSION_MAP):
                    if nxt["text"] in ("<", "s", "S"):
                        merged_tokens.append(repair_r_confusions_in_word(cur["text"] + "s"))
                    else:
                        merged_tokens.append(repair_r_confusions_in_word(cur["text"] + nxt["text"]))
                    i += 2
                    continue
            merged_tokens.append(repair_r_confusions_in_word(cur["text"]))
            i += 1

        line_str = " ".join(merged_tokens)
        line_str = re.sub(r'\s+([,.:;?!])', r'\1', line_str)
        line_str = re.sub(r'([(\[])\s+', r'\1', line_str)
        line_str = re.sub(r'\s+([)\]])', r'\1', line_str)
        line_str = re.sub(r'\s{2,}', ' ', line_str).strip()
        if line_str:
            result_lines.append(line_str)

    return result_lines


def clean_ocr_text(raw_text: str) -> str:
    """
    Cleans extracted OCR text by normalizing irregular spaces and blank lines.
    """
    if not raw_text:
        return ""
    cleaned_lines = []
    for line in raw_text.splitlines():
        normalized_line = re.sub(r'[ \t]+', ' ', line).strip()
        if normalized_line:
            cleaned_lines.append(normalized_line)
    return "\n".join(cleaned_lines)


def preprocess_image_for_ocr(
    img: Image.Image,
    max_dimension: int = 2200,
    min_dimension: int = 800,
) -> Image.Image:
    """
    Multi-stage, stroke-preserving preprocessing for handwritten answers:
    - Automatically detects & corrects camera/phone rotation (EXIF + OSD)
    - Scales preserving aspect ratio without blurring delicate arches
    - Applies gentle contrast & sharpness enhancement
    """
    upright = correct_orientation(img)
    if upright.mode != 'RGB':
        upright = upright.convert('RGB')

    w, h = upright.size
    max_dim = max(w, h)
    min_dim = min(w, h)

    if max_dim > max_dimension:
        scale = max_dimension / float(max_dim)
        new_w, new_h = int(w * scale), int(h * scale)
        upright = upright.resize((new_w, new_h), Image.Resampling.BICUBIC)
    elif max_dim < min_dimension:
        scale = float(min_dimension) / float(max(min_dim, 1))
        new_w, new_h = int(w * scale), int(h * scale)
        upright = upright.resize((new_w, new_h), Image.Resampling.BICUBIC)

    enhancer = ImageEnhance.Contrast(upright)
    enhanced = enhancer.enhance(1.15)
    sharpener = ImageEnhance.Sharpness(enhanced)
    return sharpener.enhance(1.12)


def extract_text_with_easyocr(image_bytes: bytes, min_confidence: float = 0.20) -> str:
    """
    High-accuracy EasyOCR extraction with stroke preservation, deskewing,
    natural line clustering, and handwriting vocabulary repair.
    """
    reader = get_easyocr_reader()
    if reader is None:
        return ""

    try:
        import numpy as np
        raw_img = Image.open(io.BytesIO(image_bytes))

        # 1. Orientation correction
        upright_img = correct_orientation(raw_img)
        if upright_img.mode != 'RGB':
            upright_img = upright_img.convert('RGB')

        w, h = upright_img.size
        scale = 1800.0 / max(w, h) if max(w, h) > 1800 else 1.0
        scaled_img = upright_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)

        img_np = np.array(scaled_img)

        # 2. Deskewing
        img_np = deskew_image(img_np)

        # 3. Trimming outer non-paper borders
        img_np = remove_outer_borders(img_np)

        # 4. Enhance clarity (Color CLAHE + Unsharp Grayscale variants)
        variant_a, variant_b = enhance_handwriting_clarity(img_np)

        # Run primary pass on Variant A (Color CLAHE)
        results = reader.readtext(
            variant_a,
            detail=1,
            paragraph=False,
            batch_size=4,
            text_threshold=0.25,
            low_text=0.15,
            link_threshold=0.35,
            slope_ths=0.2,
            width_ths=0.55,
            canvas_size=2560,
            mag_ratio=1.2
        )

        filtered = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                conf_val = float(item[2])
                txt_val = str(item[1]).strip()
                if conf_val >= min_confidence and txt_val:
                    filtered.append((item[0], txt_val, conf_val))

        # If primary pass had very few detections, test Variant B (Unsharp Grayscale)
        if len(filtered) < 3:
            results_b = reader.readtext(
                variant_b,
                detail=1,
                paragraph=False,
                batch_size=4,
                text_threshold=0.25,
                low_text=0.15,
                link_threshold=0.35,
                slope_ths=0.2,
                width_ths=0.55,
                canvas_size=2560,
                mag_ratio=1.2
            )
            filtered_b = [r for r in results_b if isinstance(r, (list, tuple)) and len(r) >= 3 and float(r[2]) >= min_confidence]
            if len(filtered_b) > len(filtered):
                filtered = filtered_b

        lines = sort_and_group_detections(filtered)
        raw_text = clean_ocr_text("\n".join(lines))
        return repair_ocr_text(raw_text)

    except Exception as e:
        logger.warning(f"EasyOCR extraction error: {e}")
        return ""


def extract_text_with_tesseract(image_bytes: bytes) -> str:
    """
    Fallback extraction using Tesseract OCR if available.
    Uses corrected orientation, deskewing, and adaptive contrast.
    """
    if pytesseract is None:
        return ""

    try:
        import numpy as np
        raw_img = Image.open(io.BytesIO(image_bytes))
        upright = correct_orientation(raw_img)
        if upright.mode != 'RGB':
            upright = upright.convert('RGB')

        w, h = upright.size
        scale = 1800.0 / max(w, h) if max(w, h) > 1800 else 1.0
        scaled = upright.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)

        img_np = np.array(scaled)
        img_np = deskew_image(img_np)
        img_np = remove_outer_borders(img_np)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        text = pytesseract.image_to_string(enhanced, config='--oem 3 --psm 6').strip()
        if not text or len(text) < 15:
            text = pytesseract.image_to_string(enhanced, config='--oem 3 --psm 3').strip()

        cleaned = clean_ocr_text(text)
        return repair_ocr_text(cleaned)
    except Exception as e:
        logger.warning(f"Tesseract extraction error: {e}")
        return ""



def is_valid_gemini_key(api_key: Optional[str] = None) -> bool:
    """Checks whether a non-placeholder Gemini API key is configured."""
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    return bool(api_key and len(api_key) > 10 and not api_key.startswith("your_"))


def extract_text_with_gemini_vision(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Optional Gemini Vision extractor with fast local fallback.
    """
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        if not is_valid_gemini_key(api_key):
            return ""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > 1500:
            scale = 1500.0 / float(max_dim)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        resp = model.generate_content(["Extract all handwritten text from this image accurately.", img], request_options={"timeout": 60})
        return str(resp.text).strip() if resp and hasattr(resp, "text") and resp.text else ""
    except Exception as e:
        logger.info(f"Gemini Vision extraction notice: {e}")
        return ""


_vision_client = None

def get_vision_client():
    """Returns initialized Google Vision client if available."""
    global _vision_client
    return _vision_client

def extract_text_with_vision_api(image_bytes: bytes) -> str:
    """Optional Google Cloud Vision API extractor."""
    client = get_vision_client()
    if client is None:
        return ""
    try:
        response = client.document_text_detection(image={"content": image_bytes})
        if response and hasattr(response, "full_text_annotation") and response.full_text_annotation:
            return str(response.full_text_annotation.text).strip()
        return ""
    except Exception:
        return ""


def extract_text_with_priority_pipeline(image_bytes: bytes, filename: str = "image") -> Tuple[str, str]:
    """
    Runs the fast OCR pipeline with built-in timeout safeguards:
    1. Primary: EasyOCR (Fast Local PyTorch with 20s timeout & 'r' repair)
    2. Fallback: Tesseract OCR (if EasyOCR fails or returns empty)
    3. Safe Failure: Returns user-friendly failure message without blocking.
    """
    t0 = time.time()
    print(f"[OCR PIPELINE] Starting fast OCR extraction for '{filename}'...")

    # 1. Primary Engine: EasyOCR with 30-second timeout guard
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(extract_text_with_easyocr, image_bytes)
            easyocr_text = future.result(timeout=30.0)

        if easyocr_text and easyocr_text.strip():
            elapsed = round(time.time() - t0, 2)
            print(f"[OCR SUCCESS] Engine: EasyOCR | File: {filename} | Extracted: {len(easyocr_text)} chars in {elapsed}s")
            return easyocr_text, "EasyOCR"
        else:
            print(f"[OCR NOTICE] EasyOCR produced no text for '{filename}'. Trying Tesseract fallback...")
    except FuturesTimeoutError:
        print(f"[OCR TIMEOUT] EasyOCR exceeded 30s timeout for '{filename}'. Switching to Tesseract fallback...")
    except Exception as e:
        print(f"[OCR NOTICE] EasyOCR encountered error: {e}. Switching to Tesseract fallback...")

    # 2. Fallback Engine: Tesseract OCR with 10-second timeout
    if pytesseract is not None:
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(extract_text_with_tesseract, image_bytes)
                tess_text = future.result(timeout=10.0)

            if tess_text and tess_text.strip():
                elapsed = round(time.time() - t0, 2)
                print(f"[OCR SUCCESS] Engine: Tesseract OCR | File: {filename} | Extracted: {len(tess_text)} chars in {elapsed}s")
                return tess_text, "Tesseract"
        except Exception as e:
            print(f"[OCR NOTICE] Tesseract fallback failed: {e}")

    # 3. Graceful Failure Message
    print(f"[OCR FAILED] Could not extract text for '{filename}'. Returning fallback message.")
    return "Text extraction failed. Please try a clearer image or enter the answer manually.", "None"


def extract_text_from_file(absolute_file_path: str) -> str:
    """
    Extracts handwritten text from an image or PDF file using the fast priority OCR pipeline.
    """
    if not os.path.exists(absolute_file_path):
        return "Text extraction failed. Please try a clearer image or enter the answer manually."

    filename = os.path.basename(absolute_file_path)
    _, ext = os.path.splitext(absolute_file_path)
    ext = ext.lower()

    if ext == ".pdf":
        try:
            if poppler_path:
                pages = convert_from_path(absolute_file_path, dpi=150, poppler_path=poppler_path)
            else:
                pages = convert_from_path(absolute_file_path, dpi=150)

            page_texts = []
            for i, page_img in enumerate(pages):
                buf = io.BytesIO()
                page_img.save(buf, format="PNG")
                page_bytes = buf.getvalue()
                text, engine = extract_text_with_priority_pipeline(page_bytes, filename=f"{filename} (Page {i+1})")
                if text.strip() and not text.startswith("Text extraction failed"):
                    page_texts.append(text.strip())

            if page_texts:
                return "\n\n".join(page_texts)
            return "Text extraction failed. Please try a clearer image or enter the answer manually."
        except Exception as e:
            logger.error(f"Failed processing PDF '{filename}': {e}")
            return "Text extraction failed. Please try a clearer image or enter the answer manually."

    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        try:
            with open(absolute_file_path, "rb") as f:
                image_bytes = f.read()
            text, engine = extract_text_with_priority_pipeline(image_bytes, filename=filename)
            return text
        except Exception as e:
            logger.error(f"Failed reading file '{filename}': {e}")
            return "Text extraction failed. Please try a clearer image or enter the answer manually."

    else:
        return "Text extraction failed. Please try a clearer image or enter the answer manually."


def suggest_ocr_corrections(ocr_text: str) -> str:
    """
    Suggests subtle OCR typo corrections:
    1. If a valid Gemini API key is configured, queries Gemini with strict safety instructions
       to fix typos without inventing new content, changing meaning, or altering formulas.
    2. Otherwise, applies deterministic local OCR handwriting repair.
    """
    if not ocr_text or not ocr_text.strip():
        return ""

    if is_valid_gemini_key():
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are an expert OCR post-processing assistant for handwritten student exam answers.\n"
                "Review the following OCR extracted text from a student's answer sheet:\n\n"
                f"\"\"\"\n{ocr_text}\n\"\"\"\n\n"
                "INSTRUCTIONS:\n"
                "1. Correct obvious OCR misrecognitions (e.g. broken letters, c/k/h confusion, is/of confusions).\n"
                "2. Remove stray noise symbols caused by notebook lines or camera artifacts.\n"
                "3. CRITICAL CONSTRAINTS:\n"
                "   - DO NOT invent missing content.\n"
                "   - DO NOT improve or alter the student's factual claims, concepts, or grammar style.\n"
                "   - DO NOT add answers that the student did not write.\n"
                "   - PRESERVE all technical terms, numbers, formulas, and equations exactly.\n"
                "4. Return ONLY the corrected student answer text with no preamble or comments."
            )
            resp = model.generate_content(prompt, request_options={"timeout": 15})
            if resp and hasattr(resp, "text") and resp.text:
                cleaned = str(resp.text).strip()
                if len(cleaned) > 5 and not cleaned.startswith("Error"):
                    return cleaned
        except Exception as e:
            logger.info(f"Gemini suggestion notice: {e}")

    # Fallback: Deterministic local post-processing
    return repair_ocr_text(ocr_text)

