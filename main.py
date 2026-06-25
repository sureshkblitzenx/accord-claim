"""
FastAPI Scanned PDF OCR Extractor
Multiple endpoints with different OCR strategies
"""
import pdfplumber
from pypdf import PdfReader
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, time
from pathlib import Path

# ── OCR / PDF helpers (imported lazily to keep startup fast) ──────────────────
import fitz                         # PyMuPDF  – rasterise pages

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
#pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"
                  # Tesseract OCR  – api1 (basic)
from PIL import Image               # Pillow
import io, base64, json
import cv2                          # OpenCV  – api2 (pre-processed)
import numpy as np
import easyocr                      # EasyOCR  – api3 (deep-learning OCR)
import pdfplumber                   # api4 – native text layer fallback
from pdf2image import convert_from_path  # high-res page images

from parser.text_parser import extract_acord_fields


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Scanned PDF OCR API",
    description=(
        "Extract text & data from scanned PDFs using four different OCR strategies.\n\n"
        "| Endpoint | Engine | Best for |\n"
        "|---|---|---|\n"
        "| `/scanned_file/api1` | Tesseract (raw) | Quick, general-purpose |\n"
        "| `/scanned_file/api2` | Tesseract + OpenCV pre-processing | Low-quality / noisy scans |\n"
        "| `/scanned_file/api3` | EasyOCR (deep-learning) | Multi-language, handwriting |\n"
        "| `/scanned_file/api4` | pdfplumber + Tesseract hybrid | Mixed native+scanned PDFs |\n"
    ),
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware



# Allow all origins (Development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EasyOCR reader initialised once (expensive)
_easy_reader = None

def get_easy_reader():
    global _easy_reader
    if _easy_reader is None:
        _easy_reader = easyocr.Reader(["en"], gpu=False)
    return _easy_reader


# ── Shared helpers ────────────────────────────────────────────────────────────

def save_upload(upload: UploadFile) -> str:
    """Save an UploadFile to a temp file and return its path."""
    suffix = Path(upload.filename or "upload.pdf").suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name



def pdf_to_pil_images(pdf_path, dpi=200):
    return convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=r"/usr/bin" #r"C:\Users\SureshKannan\projects\acord\poppler-26.02.0\Library\bin"    #r"/usr/bin"
    )


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Classic OpenCV pipeline that improves Tesseract accuracy on noisy scans:
    grayscale → denoise → adaptive threshold → deskew
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2,
    )
    # Simple deskew via moments
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        if abs(angle) > 0.5:
            (h, w) = binary.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
    return binary


def build_response(pages: list[dict], method: str, elapsed: float, extra: dict | None = None) -> dict:
    full_text = "\n\n".join(p["text"] for p in pages)
    resp = {
        "method": method,
        "total_pages": len(pages),
        "processing_time_seconds": round(elapsed, 3),
        "full_text": full_text,
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "pages": pages,
    }
    if extra:
        resp.update(extra)
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# API 1 – Tesseract RAW (no pre-processing)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/scanned_file/api1",
    summary="OCR via Tesseract (raw)",
    tags=["OCR Endpoints"],
    response_description="Extracted text per page using vanilla Tesseract OCR",
)
async def ocr_tesseract_raw(file: UploadFile = File(..., description="Scanned PDF file")):
    """
    **Strategy:** Convert each page to a high-res image, then run
    Tesseract OCR directly with no image pre-processing.
 
    - ✅ Fast, simple
    - ✅ Good for clean, high-contrast scans
    - ❌ Struggles with skewed, noisy, or low-res scans
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
 
    pdf_path = save_upload(file)
    t0 = time.time()
    all_text =[]
    try:
        images = pdf_to_pil_images(pdf_path, dpi=200)
        pages = []
        for i, img in enumerate(images, start=1):
            raw = pytesseract.image_to_string(img, config="--psm 6")
            all_text.append(raw.strip())
            data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
            avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0
            pages.append({
                "page": i,
                "text": raw.strip(),
                "avg_confidence": avg_conf,
                "word_count": len(raw.split()),
            })
        fields = extract_acord_fields("\n\n".join(all_text))
        return JSONResponse(fields)
        #return JSONResponse(content=build_response(pages, "tesseract_raw", time.time() - t0))
    finally:
        os.unlink(pdf_path)
 
# ═══════════════════════════════════════════════════════════════════════════════
# API 2 – Tesseract + OpenCV Pre-processing
# ═══════════════════════════════════════════════════════════════════════════════

def is_scanned_pdf(pdf_path: str) -> bool:
    """
    Detect whether a PDF is scanned (image-based) or normal (has a text layer).
    Strategy: try to extract text from the first few pages using pdfplumber.
    If total extracted text is very short (< 50 chars), assume it's scanned.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            sample_pages = pdf.pages[:3]  # check first 3 pages
            total_text = ""
            for page in sample_pages:
                text = page.extract_text() or ""
                total_text += text.strip()
        return len(total_text) < 50  # scanned PDFs yield near-zero extractable text
    except Exception:
        return True  # if we can't read it, fall back to OCR path


def extract_text_from_normal_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text and metadata from a native (non-scanned) PDF using pdfplumber.
    Returns a list of page dicts matching the scanned PDF output structure.
    """
    pages = []
    all_text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            all_text_parts.append(text)
            word_count = len(text.split()) if text else 0
            pages.append({
                "page": i,
                "text": text,
                "avg_confidence": 100.0,  # native text layer = perfect confidence
                "word_count": word_count,
            })

    return pages, "\n\n".join(all_text_parts)


@app.post(
    "/scanned_file/api5",
    summary="OCR via Tesseract (raw) with normal PDF fallback",
    tags=["OCR Endpoints"],
    response_description="Extracted text per page — uses Tesseract for scanned PDFs, pdfplumber for native PDFs",
)
async def ocr_tesseract_raw(file: UploadFile = File(..., description="Scanned or native PDF file")):
    """
    **Strategy:** Auto-detects whether the uploaded PDF is scanned or native.

    - For **scanned PDFs**: converts each page to a high-res image and runs Tesseract OCR.
    - For **native PDFs**: extracts the text layer directly using pdfplumber (faster, more accurate).

    - ✅ Handles both scanned and native PDFs automatically
    - ✅ Fast path for native PDFs (no OCR needed)
    - ❌ Scanned path still struggles with skewed, noisy, or very low-res scans
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_path = save_upload(file)
    t0 = time.time()
    all_text = []

    try:
        scanned = is_scanned_pdf(pdf_path)

        if scanned:
            # --- Existing OCR path for scanned PDFs ---
            images = pdf_to_pil_images(pdf_path, dpi=200)
            pages = []
            for i, img in enumerate(images, start=1):
                raw = pytesseract.image_to_string(img, config="--psm 6")
                all_text.append(raw.strip())
                data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
                avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0
                pages.append({
                    "page": i,
                    "text": raw.strip(),
                    "avg_confidence": avg_conf,
                    "word_count": len(raw.split()),
                })
        else:
            # --- New native text extraction path ---
            pages, combined_text = extract_text_from_normal_pdf(pdf_path)

            all_text = [p["text"] for p in pages]
            
            print(all_text)

        fields = extract_acord_fields("\n\n".join(all_text))
        print(all_text)
        return JSONResponse(fields)

    finally:
        os.unlink(pdf_path)


@app.post(
    "/scanned_file/api2",
    summary="OCR via Tesseract + OpenCV pre-processing",
    tags=["OCR Endpoints"],
    response_description="Extracted text after denoising, binarisation, and deskewing",
)
async def ocr_tesseract_preprocessed(file: UploadFile = File(..., description="Scanned PDF file")):
    """
    **Strategy:** Apply an OpenCV pipeline (grayscale → denoise →
    adaptive threshold → deskew) before feeding pages to Tesseract.

    - ✅ Best for low-quality, noisy, or slightly skewed scans
    - ✅ Significantly improves accuracy over raw Tesseract on bad scans
    - ❌ Slightly slower due to image processing
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_path = save_upload(file)
    t0 = time.time()
    try:
        images = pdf_to_pil_images(pdf_path, dpi=300)   # higher DPI for preprocessing
        pages = []
        for i, img in enumerate(images, start=1):
            cv_img = pil_to_cv2(img)
            processed = preprocess_for_ocr(cv_img)
            pil_proc = Image.fromarray(processed)
            raw = pytesseract.image_to_string(pil_proc, config="--psm 6 --oem 3")
            data = pytesseract.image_to_data(pil_proc, config="--psm 6", output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
            avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0
            pages.append({
                "page": i,
                "text": raw.strip(),
                "avg_confidence": avg_conf,
                "word_count": len(raw.split()),
                "preprocessing_applied": ["grayscale", "denoise", "adaptive_threshold", "deskew"],
            })
        return JSONResponse(content=build_response(pages, "tesseract_opencv_preprocessed", time.time() - t0))
    finally:
        os.unlink(pdf_path)


# ═══════════════════════════════════════════════════════════════════════════════
# API 3 – EasyOCR (deep-learning, multi-language)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/scanned_file/api3",
    summary="OCR via EasyOCR (deep-learning)",
    tags=["OCR Endpoints"],
    response_description="Extracted text with bounding boxes from EasyOCR",
)
async def ocr_easyocr(file: UploadFile = File(..., description="Scanned PDF file")):
    """
    **Strategy:** Use EasyOCR (CRAFT + CRNN deep-learning pipeline) for
    text detection and recognition.  Returns bounding-box coordinates
    and per-word confidence scores alongside the full text.

    - ✅ Excellent on handwritten text, irregular fonts, and distorted scans
    - ✅ Multi-language support (configured for English here)
    - ✅ Returns bounding boxes for every detected word
    - ❌ Slower than Tesseract; first call downloads model weights
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_path = save_upload(file)
    t0 = time.time()
    try:
        pages, combined_text = extract_text_from_normal_pdf(pdf_path)
        all_text = [p["text"] for p in pages]
        fields = extract_acord_fields("\n\n".join(all_text))
        return JSONResponse(fields)
    finally:
        os.unlink(pdf_path)


# ═══════════════════════════════════════════════════════════════════════════════
# API 4 – Hybrid (pdfplumber native text → Tesseract fallback per page)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/scanned_file/api4",
    summary="Hybrid OCR (pdfplumber + Tesseract fallback)",
    tags=["OCR Endpoints"],
    response_description="Native text where available, OCR fallback for scanned pages",
)
async def ocr_hybrid(file: UploadFile = File(..., description="Scanned PDF or mixed PDF")):
    """
    **Strategy:** Try to extract native text from each page using
    **pdfplumber** first.  If a page returns fewer than 20 characters
    (likely a scanned/image page), fall back to Tesseract OCR.

    - ✅ Optimal for **mixed PDFs** (some pages are searchable, others scanned)
    - ✅ Very fast on pages that already have a text layer
    - ✅ Returns `extraction_method` per page so you know what happened
    - ❌ Tesseract fallback has no OpenCV pre-processing (use api2 for noisy scans)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_path = save_upload(file)
    t0 = time.time()
    try:
        pages = []
        rasterised_images: dict[int, Image.Image] = {}   # lazy

        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                native_text = page.extract_text() or ""
                tables = page.extract_tables()

                if len(native_text.strip()) >= 20:
                    # ── Native text layer is usable ──────────────────────────
                    table_data = []
                    for tbl in (tables or []):
                        table_data.append([[cell or "" for cell in row] for row in tbl])
                    pages.append({
                        "page": i,
                        "extraction_method": "native_text_layer",
                        "text": native_text.strip(),
                        "word_count": len(native_text.split()),
                        "tables_found": len(table_data),
                        "tables": table_data,
                    })
                else:
                    # ── Fall back to Tesseract OCR ───────────────────────────
                    if i not in rasterised_images:
                        imgs = pdf_to_pil_images(pdf_path, dpi=200)
                        for j, im in enumerate(imgs, start=1):
                            rasterised_images[j] = im
                    img = rasterised_images.get(i)
                    ocr_text = pytesseract.image_to_string(img, config="--psm 6") if img else ""
                    pages.append({
                        "page": i,
                        "extraction_method": "tesseract_ocr_fallback",
                        "text": ocr_text.strip(),
                        "word_count": len(ocr_text.split()),
                        "tables_found": 0,
                        "tables": [],
                    })

        native_count = sum(1 for p in pages if p["extraction_method"] == "native_text_layer")
        ocr_count    = sum(1 for p in pages if p["extraction_method"] == "tesseract_ocr_fallback")
        return JSONResponse(content=build_response(
            pages, "hybrid_pdfplumber_tesseract", time.time() - t0,
            extra={
                "summary": {
                    "total_pages": total,
                    "pages_with_native_text": native_count,
                    "pages_ocr_fallback": ocr_count,
                }
            }
        ))
    finally:
        os.unlink(pdf_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/health", tags=["Utility"])
async def health():
    return {"status": "ok", "version": "1.0.0"}

from fastapi.middleware.cors import CORSMiddleware



# Allow all origins (Development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)