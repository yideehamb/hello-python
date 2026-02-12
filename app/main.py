"""
Invoice PDF parsing service.

POST /parse/invoice — accepts a PDF via multipart/form-data,
extracts text with pdfplumber, and returns structured invoice data.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
import uuid
from datetime import date
from typing import Optional

import pdfplumber
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("invoice_parser")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("INVOICE_API_KEY", "dev-key-change-me")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "20"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Invoice Parser API",
    version="1.0.0",
    description="Extracts structured data from invoice PDFs.",
)
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Invoice Parser API",
        "docs": "/docs"
    }
# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(_api_key_header)):
    if api_key is None or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# Per-request usage logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def usage_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()

    logger.info(
        "req_start request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "req_end request_id=%s status=%s elapsed_ms=%s",
        request_id,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Response models (OpenAPI schema)
# ---------------------------------------------------------------------------
class LineItem(BaseModel):
    description: Optional[str] = Field(None, examples=["Widget A"])
    quantity: Optional[float] = Field(None, examples=[2.0])
    unit_price: Optional[float] = Field(None, examples=[9.99])
    line_total: Optional[float] = Field(None, examples=[19.98])


class InvoiceResponse(BaseModel):
    vendor_name: Optional[str] = Field(None, examples=["Acme Corp"])
    invoice_number: Optional[str] = Field(None, examples=["INV-2024-001"])
    invoice_date: Optional[str] = Field(
        None,
        description="ISO 8601 date string",
        examples=["2024-06-15"],
    )
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = Field(None, examples=[199.80])
    tax: Optional[float] = Field(None, examples=[17.98])
    total: Optional[float] = Field(None, examples=[217.78])
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="0 = no useful data extracted, 1 = all fields found",
        examples=[0.75],
    )


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    """Return concatenated text from every page of a PDF."""
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Defensive parsing helpers
# ---------------------------------------------------------------------------
_MONEY_RE = re.compile(r"[\$€£]?\s*([\d,]+\.?\d*)")


def _parse_money(raw: Optional[str]) -> Optional[float]:
    """Try to pull a numeric value from a money-like string. Returns None on failure."""
    if raw is None:
        return None
    m = _MONEY_RE.search(raw.replace(",", ""))
    if m:
        try:
            return round(float(m.group(1)), 2)
        except ValueError:
            return None
    return None


def _safe_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", raw)
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


# Date patterns: "Jan 1, 2024", "01/15/2024", "2024-01-15", "15 January 2024", etc.
_DATE_PATTERNS: list[tuple[str, str]] = [
    # ISO
    (r"(\d{4})-(\d{1,2})-(\d{1,2})", "ymd"),
    # US  MM/DD/YYYY  or  MM-DD-YYYY
    (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "mdy"),
    # "Jan 15, 2024" / "January 15, 2024"
    (
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})",
        "Mdy",
    ),
    # "15 January 2024"
    (
        r"(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(\d{4})",
        "dMy",
    ),
]

_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_date(text: str) -> Optional[str]:
    """Return the first plausible date found in *text* as an ISO-8601 string, or None."""
    for pattern, kind in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        try:
            if kind == "ymd":
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif kind == "mdy":
                d = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
            elif kind == "Mdy":
                month = _MONTH_MAP[m.group(1).lower()]
                d = date(int(m.group(3)), month, int(m.group(2)))
            elif kind == "dMy":
                month = _MONTH_MAP[m.group(2).lower()]
                d = date(int(m.group(3)), month, int(m.group(1)))
            else:
                continue
            return d.isoformat()
        except (ValueError, KeyError):
            continue
    return None


def _search(text: str, *patterns: str) -> Optional[str]:
    """Return the first regex-group match for any of the supplied patterns, or None."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Line-item extraction
# ---------------------------------------------------------------------------
_LINE_ITEM_RE = re.compile(
    r"^(.+?)\s+"           # description (non-greedy)
    r"(\d+(?:\.\d+)?)\s+"  # quantity
    r"[\$€£]?\s*([\d,]+\.\d{2})\s+"  # unit price
    r"[\$€£]?\s*([\d,]+\.\d{2})$",   # line total
    re.MULTILINE,
)


def _extract_line_items(text: str) -> list[LineItem]:
    items: list[LineItem] = []
    for m in _LINE_ITEM_RE.finditer(text):
        items.append(
            LineItem(
                description=m.group(1).strip() or None,
                quantity=_safe_float(m.group(2)),
                unit_price=_safe_float(m.group(3)),
                line_total=_safe_float(m.group(4)),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Main parsing orchestrator
# ---------------------------------------------------------------------------
_TOTAL_FIELDS = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "line_items",
    "subtotal",
    "tax",
    "total",
]


def parse_invoice_text(text: str) -> InvoiceResponse:
    """Best-effort structured extraction from raw invoice text."""

    vendor_name = _search(
        text,
        r"(?:sold\s+by|vendor|from|company)[:\s]+(.+)",
        r"^([A-Z][A-Za-z &.,]+\s+(?:Inc|LLC|Ltd|Corp|Co)\.?)$",
        r"^([A-Z][A-Za-z &.,]+(?:Inc|LLC|Ltd|Corp|Co)\.?)$",
    )

    invoice_number = _search(
        text,
        r"(?:invoice\s*#|inv\s*#)[:\s]*([A-Za-z0-9\-]+)",
        r"(?:invoice\s+(?:number|no))[:\s]+([A-Za-z0-9\-]+)",
    )

    invoice_date_str: Optional[str] = None
    date_region = _search(
        text,
        r"(?:invoice\s+date|date\s+of\s+invoice|date)[:\s]*(.+)",
    )
    if date_region:
        invoice_date_str = _parse_date(date_region)
    if invoice_date_str is None:
        invoice_date_str = _parse_date(text)

    line_items = _extract_line_items(text)

    subtotal_raw = _search(text, r"(?:subtotal|sub\s+total)[:\s]*([\$€£\d,. ]+)")
    tax_raw = _search(text, r"(?:tax|vat|gst|hst)[:\s]*([\$€£\d,. ]+)")
    total_raw = _search(
        text,
        r"(?:total\s+due|amount\s+due|grand\s+total)[:\s]*([\$€£\d,. ]+)",
        r"(?:^|(?<=\s))total[:\s]*([\$€£\d,. ]+)",
    )

    subtotal = _parse_money(subtotal_raw)
    tax = _parse_money(tax_raw)
    total = _parse_money(total_raw)

    # Confidence: proportion of top-level fields that are non-null / non-empty.
    found = 0
    for field in _TOTAL_FIELDS:
        val = locals().get(field)
        if val is not None and val != [] and val != "":
            found += 1
    confidence = round(found / len(_TOTAL_FIELDS), 2)

    return InvoiceResponse(
        vendor_name=vendor_name,
        invoice_number=invoice_number,
        invoice_date=invoice_date_str,
        line_items=line_items,
        subtotal=subtotal,
        tax=tax,
        total=total,
        confidence_score=confidence,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/parse/invoice",
    response_model=InvoiceResponse,
    summary="Parse an invoice PDF",
    dependencies=[Depends(verify_api_key)],
)
async def parse_invoice(file: UploadFile = File(..., description="Invoice PDF")):
    # Validate content type
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Expected application/pdf.",
        )

    # Read + size guard
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Write to temp file for pdfplumber (requires seekable file)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()

        try:
            text = extract_text_from_pdf(tmp.name)
        except Exception:
            logger.exception("Failed to extract text from PDF")
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from the uploaded PDF.",
            )

    if not text.strip():
        return InvoiceResponse(confidence_score=0.0)

    logger.info("Extracted %d characters from PDF", len(text))
    return parse_invoice_text(text)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}
