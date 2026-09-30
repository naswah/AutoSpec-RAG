import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _path(env_key, *default_parts):
    value = os.environ.get(env_key)
    return value if value else str(BASE_DIR.joinpath(*default_parts))


PDF_PATH = _path("AUTOSPEC_PDF_PATH", "local", "samples", "test.pdf")
OUTPUT_BASE = _path("AUTOSPEC_OUTPUT_BASE", "local", "runs")
RESULTS = _path("AUTOSPEC_RESULTS", "local", "results")
MASTERFORMAT_CSV = _path(
    "AUTOSPEC_MASTERFORMAT_CSV", "data", "masterformat", "masterformat_2018.csv"
)
CHUNKS_PATH = _path("AUTOSPEC_CHUNKS_PATH", "local", "debug")
SCALE_PATH = _path("AUTOSPEC_SCALE_PATH", "local", "scale")

# Tesseract lives outside the repo; default to the standard Windows install
# location and let AUTOSPEC_OCR_PATH override it (e.g. /usr/bin/tesseract on Linux).
OCR_PATH = os.environ.get(
    "AUTOSPEC_OCR_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)