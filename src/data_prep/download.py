"""Download News Commentary v18 CES-DEU parallel corpus."""

import gzip
import shutil
import urllib.request
from pathlib import Path

from src.config import RAW_DIR

DOWNLOAD_URL = "https://data.statmt.org/news-commentary/v18/training/news-commentary-v18.cs-de.tsv.gz"
RAW_GZ = RAW_DIR / "news-commentary-v18.cs-de.tsv.gz"
RAW_TSV = RAW_DIR / "news-commentary-v18.cs-de.tsv"


def download_corpus(force: bool = False) -> Path:
    """Download and extract News Commentary v18 cs-de."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_TSV.exists() and not force:
        print(f"Already extracted: {RAW_TSV}")
        return RAW_TSV

    if not RAW_GZ.exists() or force:
        print(f"Downloading {DOWNLOAD_URL} ...")
        urllib.request.urlretrieve(DOWNLOAD_URL, RAW_GZ)
        print(f"Saved to {RAW_GZ}")

    print("Extracting ...")
    with gzip.open(RAW_GZ, "rb") as f_in, open(RAW_TSV, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Extracted to {RAW_TSV}")
    return RAW_TSV


if __name__ == "__main__":
    download_corpus()
