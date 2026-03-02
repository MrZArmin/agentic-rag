import logging
import re
import requests
import fitz  # PyMuPDF
from pathlib import Path
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    PDF_SOURCES, DATA_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_PAGE_LENGTH, SEPARATORS,
)

logger = logging.getLogger(__name__)

def download_pdfs(
    sources: dict | None = None,
    data_dir: Path | None = None,
) -> dict[str, Path]:
    """Download PDFs from configured sources, skip already existing files."""
    sources = sources or PDF_SOURCES
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    file_paths: dict[str, Path] = {}

    for key, info in sources.items():
        filepath = data_dir / f"{key}.pdf"
        file_paths[key] = filepath

        if filepath.exists():
            logger.info("%s már letöltve", info["title"])
            continue

        logger.info("%s letöltése...", info["title"])
        try:
            resp = requests.get(
                info["url"], timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (Academic Research)"},
            )
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            logger.info("%s OK (%.0f KB)", info["title"], len(resp.content) / 1024)
        except Exception as e:
            logger.error("%s letöltési hiba: %s", info["title"], e)
            file_paths.pop(key, None)

    return file_paths

def extract_text_from_pdf(
    filepath: Path,
    source_key: str,
    metadata: dict,
) -> list[Document]:
    """Extract and clean text from a PDF, returning one Document per page."""
    docs: list[Document] = []
    try:
        with fitz.open(str(filepath)) as pdf:
            for page_num in range(len(pdf)):
                text = pdf[page_num].get_text("text")

                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]+", " ", text)
                text = text.strip()

                if len(text) < MIN_PAGE_LENGTH:
                    continue

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": source_key,
                        "title": metadata.get("title", ""),
                        "page": page_num + 1,
                        "total_pages": len(pdf),
                    },
                ))
    except Exception as e:
        logger.error("Hiba %s feldolgozásakor: %s", filepath.name, e)

    return docs


def load_all_documents(pdf_paths: dict[str, Path]) -> list[Document]:
    """Load and extract text from all downloaded PDFs."""
    all_docs: list[Document] = []
    for key, filepath in pdf_paths.items():
        meta = PDF_SOURCES[key]
        page_docs = extract_text_from_pdf(filepath, key, meta)
        all_docs.extend(page_docs)
        logger.info("%s: %d oldal", meta["title"], len(page_docs))
    return all_docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)


def print_chunk_stats(chunks: list[Document]) -> None:
    """Log chunk count, size statistics, and per-source distribution."""
    lens = [len(c.page_content) for c in chunks]
    logger.info("Chunk-olás: %d chunk", len(chunks))
    logger.info("Átlag: %.0f, Min: %d, Max: %d", sum(lens) / len(lens), min(lens), max(lens))

    counts = Counter(c.metadata["source"] for c in chunks)
    logger.info("Forrás szerint:")
    for src, n in counts.most_common():
        title = PDF_SOURCES.get(src, {}).get("title", src)
        logger.info("  %s: %d", title, n)