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

def download_pdfs(
    sources: dict | None = None,
    data_dir: Path | None = None,
) -> dict[str, Path]:
    sources = sources or PDF_SOURCES
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    file_paths: dict[str, Path] = {}

    for key, info in sources.items():
        filepath = data_dir / f"{key}.pdf"
        file_paths[key] = filepath

        if filepath.exists():
            print(f"{info['title']} már letöltve")
            continue

        print(f"{info['title']}...", end=" ")
        try:
            resp = requests.get(
                info["url"], timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (Academic Research)"},
            )
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            print(f"OK ({len(resp.content) / 1024:.0f} KB)")
        except Exception as e:
            print(f"Hiba: {e}")
            file_paths.pop(key, None)

    return file_paths

def extract_text_from_pdf(
    filepath: Path,
    source_key: str,
    metadata: dict,
) -> list[Document]:
    docs: list[Document] = []
    try:
        pdf = fitz.open(str(filepath))
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
        pdf.close()
    except Exception as e:
        print(f"Hiba {filepath.name} feldolgozásakor: {e}")

    return docs


def load_all_documents(pdf_paths: dict[str, Path]) -> list[Document]:
    all_docs: list[Document] = []
    for key, filepath in pdf_paths.items():
        meta = PDF_SOURCES[key]
        page_docs = extract_text_from_pdf(filepath, key, meta)
        all_docs.extend(page_docs)
        print(f"{meta['title']}: {len(page_docs)} oldal")
    return all_docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)


def print_chunk_stats(chunks: list[Document]) -> None:
    lens = [len(c.page_content) for c in chunks]
    print(f"Chunk-olás: {len(chunks)} chunk")
    print(f"Átlag: {sum(lens)/len(lens):.0f}, Min: {min(lens)}, Max: {max(lens)}")

    counts = Counter(c.metadata["source"] for c in chunks)
    print("Forrás szerint:")
    for src, n in counts.most_common():
        title = PDF_SOURCES.get(src, {}).get("title", src)
        print(f"{title}: {n}")