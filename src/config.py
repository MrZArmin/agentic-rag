
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

base_dir = Path(__file__).parent.parent
config_path = base_dir / "config/config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)


LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")
HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
HF_MODEL: str = os.getenv("HF_MODEL", "")


# embedding
EMBEDDING_MODEL: str = config["embedding"]["model_name"]
EMBEDDING_DEVICE: str = config["embedding"]["device"]
EMBEDDING_NORMALIZE: bool = config["embedding"]["normalize"]

# document processing
CHUNK_SIZE: int = config["document_processing"]["chunk_size"]
CHUNK_OVERLAP: int = config["document_processing"]["chunk_overlap"]
MIN_PAGE_LENGTH: int = config["document_processing"]["min_page_length"]
SEPARATORS: list[str] = config["document_processing"]["separators"]

# retrieval
TOP_K: int = config["retrieval"]["top_k"]
SEARCH_TYPE: str = config["retrieval"]["search_type"]

# agent
MAX_RETRIEVAL_RETRIES: int = config["agent"]["max_retrieval_retries"]
MAX_GENERATION_RETRIES: int = config["agent"]["max_generation_retries"]
GRADER_DOC_PREVIEW_CHARS: int = config["agent"]["grader_doc_preview_chars"]
LLM_TEMP_PRECISE: float = config["agent"]["llm_temperature_precise"]
LLM_TEMP_CREATIVE: float = config["agent"]["llm_temperature_creative"]

# paths (base_dir-hez relatív, így CWD-független)
DATA_DIR: Path = base_dir / config["paths"]["data_dir"]
CHROMA_DIR: Path = base_dir / config["paths"]["chroma_dir"]

# PDF
PDF_SOURCES: dict = config["pdf_sources"]


def print_config() -> None:
    print("Konfiguráció betöltve:")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model:    {OLLAMA_MODEL if LLM_PROVIDER == 'ollama' else HF_MODEL}")
    print(f"Embedding:    {EMBEDDING_MODEL}")
    print(f"Chunk:        {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
    print(f"Top-K:        {TOP_K}")
    print(f"PDF források: {len(PDF_SOURCES)} dokumentum")
    print(f"HF Token:     {'van' if HF_API_TOKEN else 'nincs'}")