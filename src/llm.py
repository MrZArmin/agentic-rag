from langchain_core.language_models import BaseChatModel

from src.config import (
    LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_BASE_URL,
    HF_API_TOKEN, HF_MODEL,
    LLM_TEMP_PRECISE, LLM_TEMP_CREATIVE,
)


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )

    if LLM_PROVIDER == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(
            repo_id=HF_MODEL,
            huggingfacehub_api_token=HF_API_TOKEN,
            temperature=temperature,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=llm)

    raise ValueError(f"Ismeretlen LLM provider: {LLM_PROVIDER}")


_llm_precise: BaseChatModel | None = None
_llm_creative: BaseChatModel | None = None


def get_llm_precise() -> BaseChatModel:
    global _llm_precise
    if _llm_precise is None:
        _llm_precise = get_llm(LLM_TEMP_PRECISE)
    return _llm_precise


def get_llm_creative() -> BaseChatModel:
    global _llm_creative
    if _llm_creative is None:
        _llm_creative = get_llm(LLM_TEMP_CREATIVE)
    return _llm_creative


def test_llm() -> None:
    print(f"LLM teszt ({LLM_PROVIDER})...")
    resp = get_llm_precise().invoke("Say 'Hello, working!' and nothing else.")
    text = resp.content if hasattr(resp, "content") else str(resp)
    print(f"Válasz: {text.strip()}")
