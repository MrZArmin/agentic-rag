import logging
from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import GRADER_DOC_PREVIEW_CHARS
from src.llm import get_llm_precise, get_llm_creative

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    generation: str
    documents: list[Document]
    retrieval_count: int
    generation_count: int
    route_decision: str
    hallucination_check: str


def make_initial_state(question: str) -> AgentState:
    """Create a blank agent state for a new question."""
    return AgentState(
        question=question,
        generation="",
        documents=[],
        retrieval_count=0,
        generation_count=0,
        route_decision="",
        hallucination_check="",
    )


def _build_router_chain():
    return (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a binary classifier. Classify the user's message.\n\n"
             "Output EXACTLY one word:\n"
             "- retrieve = question about AI, ML, NLP, transformers, RAG, research\n"
             "- direct = greeting, chitchat, off-topic, math, personal question\n\n"
             "Examples:\n"
             "User: How does attention work? → retrieve\n"
             "User: Hello! → direct\n"
             "User: What is corrective RAG? → retrieve\n"
             "User: What is your name? → direct\n"
             "User: What is 2+2? → direct\n"
             "User: How do agents improve retrieval? → retrieve\n"
             "User: Good morning → direct\n\n"
             "Output ONLY one word, no explanation."),
            ("human", "{question}"),
        ])
        | get_llm_precise()
        | StrOutputParser()
    )


def _build_grader_chain():
    return (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a document relevance grader.\n"
             "Output ONLY \"relevant\" if the document helps answer the question.\n"
             "Output ONLY \"not_relevant\" otherwise.\n"
             "One word only."),
            ("human",
             "Question: {question}\n\nDocument:\n{document}\n\nRelevant?"),
        ])
        | get_llm_precise()
        | StrOutputParser()
    )


def _build_rewriter_chain():
    return (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a query rewriter for an AI research paper RAG system.\n"
             "Rewrite the question to improve retrieval. Use specific technical terms "
             "(transformer, attention mechanism, retrieval-augmented, chain-of-thought, etc.).\n"
             "Output ONLY the rewritten question."),
            ("human", "Original: {question}\n\nRewritten:"),
        ])
        | get_llm_precise()
        | StrOutputParser()
    )


def _build_generator_chain():
    return (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a research assistant that answers questions based on AI papers.\n\n"
             "Rules:\n"
             "1. Answer ONLY based on the provided context.\n"
             "2. Reference which paper the information comes from.\n"
             "3. If context is insufficient, say so explicitly.\n"
             "4. Be concise: 2-4 sentences unless more detail is needed."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ])
        | get_llm_creative()
        | StrOutputParser()
    )


def _build_direct_chain():
    return (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. The question doesn't need paper retrieval. "
             "Respond briefly. Mention you can help with AI research paper questions."),
            ("human", "{question}"),
        ])
        | get_llm_creative()
        | StrOutputParser()
    )


def _build_hallucination_chain():
    return (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a hallucination checker.\n"
             "Output ONLY \"grounded\" if the answer is supported by the documents.\n"
             "Output ONLY \"not_grounded\" if it contains unsupported claims.\n"
             "One word only."),
            ("human",
             "Documents:\n{documents}\n\nAnswer:\n{generation}\n\nGrounded?"),
        ])
        | get_llm_precise()
        | StrOutputParser()
    )


_chains: dict = {}


def _get_chain(name: str):
    if name not in _chains:
        builders = {
            "router": _build_router_chain,
            "grader": _build_grader_chain,
            "rewriter": _build_rewriter_chain,
            "generator": _build_generator_chain,
            "direct": _build_direct_chain,
            "hallucination": _build_hallucination_chain,
        }
        _chains[name] = builders[name]()
    return _chains[name]


VALID_ROUTES = {"retrieve", "direct"}


def route_question(state: AgentState) -> AgentState:
    """Classify question as 'retrieve' or 'direct' using the router LLM."""
    try:
        raw = _get_chain("router").invoke({"question": state["question"]}).strip().lower()
        logger.debug("Router raw output: '%s'", raw)

        first_word = raw.split()[0].strip(".,!?:;\"'") if raw.split() else ""
        decision = first_word if first_word in VALID_ROUTES else "retrieve"
    except Exception as e:
        logger.warning("Router hiba, fallback to retrieve: %s", e)
        decision = "retrieve"

    logger.info("Router: %s", decision)
    return {**state, "route_decision": decision}


def make_retrieve_node(retriever):
    """Create a retrieval node bound to the given retriever."""
    def retrieve_documents(state: AgentState) -> AgentState:
        try:
            docs = retriever.invoke(state["question"])
        except Exception as e:
            logger.error("Retriever hiba: %s", e)
            docs = []
        count = state.get("retrieval_count", 0) + 1
        logger.info("Retriever: %d doc (próba #%d)", len(docs), count)
        return {**state, "documents": docs, "retrieval_count": count}
    return retrieve_documents


def grade_documents(state: AgentState) -> AgentState:
    """Filter retrieved documents by LLM-based relevance grading."""
    relevant = []
    for i, doc in enumerate(state["documents"]):
        preview = doc.page_content[:GRADER_DOC_PREVIEW_CHARS]
        try:
            raw = _get_chain("grader").invoke({
                "question": state["question"], "document": preview,
            }).strip().lower()
            keep = raw.split()[0].strip(".,!?:;\"'") == "relevant" if raw.split() else False
        except Exception as e:
            logger.warning("Grader hiba doc [%d]: %s, megtartjuk", i + 1, e)
            keep = True
        logger.debug("Doc [%d]: %s", i + 1, "relevant" if keep else "filtered")
        if keep:
            relevant.append(doc)
    logger.info("Grader: %d/%d releváns", len(relevant), len(state["documents"]))
    return {**state, "documents": relevant}


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval results."""
    try:
        rewritten = _get_chain("rewriter").invoke({"question": state["question"]}).strip()
    except Exception as e:
        logger.warning("Rewriter hiba, eredeti kérdés marad: %s", e)
        rewritten = state["question"]
    logger.info("Rewriter: '%s' → '%s'", state["question"], rewritten)
    return {**state, "question": rewritten}


def generate_response(state: AgentState) -> AgentState:
    """Generate an answer from retrieved documents."""
    parts = []
    for doc in state["documents"]:
        src = doc.metadata.get("title", "?")
        pg = doc.metadata.get("page", "?")
        parts.append(f"[{src}, p.{pg}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(parts) or "No relevant documents found."

    gen_count = state.get("generation_count", 0) + 1
    try:
        response = _get_chain("generator").invoke({
            "context": context, "question": state["question"],
        })
    except Exception as e:
        logger.error("Generator hiba: %s", e)
        response = "Sajnálom, nem sikerült választ generálni. Kérlek próbáld újra."
    logger.info("Generator (#%d): %s...", gen_count, response[:150])
    return {**state, "generation": response, "generation_count": gen_count}


def generate_direct_response(state: AgentState) -> AgentState:
    """Respond without retrieval (greetings, off-topic)."""
    try:
        response = _get_chain("direct").invoke({"question": state["question"]})
    except Exception as e:
        logger.error("Direct generator hiba: %s", e)
        response = "Sajnálom, nem sikerült választ generálni."
    logger.info("Direct: %s...", response[:150])
    return {**state, "generation": response}


def check_hallucination(state: AgentState) -> AgentState:
    """Check whether the generated answer is grounded in the documents."""
    doc_texts = "\n\n".join(d.page_content for d in state["documents"])
    try:
        raw = _get_chain("hallucination").invoke({
            "documents": doc_texts, "generation": state["generation"],
        }).strip().lower()
        first_word = raw.split()[0].strip(".,!?:;\"'") if raw.split() else ""
        grounded = first_word == "grounded"
    except Exception as e:
        logger.warning("Hallucination check hiba, feltételezzük grounded: %s", e)
        grounded = True
    label = "grounded" if grounded else "not_grounded"
    logger.info("Hallucination: %s (%s)", "OK" if grounded else "NOT OK", label)
    return {**state, "hallucination_check": label}
