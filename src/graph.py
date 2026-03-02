import time
from langgraph.graph import StateGraph, END, START
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import MAX_RETRIEVAL_RETRIES, MAX_GENERATION_RETRIES
from src.nodes import (
    AgentState, make_initial_state,
    route_question, make_retrieve_node, grade_documents,
    rewrite_query, generate_response, generate_direct_response,
    check_hallucination,
)


def build_graph(retriever: VectorStoreRetriever) -> StateGraph:
    wf = StateGraph(AgentState)

    # Nodes
    wf.add_node("router", route_question)
    wf.add_node("retrieve", make_retrieve_node(retriever))
    wf.add_node("grade_docs", grade_documents)
    wf.add_node("rewrite_query", rewrite_query)
    wf.add_node("generate", generate_response)
    wf.add_node("direct_generate", generate_direct_response)
    wf.add_node("check_hallucination", check_hallucination)

    def _router_edge(state: AgentState) -> str:
        return state.get("route_decision", "retrieve")

    def _grading_edge(state: AgentState) -> str:
        if len(state.get("documents", [])) > 0:
            return "generate"
        if state.get("retrieval_count", 0) < MAX_RETRIEVAL_RETRIES:
            return "rewrite"
        return "generate"

    def _hallucination_edge(state: AgentState) -> str:
        if state.get("hallucination_check") == "grounded":
            return "end"
        if state.get("generation_count", 0) < MAX_GENERATION_RETRIES + 1:
            return "regenerate"
        return "end"

    wf.add_conditional_edges("router", _router_edge, {
        "retrieve": "retrieve", "direct": "direct_generate",
    })
    wf.add_conditional_edges("grade_docs", _grading_edge, {
        "generate": "generate", "rewrite": "rewrite_query",
    })
    wf.add_conditional_edges("check_hallucination", _hallucination_edge, {
        "end": END, "regenerate": "generate",
    })

    wf.add_edge(START, "router")
    wf.add_edge("retrieve", "grade_docs")
    wf.add_edge("rewrite_query", "retrieve")
    wf.add_edge("generate", "check_hallucination")
    wf.add_edge("direct_generate", END)

    return wf.compile()

def run_agent(app, question: str, *, verbose: bool = True) -> dict:
    if verbose:
        print(f"{'='*70}")
        print(f"{question}")
        print(f"{'='*70}")

    result = app.invoke(make_initial_state(question))

    if verbose:
        print(f"\n{'─'*70}")
        print(f"VÁLASZ:\n{result.get('generation', '—')}")
        print(f"{'─'*70}")
        print(f"Route={result.get('route_decision')}, "
              f"Retrieval×{result.get('retrieval_count', 0)}, "
              f"Generate×{result.get('generation_count', 0)}, "
              f"Docs={len(result.get('documents', []))}")
        print(f"{'='*70}\n")

    return result


def measure_latency(app, question: str, n_runs: int = 3) -> dict:
    times = []
    for i in range(n_runs):
        t0 = time.time()
        run_agent(app, question, verbose=False)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.2f}s")
    return {
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }