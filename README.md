# Agentic RAG Chatbot

## Miről szól?

Egy Corrective RAG chatbot, ami 5 arXiv PDF-ből tud válaszolni kérdésekre. A lényege, hogy nem csak simán retrieval + generálás van, hanem a rendszer menet közben dönt: releváns-e amit talált, kell-e újrafogalmazni a kérdést, megalapozott-e a válasz.

A pipeline LangGraph-fal van megépítve.

A router eldönti, hogy egyáltalán kell-e retrieval (üdvözlésre, off-topic kérdésre nem indul el a pipeline). A grader szűri a visszakapott dokumentumokat, ha egyik sem releváns, a query rewriter átfogalmazza a kérdést és újrapróbálja. A végén a hallucination check megnézi, hogy a válasz tényleg a dokumentumokból jön-e.

## Miért CRAG?

A sima RAG-nak az a hibája, hogy vakon bízik a retrieval eredményében. Ha rossz dokumentumokat húz be, rossz választ generál, ezt a felhasználó nem is tudja. A Corrective RAG ([Yan et al., 2024](https://arxiv.org/abs/2401.15884)) pont ezt orvosolja: értékeli a retrieval minőségét, és ha kell, korrigál.

Ez volt a legegyszerűbb módja annak, hogy valódi agentic viselkedést mutassak, így a rendszer nem egy fix pipeline-on megy végig, hanem döntéseket hoz a saját outputja alapján.

## Tudásbázis

5 PDF, egymásra épülő sorrendben:

| Paper | Miért van benne |
|-------|----------------|
| Attention Is All You Need (2017) | Alap, a Transformer architektúra |
| RAG (Lewis et al., 2020) | Az eredeti RAG koncepció |
| Chain-of-Thought (Wei et al., 2022) | Prompting technika, amit a rendszer is használ |
| Corrective RAG (Yan et al., 2024) | Az architektúra alapja |
| Agentic RAG Survey (2025) | Tágabb kontextus az agentic megközelítésekről |

## Tech stack

- **LLM**: Ollama (Mistral 7B) lokálisan, vagy HuggingFace Inference API
- **Embeddings**: `all-MiniLM-L6-v2` (384 dim)
- **Vector DB**: ChromaDB (minden futásnál újraépül)
- **Framework**: LangChain + LangGraph
- **Nyelv**: Python 3.12

Az Ollama-t választottam mert ingyenes, lokálisan fut, és egy 7B-os modell 16GB RAM-mal még kezelhető. A HuggingFace alternatíva arra az esetre van, ha valakinek nincs kapacitása lokálisan futtatni.

## Futtatás

```bash
# 1. Klónozás + virtuális környezet
git clone <url>
cd agentic-rag
python -m venv .venv
source .venv/bin/activate

# 2. Függőségek
pip install -r requirements.txt

# 3. Konfiguráció
cp .env.example .env

# 4. Ha Ollama-t használsz, indítsd el és húzd le a modellt
ollama pull mistral:7b-instruct-v0.3-q4_K_M

# 5. Notebook megnyitása
jupyter notebook AgenticRag.ipynb
```

A notebook cellái sorrendben futtathatók, mindent végigvezet: PDF letöltés, chunkolás, embedding, vector store, node tesztek, majd a teljes pipeline demo.

## Projekt struktúra

```
├── AgenticRag.ipynb          # Fő notebook (bemutató + futtatás)
├── config/
│   └── config.json           # Minden paraméter egy helyen
├── src/
│   ├── config.py             # Config betöltés
│   ├── document_processing.py # PDF letöltés, szövegkinyerés, chunkolás
│   ├── vector_store.py       # Embedding + ChromaDB
│   ├── llm.py                # LLM provider absztrakt
│   ├── nodes.py              # Agent node-ok (router, grader, rewriter, stb.)
│   └── graph.py              # LangGraph gráf összerakás
├── data/
│   ├── pdfs/                 # Letöltött paperek (gitignore-ban)
│   └── chroma_db/            # Vector store (gitignore-ban)
├── requirements.txt
├── .env.example
└── .gitignore
```

A config-ból minden lényeges paraméter állítható: chunk méret, overlap, top-k, temperature, retry limitek. Az LLM cserélhető `.env`-ből, nem kell kódot módosítani.

## Ismert limitációk

- **Lassú**: ~55s / kérdés CPU-n, 7B modellel. A 6 szekvenciális LLM hívás a szűk keresztmetszet.
- **Nincs conversation memory**: minden kérdés izolált, nincs kontextus az előző kérdésekből.
- **Csak similarity search**: hybrid search (vector + BM25) javítana a retrieval minőségen.
- **Kis tudásbázis**: 5 paper, ~540 chunk, éles rendszerben nagyságrendekkel több kellene.

Ezekről bővebben a notebook utolsó szekciójában írok.
