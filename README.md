# 🧊🍳 FRIDGE-RAG

> **From fridge photo ➜ confident ingredients ➜ useful recipes**  
> A practical, production-minded prototype that combines **computer vision** and **retrieval-augmented generation (RAG)** for real kitchen decisions.

---

## ✨ What this project does

Most "what can I cook?" demos stop at simple ingredient detection. **FRIDGE-RAG** goes further:

- 🥕 Detects fridge ingredients using an **ensemble of vision models**.
- 📚 Retrieves relevant recipes from a **semantic vector database**.
- 🧠 Re-ranks candidates with an LLM and provides **clear rationale**.
- 🏗️ Separates concerns (API, orchestration, vision, retrieval, UI) for maintainability.

---

## 🧭 End-to-end architecture

### 1) Runtime flow

```text
[Client App / Streamlit UI]
            |
            v
      [FastAPI Gateway]
            |
            v
   [Pipeline Orchestrator]
     |               |
     |               +--------------------------+
     v                                          v
[Vision Ensemble]                          [RAG Service]
(YOLO + DETR + CLIP)        query ---> [Embedding + Chroma Retrieval]
     |                                          |
     +------ detected ingredients --------------+
                         |
                         v
               [LLM Re-ranker / Explainer]
                         |
                         v
         [Ranked recipe candidates + rationale]
```

### 2) Offline indexing flow

```text
[Kaggle recipe dataset]
          |
          v
 [scripts/build_vectordb.py]
  - cleaning
  - chunking/formatting
  - embedding generation
          |
          v
   [Local Chroma recipe index]
```

---

## 🗂️ Repository layout

```bash
FRIDGE-RAG/
├── api/                  # FastAPI app + schemas
├── dashboard/            # Streamlit front-end
├── scripts/              # Offline jobs (index build)
├── src/
│   ├── pipeline.py       # End-to-end orchestration
│   ├── config.py         # Runtime configuration
│   ├── rag/              # Ingestion, retrieval, reranking
│   └── vision/           # YOLO/DETR/CLIP + ensemble logic
├── tests/                # Unit/integration-oriented tests
├── requirements.txt
└── README.md
```

---

## 🧰 Tech stack

- ⚡ **Backend API:** FastAPI + Uvicorn
- 🖥️ **UI:** Streamlit
- 👁️ **Vision models:** YOLOv8, DETR, CLIP (ensemble strategy)
- 🔎 **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- 🧱 **Vector store:** ChromaDB
- 🤖 **LLM layer:** OpenAI models for reranking/explanations
- 🍲 **Dataset:** Food.com Recipes and Reviews (Kaggle)

Dataset link: https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews/data

---

## 🚀 Quick start

### 1) Prerequisites

- Python 3.10+
- `pip` and virtual environment tooling
- Kaggle API credentials
- OpenAI API key

### 2) Install

```bash
git clone <your-repo-url>
cd FRIDGE-RAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Configure environment

```bash
cp .env.example .env
```

Add your key:

```env
OPENAI_API_KEY=your_openai_key
```

### 4) Configure Kaggle credentials

```bash
mkdir -p ~/.kaggle
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 5) Build the vector database

```bash
python scripts/build_vectordb.py
```

---

## ▶️ Run locally

### API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

---

## ✅ Testing

```bash
pytest tests/ -v
```

Optional focused runs:

```bash
pytest tests/test_pipeline.py -v
pytest tests/test_vision.py -v
pytest tests/test_rag.py -v
```

---

## 📁 Why `data/` is not on GitHub (but appears locally)

This is intentional and important.

### Short answer

- 🚫 `data/` is usually listed in `.gitignore`, so Git does **not** track or upload it.
- 💻 When you run local setup/indexing scripts, they **create `data/` on your machine**.
- 👀 Your coding editor (VS Code, Cursor, PyCharm, etc.) shows **local folders**, not only tracked Git files.

So even if GitHub does not display `data/`, your editor will show it once created locally.

### Why we do this

Keeping `data/` out of GitHub helps avoid:

- oversized repositories and slow clones,
- committing generated artifacts repeatedly,
- accidental exposure of local or licensed dataset outputs,
- noisy diffs from frequently regenerated indexes.

### Typical local workflow

1. Clone repository from GitHub (no `data/` yet).
2. Run setup/build commands (`scripts/build_vectordb.py`).
3. Script generates local artifacts under `data/`.
4. Editor displays the folder immediately.
5. Git still ignores it, so `git status` stays clean for those files.

---

## 🛠️ Operational notes

- Rebuild the vector index whenever recipe corpus or embedding model changes.
- For production hardening, add:
  - containerized services,
  - centralized logging/tracing,
  - secret management,
  - model/version pinning,
  - request throttling and caching.

---

## 🗺️ Roadmap

- Add dietary and allergen-aware filtering.
- Add retrieval quality evaluation (Precision@K / Recall@K).
- Add Dockerized local stack and CI checks.
- Add feedback loop for ranking quality improvements.

---

## 🤝 Contributing

PRs and ideas are welcome. If you propose architecture or pipeline changes, include:

- motivation,
- impact on retrieval quality,
- expected runtime/memory tradeoffs,
- testing evidence.
