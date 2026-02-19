# GraphRAG Movie Recommendation System 🎬

A hybrid recommendation system combining **Graph Neural Networks (GNN)**, **Knowledge Graphs (Neo4j)**, and **Large Language Models (GPT-4o)** to provide context-aware, explainable movie recommendations.

## 📂 Repository Structure

```
.
├── src/                  # 🚀 Core Application Code
│   ├── fast_recommendation.py   # Cached, instant recommendation (0.004s)
│   └── k_rag_rec.py             # Real-time RAG pipeline (Retrieval + LLM)
├── data/                 # 💾 Data Assets (Embeddings, Graph, Cache)
│   ├── movie_logic_cache.json   # Pre-computed logic for fast mode
│   ├── narrative_index.faiss    # Vector search index
│   ├── sw_embeddings.npy        # GNN embeddings
│   └── full_graph.pt            # PyTorch Geometric graph data
├── notebooks/            # 📓 Jupyter Notebooks
│   ├── gnn학습.ipynb             # GNN training notebook
│   └── 추천.ipynb                # Recommendation experiments
├── scripts/              # 🛠️ Utility Scripts
│   ├── precompute_logic.py      # Script to generate movie_logic_cache.json
│   ├── gnn_train_corrected.py   # Refactored GNN training script
│   └── sanitize_notebooks.py    # Security tool to clean API keys
├── docs/                 # 📚 Documentation
│   ├── walkthrough.md           # Detailed project walkthrough
│   ├── framework_presentation.md# System architecture presentation
│   └── gnn_guide.md             # GNN implementation guide
└── README.md             # This file
```

## 🛠️ Setup

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   
   # Install dependencies
   pip install neo4j torch numpy faiss-cpu openai python-dotenv tqdm
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory:
   ```ini
   OPENAI_API_KEY=your_openai_api_key_here
   # NEO4J_PASSWORD=... (Optional if using hardcoded auth)
   ```

## 🚀 Usage

### 1. Fast Mode (Recommended)
Instant recommendations using pre-computed logic.
```bash
cd src
python fast_recommendation.py
```

### 2. Real-time RAG Mode
Deep reasoning using live Graph retrieval and LLM generation.
```bash
cd src
python k_rag_rec.py
```

## 🧠 Core Technologies
- **Neo4j:** Stores movie narratives (Scenes, Emotions, Tropes) as a Knowledge Graph.
- **GNN (GAT):** Learns structural embeddings via Contrastive Learning.
- **FAISS:** Enables high-speed vector similarity search.
- **GPT-4o:** Synthesizes graph paths into natural language explanations.

## 📝 Documentation
For more details, check the `docs/` folder:
- [Walkthrough](docs/walkthrough.md): Comprehensive guide to the optimized system.
- [Presentation](docs/framework_presentation.md): High-level architectural overview.
