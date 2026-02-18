# GraphK-RagRec Optimization Walkthrough

We successfully implemented and optimized the Graph-based Movie Recommendation System. This walkthrough explains the two approaches we developed: the ultra-fast Cached Recommendation and the paper-faithful Real-time K-RagRec.

## 1. Project Overview

The goal was to improve the recommendation speed while maintaining the quality of graph-based reasoning. We implemented two distinct strategies to address different needs (Speed vs Flexibility).

### 🚀 Approach A: Fast Recommendation (Cached)
- **Script:** `fast_recommendation.py`
- **Method:** Pre-compute all movie relationships and store them in `movie_logic_cache.json`. Executing the script simply looks up the result.
- **Speed:** **~0.004 seconds** (Instant) @
- **Pros:** Extremely fast, zero API cost during inference.
- **Cons:** Pre-computation required (once), cannot handle dynamic user queries or unseen movies without re-caching.

### 🧠 Approach B: Real-time K-RagRec (Paper Faithful)
- **Script:** `k_rag_rec.py`
- **Method:** 
  1. **Semantic Search:** Find candidates using FAISS.
  2. **Sub-graph Retrieval:** Query Neo4j for paths between target and candidates.
  3. **LLM Inference:** GPT-4o generates reasoning based on paths.
- **Optimization:** Prompt engineering reduced response length, cutting time from 17s to ~7s.
- **Speed:** **~7.6 seconds** (Optimized from 17s)
- **Pros:** Handles dynamic queries, provides fresh reasoning, works for new data without full re-compute.
- **Cons:** Slower, incurs API costs per request.

## 2. Key Files & Usage

### Setup
Ensure you have the `.env` file configured with your `OPENAI_API_KEY`.

```bash
# Install dependencies
pip install neo4j torch numpy faiss-cpu openai python-dotenv
```

### Running Fast Recommendation
```bash
python fast_recommendation.py
# Output: Recommendations in ~0.004s
```

### Running Real-time K-RagRec
```bash
python k_rag_rec.py
# Output: Live reasoning & recommendations in ~7s
```

## 3. Optimization Details
- **OpenMP Fix:** Solved library conflict using `os.environ['KMP_DUPLICATE_LIB_OK']='True'`.
- **Prompt Engineering:** Forced GPT-4o to be concise ("max 20 words"), significantly reducing latency.
- **Security:** Implemented `.env` for safe API key management, ensuring no secrets are pushed to GitHub.

## 4. Repository Status
All code has been pushed to `graphrag_recommendation` branch `main`.
- `fast_recommendation.py`: Verified ✅
- `k_rag_rec.py`: Verified & Secured ✅
- `precompute_logic.py`: Utility for Approach A ✅
