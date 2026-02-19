# 🎬 GraphRAG 영화 추천 시스템 (K-RagRec)

## 1. 🎯 프로젝트 목표
- **"영화의 서사(Narrative)와 감정(Emotion)을 이해하는 추천"**
- 단순 평점/장르 기반 추천의 한계를 넘어, **"이 영화가 왜 당신에게 맞는지"** 논리적으로 설명할 수 있는 추천 시스템 구축.
- **Neo4j (Knowledge Graph)**와 **GNN (Graph Neural Network)**, **LLM (GPT-4o)**의 강점을 결합한 **GraphRAG** 아키텍처 구현.

---

## 2. 🏗️ 전체 시스템 아키텍처 (4단계 파이프라인)

### Step 1: 지식 그래프 구축 (LLM-based Graph Construction)
- **데이터 소스:** 83편의 영화 스크립트 (PDF)
- **구축 파이프라인:**
    1. **Pre-processing:** `INT/EXT` 패턴을 인식하여 스크립트를 3,800여 개의 "Scene"으로 분할.
    2. **Sliding Window:** 3개의 Scene을 하나의 윈도우로 묶어 LLM(Qwen)에 입력 (맥락 유지).
    3. **Extraction (Ontology):**
        - **Narrative Phase:** 기(Exposition) → 승(Rising) → 전(Climax) → 결(Resolution)
        - **Trope:** Revenge, Quest, Discovery 등 서사 장치 추출.
        - **Emotion:** Plutchik 바퀴 기반의 8가지 감정 추출.
    4. **Graph Modeling:** Neo4j에 로드하여 약 **5,400개 노드**와 **34,000개 관계** 생성.

### Step 2: GNN 학습 (Representation Learning)
- **모델:** Graph Attention Network (GAT)
- **학습 방법:** **Self-Supervised Contrastive Learning (대조 학습)**
    - **Positive:** 같은 영화 내 인접한 장면들은 서로 비슷하다.
    - **Negative:** 다른 영화의 장면들은 서로 다르다.
- **결과:** 영화의 서사적 특징을 담은 고차원 벡터(`sw_embeddings.npy`) 생성.

### Step 3: 추천 알고리즘 (Hybrid Retrieval)
1.  **Semantic Search (유사도 검색):**
    - FAISS 인덱스를 활용해 사용자가 본 영화와 가장 벡터 유사도가 높은 후보군(Candidate) 1차 선별.
2.  **Structural Retrieval (구조적 검색):**
    - Neo4j에서 [Target 영화] ↔ [Candidate 영화] 사이의 **연결 경로(Path)**를 실시간 추출.
    - 예: `(Movie A) -[HAS_WINDOW]-> (Sadness) <-[HAS_WINDOW]- (Movie B)`

### Step 4: 설명 가능한 추천 (Explanation)
- **LLM (GPT-4o) 활용:**
    - 추출된 경로(Path) 데이터를 프롬프트로 입력.
    - **"왜 이 영화를 추천하는지"** 서사적 맥락을 반영하여 사용자에게 설명.

---

## 3. ⚡️ 성능 최적화 전략 (Performance Optimization)
실제 서비스 적용을 위해 두 가지 모드로 시스템을 최적화.

### 🅰️ Track A: Instant Mode (캐시 기반)
- **방식:** 모든 영화 간의 추천 사유를 미리 연산하여 저장(`Pre-computation`).
- **속도:** **0.004초** (실시간 대비 5,000배 가속) 🚀
- **장점:** 압도적인 반응 속도, API 비용 "0".

### 🅱️ Track B: Deep Reasoning Mode (실시간 추론)
- **방식:** 논문(K-RagRec)의 실시간 추론 파이프라인을 그대로 수행.
- **최적화:** 프롬프트 길이를 제어하여 초기 17초 → **7.6초**로 단축 성공. ⚡️
- **장점:** 새로운 영화나 복잡한 질의에도 즉각 대응 가능.

---

## 4. 🏁 결론 (Takeaway)
> "우리는 단순한 추천을 넘어, 영화의 **[구조적 유사성]**을 분석하고 **[설명 가능한]** 추천을 제공하는 차세대 추천 시스템을 완성했습니다."
