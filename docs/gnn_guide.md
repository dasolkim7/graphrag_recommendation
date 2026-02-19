# GNN 구축 가이드 — Neo4j에서 데이터 추출 → GNN 학습

## 0. 설치

```bash
pip install neo4j torch torch-geometric faiss-cpu
```

## 1. Neo4j 연결

```python
from neo4j import GraphDatabase

URI = "neo4j+s://2bdf163a.databases.neo4j.io"
AUTH = ("neo4j", "비밀번호")
driver = GraphDatabase.driver(URI, auth=dVRqLgBpDBT3tP37uYphK_zFZRjBHRizDVvRc4LCJRg)
```

## 2. 그래프 스키마 & 규모

```
Movie (83) ─HAS_WINDOW→ SceneWindow (3,848) ─IN_PHASE→ NarrativePhase (5종)
                                              ├─HAS_TROPE→ Trope (7종)
                                              ├─HAS_EMOTION→ Emotion (8종)
                                              └─IN_SETTING→ Setting (7종)

Character (1,413) ─APPEARS_IN→ SceneWindow
                   ├─BELONGS_TO→ Movie
                   ├─HAS_ARCHETYPE→ PersonaArchetype (8종)
                   └─RELATES_TO→ Character
```

**총 노드: ~5,400개 / 총 관계: ~34,000개**

---

## 3. 전체 그래프 추출 (1회만 실행)

> GNN 학습을 위해 **전체 그래프를 한번에 추출**하여 로컬 파일로 저장.
> 이후 학습 시에는 Neo4j 접속 없이 로컬 파일만 사용.

### Step 3-1: 전체 노드 추출

```python
import torch
import json
from collections import defaultdict

# ── 온톨로지 값 → 인덱스 매핑 ──
PHASES = ['Exposition', 'Rising', 'Climax', 'Falling', 'Resolution']
TROPES = ['Revenge', 'Quest', 'Discovery', 'Escape', 'Redemption', 'Sacrifice', 'Betrayal']
EMOTIONS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
SETTINGS = ['Urban', 'Nature', 'Indoor', 'Tech', 'Historical', 'Surreal', 'Void']
ARCHETYPES = ['Hero', 'Anti-Hero', 'Mentor', 'Shadow', 'Trickster', 'Rebel', 'Caregiver', 'Lone Wolf']

# ── 노드 타입 (7종) ──
NODE_TYPE_MAP = {
    'Movie': 0, 'SceneWindow': 1, 'Character': 2,
    'NarrativePhase': 3, 'Trope': 4, 'Emotion': 5,
    'Setting': 6, 'PersonaArchetype': 7
}
NUM_NODE_TYPES = len(NODE_TYPE_MAP)


def extract_all_nodes(driver):
    """Neo4j에서 전체 노드를 추출하고 고유 ID를 부여."""
    
    node_id_map = {}  # neo4j internal id → 연속 정수 인덱스
    node_features = []
    node_labels = []  # 어떤 타입인지
    node_meta = []    # 디버깅/추적용 메타데이터
    
    with driver.session() as session:
        
        # ── Movie 노드 ──
        result = session.run("MATCH (m:Movie) RETURN id(m) AS nid, m.title AS title")
        for record in result:
            idx = len(node_id_map)
            node_id_map[record["nid"]] = idx
            node_labels.append(NODE_TYPE_MAP['Movie'])
            node_meta.append({'type': 'Movie', 'title': record['title']})
        
        # ── SceneWindow 노드 ──
        result = session.run("""
            MATCH (sw:SceneWindow)
            RETURN id(sw) AS nid, sw.movie_title AS movie, sw.window_number AS wnum
        """)
        for record in result:
            idx = len(node_id_map)
            node_id_map[record["nid"]] = idx
            node_labels.append(NODE_TYPE_MAP['SceneWindow'])
            node_meta.append({
                'type': 'SceneWindow',
                'movie': record['movie'],
                'window': record['wnum']
            })
        
        # ── Character 노드 ──
        result = session.run("""
            MATCH (c:Character)
            RETURN id(c) AS nid, c.script_name AS name, c.movie_title AS movie
        """)
        for record in result:
            idx = len(node_id_map)
            node_id_map[record["nid"]] = idx
            node_labels.append(NODE_TYPE_MAP['Character'])
            node_meta.append({
                'type': 'Character',
                'name': record['name'],
                'movie': record['movie']
            })
        
        # ── 글로벌 노드들 (NarrativePhase, Trope, Emotion, Setting, PersonaArchetype) ──
        for label, type_key in [
            ('NarrativePhase', 'NarrativePhase'),
            ('Trope', 'Trope'),
            ('Emotion', 'Emotion'),
            ('Setting', 'Setting'),
            ('PersonaArchetype', 'PersonaArchetype'),
        ]:
            result = session.run(f"MATCH (n:{label}) RETURN id(n) AS nid, n.name AS name")
            for record in result:
                idx = len(node_id_map)
                node_id_map[record["nid"]] = idx
                node_labels.append(NODE_TYPE_MAP[type_key])
                node_meta.append({'type': type_key, 'name': record['name']})
    
    print(f"총 노드 수: {len(node_id_map)}")
    return node_id_map, node_labels, node_meta
```

### Step 3-2: 전체 엣지 추출

```python
# ── 관계 타입 → 인덱스 매핑 ──
EDGE_TYPE_MAP = {
    'HAS_WINDOW': 0, 'IN_PHASE': 1, 'HAS_TROPE': 2,
    'HAS_EMOTION': 3, 'IN_SETTING': 4, 'APPEARS_IN': 5,
    'BELONGS_TO': 6, 'HAS_ARCHETYPE': 7, 'RELATES_TO': 8
}


def extract_all_edges(driver, node_id_map):
    """Neo4j에서 전체 관계를 추출하여 edge_index 텐서로 변환."""
    
    src_list = []
    dst_list = []
    edge_types = []
    
    with driver.session() as session:
        # 모든 관계를 한번에 가져오기
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN id(a) AS src, id(b) AS dst, type(r) AS rel_type
        """)
        
        for record in result:
            src_neo = record["src"]
            dst_neo = record["dst"]
            rel_type = record["rel_type"]
            
            # node_id_map에 있는 노드만 포함
            if src_neo in node_id_map and dst_neo in node_id_map:
                src_list.append(node_id_map[src_neo])
                dst_list.append(node_id_map[dst_neo])
                edge_types.append(EDGE_TYPE_MAP.get(rel_type, -1))
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    print(f"총 엣지 수: {edge_index.shape[1]}")
    return edge_index, edge_type
```

### Step 3-3: 노드 특성(Feature) 벡터 생성

```python
def build_node_features(node_labels, node_meta):
    """
    각 노드에 대해 특성 벡터를 생성.
    
    특성 구성 (총 dim = 8 + 5 + 7 + 8 + 7 + 8 = 43):
      - Node type 원핫 (8차원)
      - Phase 원핫 (5차원) — NarrativePhase 노드일 때만 활성
      - Trope 원핫 (7차원) — Trope 노드일 때만 활성
      - Emotion 원핫 (8차원) — Emotion 노드일 때만 활성
      - Setting 원핫 (7차원) — Setting 노드일 때만 활성
      - Archetype 원핫 (8차원) — PersonaArchetype 노드일 때만 활성
    """
    
    FEATURE_DIM = NUM_NODE_TYPES + len(PHASES) + len(TROPES) + len(EMOTIONS) + len(SETTINGS) + len(ARCHETYPES)
    # = 8 + 5 + 7 + 8 + 7 + 8 = 43
    
    features = torch.zeros(len(node_labels), FEATURE_DIM)
    
    for i, (label, meta) in enumerate(zip(node_labels, node_meta)):
        # 1) 노드 타입 원핫
        features[i, label] = 1.0
        
        offset = NUM_NODE_TYPES  # 8
        
        # 2) 값 기반 원핫 (해당 타입인 경우에만)
        name = meta.get('name', '')
        
        if meta['type'] == 'NarrativePhase' and name in PHASES:
            features[i, offset + PHASES.index(name)] = 1.0
        
        offset += len(PHASES)  # +5
        
        if meta['type'] == 'Trope' and name in TROPES:
            features[i, offset + TROPES.index(name)] = 1.0
        
        offset += len(TROPES)  # +7
        
        if meta['type'] == 'Emotion' and name in EMOTIONS:
            features[i, offset + EMOTIONS.index(name)] = 1.0
        
        offset += len(EMOTIONS)  # +8
        
        if meta['type'] == 'Setting' and name in SETTINGS:
            features[i, offset + SETTINGS.index(name)] = 1.0
        
        offset += len(SETTINGS)  # +7
        
        if meta['type'] == 'PersonaArchetype' and name in ARCHETYPES:
            features[i, offset + ARCHETYPES.index(name)] = 1.0
    
    print(f"특성 행렬: {features.shape}  (노드 × 특성차원)")
    return features
```

### Step 3-4: PyG Data 객체 생성 & 저장

```python
from torch_geometric.data import Data

def build_and_save_graph(driver, save_path="full_graph.pt"):
    """Neo4j → PyG Data 객체 → 파일 저장 (1회 실행)"""
    
    # 1) 노드 추출
    node_id_map, node_labels, node_meta = extract_all_nodes(driver)
    
    # 2) 엣지 추출
    edge_index, edge_type = extract_all_edges(driver, node_id_map)
    
    # 3) 특성 벡터 생성
    x = build_node_features(node_labels, node_meta)
    
    # 4) PyG Data 객체 생성
    data = Data(
        x=x,                                           # [N, 43] 특성 행렬
        edge_index=edge_index,                          # [2, E] 엣지 인덱스
        edge_type=edge_type,                            # [E] 엣지 타입
        node_type=torch.tensor(node_labels),            # [N] 노드 타입
    )
    
    # 5) 저장
    torch.save({
        'data': data,
        'node_meta': node_meta,            # 노드 메타데이터 (디버깅용)
        'node_id_map': node_id_map,        # Neo4j ID → PyG 인덱스 매핑
        'NODE_TYPE_MAP': NODE_TYPE_MAP,
        'EDGE_TYPE_MAP': EDGE_TYPE_MAP,
    }, save_path)
    
    print(f"\n✅ 그래프 저장 완료: {save_path}")
    print(f"   노드: {data.num_nodes}, 엣지: {data.num_edges}")
    print(f"   특성 차원: {data.x.shape[1]}")
    return data


# ── 실행 ──
data = build_and_save_graph(driver, "full_graph.pt")
```

**출력 예시:**
```
총 노드 수: 5377
총 엣지 수: 34405
특성 행렬: torch.Size([5377, 43])  (노드 × 특성차원)

✅ 그래프 저장 완료: full_graph.pt
   노드: 5377, 엣지: 34405
   특성 차원: 43
```

---

## 4. 학습 데이터 준비 (NeighborLoader)

> 이후로는 Neo4j 접속 불필요. `full_graph.pt`만 사용.

```python
from torch_geometric.loader import NeighborLoader

# 저장된 그래프 로드
saved = torch.load("full_graph.pt")
data = saved['data']
node_meta = saved['node_meta']
NODE_TYPE_MAP = saved['NODE_TYPE_MAP']

# SceneWindow 노드의 인덱스만 추출 (학습 대상)
sw_mask = (data.node_type == NODE_TYPE_MAP['SceneWindow'])
sw_indices = sw_mask.nonzero(as_tuple=True)[0]
print(f"SceneWindow 수: {len(sw_indices)}")  # 3,848개

# 2-hop 이웃 샘플링 로더
loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],     # 1-hop: 최대 10개, 2-hop: 최대 5개
    input_nodes=sw_indices,    # SceneWindow만 중심 노드로
    batch_size=64,
    shuffle=True,
)

# 배치 확인
batch = next(iter(loader))
print(f"배치: 노드 {batch.num_nodes}, 엣지 {batch.num_edges}")
```

---

## 5. GNN 모델 (GAT)

```python
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class NarrativeGNN(torch.nn.Module):
    def __init__(self, in_dim=43, hidden_dim=128, out_dim=256):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, dropout=0.2)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        
        # SceneWindow 중심 노드의 임베딩만 추출
        # (NeighborLoader의 batch에서 input_nodes는 앞쪽에 위치)
        batch_size = data.batch_size if hasattr(data, 'batch_size') else x.size(0)
        return x[:batch_size]  # [batch_size, out_dim]

model = NarrativeGNN(in_dim=43, hidden_dim=128, out_dim=256)
```

## 6. 자기지도 학습 (Contrastive Learning)

```python
def find_positive_pairs(node_meta, sw_indices):
    """같은 영화의 인접 윈도우를 positive pair로 정의."""
    
    # movie → [(pyg_idx, window_number), ...]
    movie_windows = defaultdict(list)
    for idx in sw_indices.tolist():
        meta = node_meta[idx]
        movie_windows[meta['movie']].append((idx, meta['window']))
    
    pairs = []
    for movie, windows in movie_windows.items():
        windows.sort(key=lambda x: x[1])  # window number 순 정렬
        for i in range(len(windows) - 1):
            pairs.append((windows[i][0], windows[i+1][0]))
    
    return pairs


def contrastive_loss(anchor, positive, negatives, temperature=0.1):
    """InfoNCE Loss"""
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / temperature
    neg_sims = F.cosine_similarity(
        anchor.unsqueeze(1), negatives, dim=-1
    ) / temperature
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sims], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)
```

### 학습 루프

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
positive_pairs = find_positive_pairs(node_meta, sw_indices)

for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Forward
        embeddings = model(batch)  # [batch_size, 256]
        
        # Positive/Negative 샘플링 (간소화 버전)
        # 같은 배치 내에서 positive pair 찾기
        # 나머지를 negative로 사용
        anchor = embeddings[0::2]
        positive = embeddings[1::2]
        min_len = min(len(anchor), len(positive))
        anchor, positive = anchor[:min_len], positive[:min_len]
        
        # 배치 내 다른 모든 임베딩을 negative로
        negatives = embeddings.unsqueeze(0).expand(min_len, -1, -1)
        
        loss = contrastive_loss(anchor, positive, negatives)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

# 모델 저장
torch.save(model.state_dict(), "narrative_gnn.pth")
```

---

## 7. 임베딩 생성 & FAISS 인덱싱

```python
import faiss
import numpy as np

model.eval()
all_embeddings = []
all_sw_meta = []  # 각 임베딩이 어떤 영화/윈도우인지

with torch.no_grad():
    for batch in loader:
        emb = model(batch)  # [batch_size, 256]
        all_embeddings.append(emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0).numpy()  # [3848, 256]

# L2 정규화 (코사인 유사도를 위해)
faiss.normalize_L2(embeddings)

# FAISS 인덱스 생성
index = faiss.IndexFlatIP(256)  # Inner Product = 코사인 (정규화 후)
index.add(embeddings)
faiss.write_index(index, "narrative_index.faiss")

print(f"✅ FAISS 인덱스 저장: {embeddings.shape[0]}개 벡터, dim={embeddings.shape[1]}")
```

### 유사 윈도우 검색 예시

```python
# 쿼리: "Oppenheimer Window 15"와 유사한 윈도우 Top-5
query_idx = 100  # 예시
query_vec = embeddings[query_idx:query_idx+1]

distances, indices = index.search(query_vec, k=5)
for dist, idx in zip(distances[0], indices[0]):
    meta = node_meta[sw_indices[idx].item()]
    print(f"  {meta['movie']} W{meta['window']} (sim={dist:.3f})")
```
