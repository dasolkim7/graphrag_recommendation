import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import faiss
import numpy as np
from collections import defaultdict
import os

def main():
    print("🔄 Loading data safely...")
    try:
        saved = torch.load("full_graph.pt", weights_only=False)
        data = saved['data']
        node_meta = saved['node_meta']
        NODE_TYPE_MAP = saved['NODE_TYPE_MAP']
    except FileNotFoundError:
        print("❌ 'full_graph.pt' 파일을 찾을 수 없습니다. 먼저 그래프를 추출해주세요.")
        return

    # SceneWindow 인덱스 추출
    sw_mask = (data.node_type == NODE_TYPE_MAP['SceneWindow'])
    sw_indices = sw_mask.nonzero(as_tuple=True)[0]

    # --- 손실 함수 & Positive Pair 생성 로직 ---
    def find_positive_pairs(node_meta, sw_indices):
        """같은 영화의 인접 윈도우를 positive pair로 정의."""
        movie_windows = defaultdict(list)
        
        # SceneWindow 글로벌 인덱스를 가져와 메타데이터 조회
        for idx_tensor in sw_indices:
            idx = idx_tensor.item()
            meta = node_meta[idx]
            if meta.get('window') is not None:
                movie_windows[meta['movie']].append((idx, meta['window']))
        
        pairs = []
        # 영화별로 윈도우 번호 순 정렬 후 인접 쌍 생성
        for movie, windows in movie_windows.items():
            windows.sort(key=lambda x: x[1])  # window number 오름차순
            for i in range(len(windows) - 1):
                # (global_idx1, global_idx2)
                pairs.append((windows[i][0], windows[i+1][0]))
        
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
            
        return torch.tensor(pairs, dtype=torch.long).t()  # [2, num_pairs]

    def contrastive_loss(embeddings, positive_pairs, temperature=0.5):
        """
        embeddings: 전체 노드 임베딩 [N, dim]
        positive_pairs: [2, num_pairs] - (idx_a, idx_b)
        """
        idx_a, idx_b = positive_pairs
        
        anchor = embeddings[idx_a]
        positive = embeddings[idx_b]
        
        # Cosine Similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / temperature
        
        # Negative Sampling (Shuffle)
        # 배치 내에서 랜덤하게 섞은 것을 negative로 간주
        batch_size = anchor.size(0)
        perm = torch.randperm(batch_size)
        negative = anchor[perm]

        neg_sim = F.cosine_similarity(anchor, negative, dim=-1) / temperature
        
        # InfoNCE Loss
        logits = torch.stack([pos_sim, neg_sim], dim=1) # [batch, 2]
        labels = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device) # 0번째가 positive
        
        return F.cross_entropy(logits, labels)

    # Positive Pairs 미리 계산
    print("🔗 Generating positive pairs (finding adjacent windows in same movies)...")
    pos_edge_index = find_positive_pairs(node_meta, sw_indices)
    print(f"✅ Positive Pairs created: {pos_edge_index.size(1)} pairs")

    if pos_edge_index.size(1) == 0:
        print("⚠️ Warning: No positive pairs found. Training might be unstable.")

    # 2. GNN 모델 정의 (GAT)
    class NarrativeGNN(torch.nn.Module):
        def __init__(self, in_dim=43, hidden_dim=128, out_dim=256):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=4, dropout=0.2)
            self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, dropout=0.2)
        
        def forward(self, x, edge_index):
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x

    model = NarrativeGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. 학습 루프
    print("🚀 GNN Training Start (Contrastive Learning)...")
    model.train()
    for epoch in range(100): 
        optimizer.zero_grad()
        
        # 전체 임베딩 생성
        out = model(data.x, data.edge_index)
        
        # Contrastive Loss (Positive Pair + Negative Sampling)
        if pos_edge_index.size(1) > 0:
            loss = contrastive_loss(out, pos_edge_index)
        else:
             # Fallback if no pairs (shouldn't happen in this dataset)
             loss = torch.tensor(0.0, requires_grad=True)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100 | Loss: {loss.item():.4f}")

    # 1. 임베딩 추출 및 L2 정규화
    model.eval()
    with torch.no_grad():
        final_out = model(data.x, data.edge_index)
        sw_embeddings = final_out[sw_indices].cpu().numpy()
    faiss.normalize_L2(sw_embeddings)

    # 2. 인덱스 생성 및 저장
    d = sw_embeddings.shape[1]
    new_index = faiss.IndexFlatIP(d)  # Cosine Similarity용
    new_index.add(sw_embeddings)

    # 파일 저장
    faiss.write_index(new_index, "narrative_index.faiss")
    np.save("sw_embeddings.npy", sw_embeddings)

    print(f"✅ Files saved: {os.path.abspath('narrative_index.faiss')}")
    print(f"✅ Vector count: {new_index.ntotal}")

if __name__ == "__main__":
    main()
