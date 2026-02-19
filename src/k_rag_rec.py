import os
from dotenv import load_dotenv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import faiss
import openai
from collections import defaultdict
from neo4j import GraphDatabase
import time

# .env 파일 로드
load_dotenv(dotenv_path="../.env")

# --- 설정 ---
# 1. Neo4j & OpenAI
URI = "neo4j+ssc://2bdf163a.databases.neo4j.io:7687"
AUTH = ("neo4j", "dVRqLgBpDBT3tP37uYphK_zFZRjBHRizDVvRc4LCJRg")
api_key = os.getenv("OPENAI_API_KEY")

# 2. 리소스 로드
print("🔄 Loading resources (Embeddings & Graph Data)...")
sw_embeddings = np.load("../data/sw_embeddings.npy")
saved = torch.load("../data/full_graph.pt", weights_only=False)
node_meta = saved['node_meta']
NODE_TYPE_MAP = saved['NODE_TYPE_MAP']

# SceneWindow 인덱스 매핑
sw_mask = (saved['data'].node_type == NODE_TYPE_MAP['SceneWindow'])
sw_indices = sw_mask.nonzero(as_tuple=True)[0]

movie_to_sw = defaultdict(list)
for idx in range(len(sw_indices)):
    meta = node_meta[sw_indices[idx].item()]
    movie_to_sw[meta['movie']].append(idx)

# 영화별 센트로이드 계산
movie_centroids = {}
for m, indices in movie_to_sw.items():
    vecs = sw_embeddings[indices].astype(np.float32)
    movie_centroids[m] = np.mean(vecs, axis=0)

# FAISS 인덱스 빌드
movie_titles = list(movie_centroids.keys())
matrix = np.array([movie_centroids[t] for t in movie_titles], dtype=np.float32)
faiss.normalize_L2(matrix)
index = faiss.IndexFlatIP(256)
index.add(matrix)

# 3. K-RagRec 구현 (Real-time Graph Retrieval + LLM Reasoning)
def retrieve_knowledge_subgraph(driver, target_movie, candidates):
    """
    논문의 'Knowledge Sub-graphs Retrieval' 단계 구현.
    target_movie와 candidate_movies 간의 연결 경로(Path)를 실시간으로 검색.
    """
    subgraphs = {}
    
    with driver.session() as session:
        for candidate in candidates:
            # 2-hop 이상의 경로 탐색 (논문에서도 Multi-hop Reasoning 강조)
            query = """
            MATCH (m1:Movie {title: $t})-[:HAS_WINDOW]->(sw1:SceneWindow)
            MATCH (m2:Movie {title: $c})-[:HAS_WINDOW]->(sw2:SceneWindow)
            MATCH p = (sw1)-[*1..2]-(sw2)
            RETURN [n in nodes(p) | coalesce(n.name, labels(n)[0])] as path
            LIMIT 5
            """
            result = session.run(query, t=target_movie, c=candidate).data()
            
            # 경로를 텍스트화 (Graph-to-Text)
            paths_text = []
            for record in result:
                path = " -> ".join(record['path'])
                paths_text.append(path)
            
            subgraphs[candidate] = paths_text
            
    return subgraphs

def generate_recommendation_with_llm(target_movie, subgraphs):
    """
    논문의 'Knowledge-augmented Recommendation' 단계 구현.
    LLM에게 그래프 정보를 제공하고 추천 이유를 생성하게 함.
    """
    client = openai.OpenAI(api_key=api_key)
    
    # Context 구성
    context_text = f"Target Movie: {target_movie}\n\n"
    for movie, paths in subgraphs.items():
        context_text += f"Candidate: {movie}\n"
        context_text += "  - Relationships:\n"
        for p in paths:
             context_text += f"    * {p}\n"
        context_text += "\n"
        
    prompt = f"""
    You are a recommendation engine. 
    User watched '{target_movie}'.
    Based on these retrieved paths:
    {context_text}
    
    Recommend each candidate in ONE short sentence (max 20 words). 
    Do NOT write intro/outro. Just list the movies and reasons.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def run_k_rag_rec(target_movie):
    start_time = time.time()
    print(f"🚀 Starting K-RagRec Pipeline for '{target_movie}'...")
    
    if target_movie not in movie_centroids:
        print("❌ Movie not found.")
        return

    # 1. Retrieval (Semantic Search)
    print("🔍 1. Semantic Retrieval (Vector Search)...")
    q_vec = movie_centroids[target_movie].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k=4) # Top 3 candidates (excluding self)
    
    candidates = []
    for idx in I[0]:
        title = movie_titles[idx]
        if title != target_movie:
            candidates.append(title)
            
    # 2. Knowledge Sub-graph Retrieval (Real-time Graph Query)
    print("🕸️ 2. Knowledge Sub-graph Retrieval (Neo4j Query)...")
    driver = GraphDatabase.driver(URI, auth=AUTH)
    subgraphs = retrieve_knowledge_subgraph(driver, target_movie, candidates)
    driver.close()
    
    # 3. LLM Reasoning & Generation
    print("🤖 3. Knowledge-Augmented Generation (LLM Inference)...")
    final_report = generate_recommendation_with_llm(target_movie, subgraphs)
    
    end_time = time.time()
    print("\n=== ✨ K-RagRec Recommendation Result ✨ ===")
    print(final_report)
    print(f"\n⏱️ Total Execution Time: {end_time - start_time:.4f} sec")


if __name__ == "__main__":
    run_k_rag_rec("Aftersun 2022")
