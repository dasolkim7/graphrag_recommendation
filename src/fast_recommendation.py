import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import torch
import numpy as np
import faiss
import openai
from collections import defaultdict

# --- 설정 ---
# CACHE_FILE = "movie_logic_cache.json" # 이 변수는 이제 사용되지 않습니다.

def load_resources():
    print("🔄 Loading resources (Embeddings & Cache)...")
    
    # 1. 캐시 로드
    if not os.path.exists("../data/movie_logic_cache.json"):
        raise FileNotFoundError(f"Run 'precompute_logic.py' first to generate ../data/movie_logic_cache.json")
        
    with open("../data/movie_logic_cache.json", "r", encoding="utf-8") as f:
        movie_logic_cache = json.load(f)
        
    # 2. 임베딩 로드 (유사도 점수 계산용)
    sw_embeddings = np.load("../data/sw_embeddings.npy")
    # full_graph.pt에서 메타데이터 로드
    saved = torch.load("../data/full_graph.pt", weights_only=False)
    node_meta = saved['node_meta']
    NODE_TYPE_MAP = saved['NODE_TYPE_MAP']  # 영화별 센트로이드 계산 (메모리 로드)
    
    sw_mask = (saved['data'].node_type == NODE_TYPE_MAP['SceneWindow'])
    sw_indices = sw_mask.nonzero(as_tuple=True)[0]
    
    movie_to_sw = defaultdict(list)
    for idx in range(len(sw_indices)):
        meta = node_meta[sw_indices[idx].item()]
        movie_to_sw[meta['movie']].append(idx)
        
    movie_centroids = {}
    for m, indices in movie_to_sw.items():
        vecs = sw_embeddings[indices].astype(np.float32)
        movie_centroids[m] = np.mean(vecs, axis=0)
        
    # FAISS 인덱스 빌드
    titles = list(movie_centroids.keys())
    matrix = np.array([movie_centroids[t] for t in titles], dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(256)
    index.add(matrix)
    
    # 2. 사전 계산된 로직 캐시 로드
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"Run 'precompute_logic.py' first to generate {CACHE_FILE}")
        
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        logic_cache = json.load(f)
        
    return index, titles, movie_centroids, logic_cache

# 전역 로드 (서버 실행 시 1회)
movie_index, movie_titles, movie_centroids, logic_cache = load_resources()

def get_common_logic(target_info, sim_info):
    """두 영화의 미리 계산된 정보(Top 20 키워드)에서 교집합 찾기"""
    # 단순화: 같은 type과 name을 가진 요소를 찾고 weight 합산
    target_set = {(x['type'], x['name']) for x in target_info}
    sim_set = {(x['type'], x['name']) for x in sim_info}
    
    common = target_set.intersection(sim_set)
    # 리스트로 변환
    result = []
    for c in common:
        result.append({'type': c[0], 'name': c[1]})
    return result[:5] # Top 5

def recommend_fast(target_movie):
    if target_movie not in movie_centroids:
        return "Movie not found"
        
    # 1. FAISS 검색 (0.01초)
    q_vec = movie_centroids[target_movie].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q_vec)
    D, I = movie_index.search(q_vec, k=4)
    
    results = []
    target_logic = logic_cache.get(target_movie, [])
    
    # 2. 로직 매칭 (메모리 연산 -> 0.001초)
    for dist, idx in zip(D[0], I[0]):
        sim_movie = movie_titles[idx]
        if sim_movie == target_movie: continue
        
        sim_logic = logic_cache.get(sim_movie, [])
        commons = get_common_logic(target_logic, sim_logic)
        
        traits = [f"{c['name']}({c['type']})" for c in commons]
        results.append({
            "title": sim_movie,
            "score": float(dist),
            "reason": traits
        })
        
    return results

def generate_report_lazy(target_movie, recommendations):
    """GPT 호출 (사용자 요청 시에만 실행)"""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt_text = ""
    for rec in recommendations:
        prompt_text += f"- {rec['title']} (유사도: {rec['score']:.4f})\n  공통점: {', '.join(rec['reason'])}\n"
        
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "영화 추천 분석가입니다."},
            {"role": "user", "content": f"'{target_movie}' 추천 결과:\n{prompt_text}\n이 영화들을 추천한 이유를 서사적으로 짧게 요약해줘."}
        ]
    )
    return response.choices[0].message.content

# --- 실행 예시 ---
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    target = "Aftersun 2022"
    print(f"🚀 Rapid Recommendation for: {target}")
    
    # 1. 추천 (즉시)
    recs = recommend_fast(target)
    for r in recs:
        print(f"▶ {r['title']} ({r['score']:.4f}) - {r['reason']}")
        
    end_time = time.time()
    print(f"\n⏱️ Total Execution Time: {end_time - start_time:.4f} sec")
    
    # 2. (옵션) 리포트 생성
    # print("\nWriting Report...")
    # print(generate_report_lazy(target, recs))
