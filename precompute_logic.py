import os
import json
import torch
import numpy as np
import faiss
from collections import defaultdict
from tqdm import tqdm
from neo4j import GraphDatabase

# --- 설정 ---
URI = "neo4j+ssc://2bdf163a.databases.neo4j.io:7687"
AUTH = ("neo4j", "dVRqLgBpDBT3tP37uYphK_zFZRjBHRizDVvRc4LCJRg")
OUTPUT_FILE = "movie_logic_cache.json"

def precompute_all_logic():
    print("🔄 Loading graph data and embeddings...")
    # 1. 데이터 로드 (영화 목록 확보)
    saved = torch.load("full_graph.pt", weights_only=False)
    node_meta = saved['node_meta']
    NODE_TYPE_MAP = saved['NODE_TYPE_MAP']
    
    # 영화 목록 추출
    movies = set()
    for meta in node_meta:
        if meta.get('movie'):
            movies.add(meta['movie'])
    movie_list = list(movies)
    print(f"🎬 Total Movies: {len(movie_list)}")

    # 2. Neo4j 연결 (1회)
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    # 3. 모든 영화 쌍에 대해 미리 계산할 수 없으므로,
    #    각 영화별 '주요 키워드(Trope, Emotion 등)'를 미리 뽑아둠.
    #    (쌍으로 조회하면 N*N이라 너무 오래 걸림 -> 영화별 요약 정보 저장)
    
    movie_structural_info = {}
    
    print("⏳ Pre-computing structural info for each movie...")
    with driver.session() as session:
        for movie in tqdm(movie_list):
            query = """
            MATCH (m:Movie {title: $title})-[:HAS_WINDOW]->(sw:SceneWindow)
            MATCH (sw)-[:HAS_TROPE|IN_PHASE|HAS_EMOTION|HAS_ARCHETYPE]->(target)
            RETURN labels(target)[0] as type, target.name as name, count(*) as weight
            ORDER BY weight DESC LIMIT 20
            """
            result = session.run(query, title=movie).data()
            # 저장 포맷: [{'type': 'Emotion', 'name': 'Sadness', 'weight': 15}, ...]
            movie_structural_info[movie] = result
            
    driver.close()
    
    # 4. 파일 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(movie_structural_info, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Saved structural info to {OUTPUT_FILE}")

if __name__ == "__main__":
    precompute_all_logic()
