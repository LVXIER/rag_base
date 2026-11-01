##这个是到检索了，rrf融合得分（粗排）。

import pickle
from pathlib import Path

import jieba
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(r"D:\人工智能集中计划\rag3_min_test\pythonProject")
CHROMA_DIR = BASE / "chroma_db"
BM25_PATH = BASE / "bm25_index.pkl"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 必须和 ingest 时一致
COLLECTION_NAME = "ngsoc_statements"


# ============ 1. 初始化：加载模型、Chroma、BM25 ============

class HybridRetriever:
    def __init__(self, chroma_dir, bm25_path, model_name, collection_name):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Loading Chroma...")
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_collection(collection_name)

        print("Loading BM25 index...")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_ids = data["record_ids"]
        self.bm25_contents = data["contents"]
        self.bm25_metadatas = data["metadatas"]

        print("Ready.\n")

    # ---------- 向量召回 ----------
    def vector_search(self, query, top_k=10):
        q_vec = self.model.encode([query]).tolist()
        res = self.collection.query(
            query_embeddings=q_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for i in range(len(res["ids"][0])):
            results.append({
                "record_id": res["ids"][0][i],
                "score": 1 - res["distances"][0][i],  # cosine distance -> similarity
                "content": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
            })
        return results

    # ---------- BM25 召回 ----------
    def bm25_search(self, query, top_k=10):
        tokenized_q = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_q)

        # 按分数降序取 top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            results.append({
                "record_id": self.bm25_ids[idx],
                "score": float(score),
                "content": self.bm25_contents[idx],
                "metadata": self.bm25_metadatas[idx],
            })
        return results

    # ---------- RRF 融合（手搓核心） ----------
    @staticmethod
    def reciprocal_rank_fusion(ranked_lists, k=60, weights=None):
        """
        ranked_lists: [[id1, id2, ...], [id1, id3, ...]] 每路召回的 id 有序列表
        k: RRF 平滑常数
        weights: 每路权重
        返回: [(record_id, rrf_score), ...] 降序
        """
        if weights is None:
            weights = [1.0] * len(ranked_lists)

        scores = {}
        for list_idx, ranked in enumerate(ranked_lists):
            w = weights[list_idx]
            for rank, rid in enumerate(ranked, start=1):
                scores[rid] = scores.get(rid, 0.0) + w * (1.0 / (k + rank))

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ---------- 融合 + 去重 + 截断 ----------
    def merge_with_dedup(self, vector_results, bm25_results, top_k=10,
                         k=60, vector_weight=0.7, bm25_weight=0.3):
        vec_ids = [r["record_id"] for r in vector_results]
        bm25_ids = [r["record_id"] for r in bm25_results]

        fused = self.reciprocal_rank_fusion(
            [vec_ids, bm25_ids],
            k=k,
            weights=[vector_weight, bm25_weight],
        )

        # id -> 原始结果，用于补回 content 和 metadata
        id_to_result = {}
        for r in vector_results:
            id_to_result[r["record_id"]] = r
        for r in bm25_results:
            if r["record_id"] not in id_to_result:
                id_to_result[r["record_id"]] = r

        merged = []
        for rid, rrf_score in fused[:top_k]:
            original = id_to_result[rid]
            if rid in vec_ids and rid in bm25_ids:
                source = "both"
            elif rid in vec_ids:
                source = "vector"
            else:
                source = "bm25"
            merged.append({
                "record_id": rid,
                "rrf_score": rrf_score,
                "source": source,
                "content": original["content"],
                "metadata": original["metadata"],
            })
        return merged

    # ---------- 完整混合检索入口 ----------
    def hybrid_search(self, query, recall_k=10, final_k=5, k=60,
                      vector_weight=0.7, bm25_weight=0.3):
        vec_results = self.vector_search(query, top_k=recall_k)
        bm25_results = self.bm25_search(query, top_k=recall_k)
        merged = self.merge_with_dedup(
            vec_results, bm25_results,
            top_k=final_k, k=k,
            vector_weight=vector_weight, bm25_weight=bm25_weight,
        )
        return {
            "query": query,
            "vector_hits": len(vec_results),
            "bm25_hits": len(bm25_results),
            "merged": merged,
        }


# ============ 2. 输出融合后的表 ============

def print_result_table(result):
    print(f"\nQuery: {result['query']}")
    print(f"Vector hits: {result['vector_hits']} | BM25 hits: {result['bm25_hits']}")
    print("-" * 100)

    rows = []
    for rank, item in enumerate(result["merged"], start=1):
        content_preview = item["content"].replace("\n", " ")[:60]
        rows.append({
            "排名": rank,
            "record_id": item["record_id"],
            "RRF分数": round(item["rrf_score"], 5),
            "来源": item["source"],
            "内容摘要": content_preview + ("..." if len(item["content"]) > 60 else ""),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print("-" * 100)


# ============ 3. 演示 ============

if __name__ == "__main__":
    retriever = HybridRetriever(
        chroma_dir=CHROMA_DIR,
        bm25_path=BM25_PATH,
        model_name=EMBEDDING_MODEL,
        collection_name=COLLECTION_NAME,
    )

    queries = [
        "ZTNA 不支持与哪个产品联动?",
        "工作空间内剪切板数据怎么导出到空间外?",
        "零信任三层应用需要客户端吗?",
    ]

    for q in queries:
        result = retriever.hybrid_search(
            query=q,
            recall_k=10,      # 每路召回 10 条
            final_k=5,        # 融合后保留 5 条
            k=60,             # RRF 平滑常数
            vector_weight=0.7,
            bm25_weight=0.3,
        )
        print_result_table(result)