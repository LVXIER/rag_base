import json
from pathlib import Path
from collections import defaultdict

from rrf import HybridRetriever
from reranker import Reranker

BASE = Path(r"D:\人工智能集中计划\rag3_min_test\pythonProject")
EVAL_SET_PATH = BASE / "data/output/questions.json"

CHROMA_DIR = BASE / "chroma_db"
BM25_PATH = BASE / "bm25_index.pkl"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "ngsoc_statements"

TOP_K = 7   # 评估的 K，算 Recall@5


# ============ 指标函数 ============

def recall_at_k(gold_ids, retrieved_ids, k):
    """金标 id 是否出现在前 k 个检索结果里，返回 0/1"""
    top_k = retrieved_ids[:k]
    return 1.0 if any(gid in top_k for gid in gold_ids) else 0.0


def reciprocal_rank(gold_ids, retrieved_ids):
    """第一个正确结果的倒数排名；没命中返回 0"""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            
            return 1.0 / rank
    return 0.0


def evaluate_one_mode(retriever, reranker, eval_set, mode, top_k=8):
    """
    mode: "rrf" 或 "rrf+rerank"
    返回 (overall_metrics, per_type_metrics, per_query_details)
    """
    recalls = []
    rrs = []
    per_type_recall = defaultdict(list)
    per_type_rr = defaultdict(list)
    details = []

    for q in eval_set:
        query = q["query"]
        gold_ids = q["gold_doc_ids"]
        qtype = q.get("question_type", "unknown")

        # 1. 两路召回
        vec = retriever.vector_search(query, top_k=10)
        bm25 = retriever.bm25_search(query, top_k=10)

        # 2. RRF 融合（候选池给大一点，方便重排挑）
        merged = retriever.merge_with_dedup(
            vec, bm25, top_k=10, k=60,
            vector_weight=0.7, bm25_weight=0.3,
        )

        # 3. 按模式决定最终列表
        if mode == "rrf":
            final = merged[:top_k]
        elif mode == "rrf+rerank":
            final = reranker.rerank(query, merged, final_k=top_k, threshold=0.0, rrf_weight=0.0)
        else:
            raise ValueError(f"unknown mode: {mode}")

        retrieved_ids = [c["record_id"] for c in final]

        # 4. 算指标
        r = recall_at_k(gold_ids, retrieved_ids, top_k)
        rr = reciprocal_rank(gold_ids, retrieved_ids)

        recalls.append(r)
        rrs.append(rr)
        per_type_recall[qtype].append(r)
        per_type_rr[qtype].append(rr)

        details.append({
            "question_id": q["question_id"],
            "query": query,
            "gold_doc_ids": gold_ids,
            "retrieved_ids": retrieved_ids,
            "hit": bool(r),
            "rr": rr,
            "question_type": qtype,
        })

    overall = {
        f"Recall@{top_k}": sum(recalls) / len(recalls) if recalls else 0.0,
        "MRR": sum(rrs) / len(rrs) if rrs else 0.0,
        "n": len(recalls),
    }

    per_type = {}
    for qtype in per_type_recall:
        per_type[qtype] = {
            f"Recall@{top_k}": sum(per_type_recall[qtype]) / len(per_type_recall[qtype]),
            "MRR": sum(per_type_rr[qtype]) / len(per_type_rr[qtype]),
            "n": len(per_type_recall[qtype]),
        }

    return overall, per_type, details


# ============ 报告输出 ============

def print_comparison(rrf_overall, rerank_overall, top_k):
    print("\n" + "=" * 70)
    print("Overall Comparison")
    print("=" * 70)
    print(f"{'Metric':<15} {'RRF only':<15} {'RRF + Rerank':<15} {'Δ':<10}")
    print("-" * 70)

    for metric in [f"Recall@{top_k}", "MRR"]:
        a = rrf_overall[metric]
        b = rerank_overall[metric]
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<15} {a:<15.4f} {b:<15.4f} {sign}{delta:<10.4f}")

    print(f"{'样本数':<15} {rrf_overall['n']:<15} {rerank_overall['n']:<15}")
    print("=" * 70)


def print_per_type(rrf_per_type, rerank_per_type, top_k):
    print("\n" + "=" * 70)
    print("Per Question Type")
    print("=" * 70)
    types = sorted(set(rrf_per_type) | set(rerank_per_type))
    print(f"{'题型':<12} {'n':<5} {'RRF Recall':<12} {'Rerank Recall':<14} {'RRF MRR':<10} {'Rerank MRR':<10}")
    print("-" * 70)
    for t in types:
        a = rrf_per_type.get(t, {})
        b = rerank_per_type.get(t, {})
        n = a.get("n", b.get("n", 0))
        ar = a.get(f"Recall@{top_k}", 0.0)
        br = b.get(f"Recall@{top_k}", 0.0)
        am = a.get("MRR", 0.0)
        bm = b.get("MRR", 0.0)
        print(f"{t:<12} {n:<5} {ar:<12.4f} {br:<14.4f} {am:<10.4f} {bm:<10.4f}")
    print("=" * 70)


def print_failures(details, mode_name, top_k, limit=5):
    """打印没命中的 query，方便归因"""
    misses = [d for d in details if not d["hit"]]
    print(f"\n[{mode_name}] Missed {len(misses)} / {len(details)} queries")
    for d in misses[:limit]:
        print(f"  - {d['question_id']} ({d['question_type']}): {d['query'][:50]}")
        print(f"    gold={d['gold_doc_ids']}, retrieved={d['retrieved_ids']}")


def save_details(details, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"Details saved to {path}")


# ============ 主流程 ============

def main():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)
    print(f"Eval set size: {len(eval_set)}")

    retriever = HybridRetriever(
        chroma_dir=CHROMA_DIR,
        bm25_path=BM25_PATH,
        model_name=EMBEDDING_MODEL,
        collection_name=COLLECTION_NAME,
    )
    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("\nRunning RRF-only...")
    rrf_overall, rrf_per_type, rrf_details = evaluate_one_mode(
        retriever, reranker, eval_set, mode="rrf", top_k=TOP_K
    )

    print("Running RRF + Rerank...")
    rerank_overall, rerank_per_type, rerank_details = evaluate_one_mode(
        retriever, reranker, eval_set, mode="rrf+rerank", top_k=TOP_K
    )

    print_comparison(rrf_overall, rerank_overall, TOP_K)
    print_per_type(rrf_per_type, rerank_per_type, TOP_K)
    print_failures(rrf_details, "RRF only", TOP_K)
    print_failures(rerank_details, "RRF + Rerank", TOP_K)

    save_details(rrf_details, BASE / "eval_rrf_details.json")
    save_details(rerank_details, BASE / "eval_rerank_details.json")

    # 保存汇总
    summary = {
        "top_k": TOP_K,
        "rrf_only": rrf_overall,
        "rrf_rerank": rerank_overall,
        "per_type_rrf": rrf_per_type,
        "per_type_rerank": rerank_per_type,
    }
    with open(BASE / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {BASE / 'eval_summary.json'}")


if __name__ == "__main__":
    main()