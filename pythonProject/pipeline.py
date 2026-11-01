# pipeline.py
from rrf import (
    HybridRetriever, CHROMA_DIR, BM25_PATH,
    EMBEDDING_MODEL, COLLECTION_NAME, print_result_table,
)
from reranker import Reranker, print_rerank_table


class SearchPipeline:
    """编排层：只负责把召回结果喂给重排，不含任何检索/重排算法。"""

    def __init__(self):
        self.retriever = HybridRetriever(
            chroma_dir=CHROMA_DIR,
            bm25_path=BM25_PATH,
            model_name=EMBEDDING_MODEL,
            collection_name=COLLECTION_NAME,
        )
        self.reranker = Reranker()

    def search(self, query, recall_k=20, rrf_keep=20, final_k=5,
               rrf_weight_for_rerank=0.0):
        # 1) 混合召回 + RRF（粗排）——先多留一些
        rrf_result = self.retriever.hybrid_search(
            query=query,
            recall_k=recall_k,
            final_k=rrf_keep,      # 注意：这是 RRF 阶段保留数，不是最终数
        )

        # 2) 重排（精排）——直接把 merged 传进去
        candidates = rrf_result["merged"]
        final = self.reranker.rerank(
            query=query,
            candidates=candidates,
            final_k=final_k,
            threshold=0.0,
            rrf_weight=rrf_weight_for_rerank,
        )
        return {"rrf": rrf_result, "final": final}


if __name__ == "__main__":
    pipe = SearchPipeline()
    queries = [
        "ZTNA 不支持与哪个产品联动?",
        "工作空间内剪切板数据怎么导出到空间外?",
        "零信任三层应用需要客户端吗?",
    ]
    for q in queries:
        out = pipe.search(q, recall_k=20, rrf_keep=20, final_k=5)
        print_result_table(out["rrf"])                    # 召回阶段表
        print_rerank_table(q, out["rrf"]["merged"], out["final"])  # 重排对比表