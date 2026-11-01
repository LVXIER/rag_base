from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Reranker ready.\n")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        final_k: int = 5,
        threshold: float = 0.0,
        rrf_weight: float = 0.0,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        candidates: merge_with_dedup 的输出，每条含 record_id / rrf_score / content / metadata / source
        final_k: 最终保留几条
        threshold: 重排分数低于此值直接丢弃
        rrf_weight: 0 表示纯重排；>0 表示混合 RRF 排名信号
        rrf_k: 混合时 RRF 排名的平滑常数
        """
        if not candidates:
            return []

        # 1. 构造 (query, content) 对
        pairs = [(query, c["content"]) for c in candidates]

        # 2. 批量打分
        scores = self.model.predict(pairs)  # numpy array，与 pairs 一一对应

        # 3. 写回 rerank_score，并可选混合 RRF 排名信号
        for rank, (c, s) in enumerate(zip(candidates, scores), start=1):
            c["rerank_score"] = float(s)
            if rrf_weight > 0:
                # 用 RRF 中的排名（不是 rrf_score 绝对值）做混合
                # 排名越靠前，1/(rrf_k + rank) 越大
                rrf_signal = 1.0 / (rrf_k + rank)
                c["final_score"] = (1 - rrf_weight) * c["rerank_score"] + rrf_weight * rrf_signal
            else:
                c["final_score"] = c["rerank_score"]

        # 4. 按 final_score 降序
        reranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

        # 5. 阈值截断
        filtered = [c for c in reranked if c["rerank_score"] >= threshold]

        # 6. 截到 final_k
        return filtered[:final_k]


def print_rerank_table(query, before, after):
    """打印重排前后的对比表，用于肉眼验证"""
    print(f"\nQuery: {query}")
    print("=" * 110)

    print("\n[RRF 融合后]")
    print(f"{'排名':<4} {'record_id':<14} {'rrf_score':<12} {'source':<8} 内容摘要")
    for i, c in enumerate(before, start=1):
        preview = c["content"].replace("\n", " ")[:50]
        print(f"{i:<4} {c['record_id']:<14} {c['rrf_score']:<12.5f} {c['source']:<8} {preview}")

    print("\n[重排后]")
    print(f"{'排名':<4} {'record_id':<14} {'rerank':<10} {'final':<10} 内容摘要")
    for i, c in enumerate(after, start=1):
        preview = c["content"].replace("\n", " ")[:50]
        print(f"{i:<4} {c['record_id']:<14} {c['rerank_score']:<10.4f} {c['final_score']:<10.4f} {preview}")

    print("=" * 110)


if __name__ == "__main__":
    # 示例：模拟一个 RRF 表
    query = "ZTNA 不支持与哪个产品联动?"
    rrf_table = [
        {"record_id": "p1_001_k", "rrf_score": 0.01639, "source": "both",
         "content": "针对问题「ZTNA 不支持与哪个产品联动?」，正确答案是：A.VPN。",
         "metadata": {}},
        {"record_id": "p1_025_k", "rrf_score": 0.01563, "source": "both",
         "content": "针对问题「ZTNA多因子认证中，哪种认证方式不支持?」，正确答案是：D.扫码认证。",
         "metadata": {}},
        {"record_id": "p1_004_k", "rrf_score": 0.01129, "source": "vector",
         "content": "针对问题「ZTNA 的短信认证支持哪种认证方式?」，正确答案是：B.阿里云。",
         "metadata": {}},
        {"record_id": "p1_006_k", "rrf_score": 0.00781, "source": "bm25",
         "content": "针对问题「如果用户要开启 ZTNA 的 SPA 敲门功能...」，正确答案是：C.UDP+TCP SPA 端口敲门。",
         "metadata": {}},
        {"record_id": "p1_030_k", "rrf_score": 0.00769, "source": "vector",
         "content": "针对问题「关于零信任应用发布，以下说法错误的是...」，正确答案是：D.零信任支持WEB应用开启SPA访问。",
         "metadata": {}},
    ]

    reranker = Reranker()
    after = reranker.rerank(
        query=query,
        candidates=rrf_table,
        final_k=3,
        threshold=0.0,      # 先不截断，看分数分布
        rrf_weight=0.0,     # 先纯重排
    )
    print_rerank_table(query, rrf_table, after)