## 最小校准+向量化+入库+BM25
import json
import hashlib
import pickle
from pathlib import Path

import jieba
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

BASE = Path(r"D:\人工智能集中计划\rag3_min_test\pythonProject")

STATEMENTS_PATH = BASE / "data/output/statements.json"
CHROMA_DIR = BASE / "chroma_db"
BM25_PATH = BASE / "bm25_index.pkl"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 跑通用，后面换 BAAI/bge-m3
COLLECTION_NAME = "ngsoc_statements"
MIN_CONTENT_LEN = 10


def load_and_validate(path):
    """读 statements.json，做最小校验，返回 (accepted, rejected)"""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    accepted, rejected = [], []
    seen_ids = set()

    for i, r in enumerate(records):
        errors = []
        for field in ["record_id", "content", "content_type", "token_count", "content_hash", "source_file", "source_locator"]:
            if not r.get(field):
                errors.append(f"missing_{field}")

        content = r.get("content", "").strip()
        if len(content) < MIN_CONTENT_LEN:
            errors.append("content_too_short")

        rid = r.get("record_id")
        if rid and rid in seen_ids:
            errors.append("duplicate_record_id")
        elif rid:
            seen_ids.add(rid)

        if errors:
            rejected.append({"index": i, "record_id": rid, "errors": errors})
        else:
            accepted.append(r)

    return accepted, rejected


def build_index(statements):
    """向量化 + 写 Chroma + 建 BM25"""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    contents = [s["content"] for s in statements]
    print(f"Encoding {len(contents)} statements...")
    embeddings = model.encode(contents, batch_size=16, show_progress_bar=True).tolist()

    # ---- Chroma ----
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # 每次全量重建，避免旧数据残留
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[s["record_id"] for s in statements],
        embeddings=embeddings,
        documents=contents,
        metadatas=[{
            "content_type": s["content_type"],
            "source_file": s["source_file"],
            "source_locator": s["source_locator"],
            "token_count": s["token_count"],
        } for s in statements],
    )
    print(f"Chroma written: {collection.count()} records")

    # ---- BM25 ----
    tokenized = [list(jieba.cut(s["content"])) for s in statements]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "record_ids": [s["record_id"] for s in statements],
            "contents": contents,
            "metadatas": [{
                "content_type": s["content_type"],
                "source_file": s["source_file"],
                "source_locator": s["source_locator"],
            } for s in statements],
        }, f)
    print(f"BM25 written: {BM25_PATH}")

    return collection


def main():
    statements, rejected = load_and_validate(STATEMENTS_PATH)

    print(f"Total records: {len(statements) + len(rejected)}")
    print(f"Accepted: {len(statements)}")
    print(f"Rejected: {len(rejected)}")
    if rejected:
        for r in rejected[:5]:
            print(f"  rejected #{r['index']} {r['record_id']}: {r['errors']}")

    if not statements:
        print("No accepted records, abort.")
        return

    build_index(statements)
    print("Ingest done.")


if __name__ == "__main__":
    main()