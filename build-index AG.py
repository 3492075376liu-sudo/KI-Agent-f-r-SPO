from pathlib import Path
import json
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ===== 路径配置 =====
CHUNK_INPUT_PATH = r"D:\KI-Agent\out\AG-besser chunk.jsonl"

EMB_OUTPUT_PATH = r"D:\KI-Agent\out\AG_besser_embeddings.npy"
MAPPING_OUTPUT_PATH = r"D:\KI-Agent\out\AG_besser_mapping.jsonl"
FAISS_INDEX_PATH = r"D:\KI-Agent\out\AG_besser_faiss.index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(path: str) -> List[Dict]:
    """从 JSONL 文件读取所有 chunks（每行一个 JSON）"""
    chunks: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败：{path} 第 {line_no} 行：{e}") from e
    return chunks


def _get_text(ch: Dict) -> str:
    """兼容不同 chunk 字段名：text / chunk / content"""
    return (ch.get("text") or ch.get("chunk") or ch.get("content") or "").strip()


def encode_chunks(chunks: List[Dict], model_name: str) -> np.ndarray:
    """用 SentenceTransformer 对每个 chunk 文本算向量"""
    model = SentenceTransformer(model_name)
    texts = [_get_text(ch) for ch in chunks]

    print(f"🧠 计算 embeddings...（texts: {len(texts)}）")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 归一化后用 Inner Product ≈ Cosine
    )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    """保存 embedding 矩阵为 .npy"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)


def save_mapping(chunks: List[Dict], path: str) -> None:
    """
    保存 index -> chunk 的映射表（JSONL）
    默认写 idx, id, page, section, doc（字段不存在也没事，保留为 null）
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, ch in enumerate(chunks):
            rec = {
                "idx": idx,
                "id": ch.get("id"),
                "page": ch.get("page"),
                "section": ch.get("section"),
                "doc": ch.get("doc"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_faiss_index(embeddings: np.ndarray, path: str) -> None:
    """建立 FAISS 索引并保存"""
    num_vectors, dim = embeddings.shape
    print(f"📦 建 FAISS index...（vectors: {num_vectors}, dim: {dim}）")

    index = faiss.IndexFlatIP(dim)  # Inner Product + normalize => cosine
    index.add(embeddings)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))

    print(f"✅ FAISS index 已保存：{out_path}")


def main() -> None:
    chunks = load_chunks(CHUNK_INPUT_PATH)
    print(f"📥 chunks 读取完成：{len(chunks)}")

    if not chunks:
        print("❌ chunks 为空：检查 CHUNK_INPUT_PATH 是否正确")
        return

    embeddings = encode_chunks(chunks, MODEL_NAME)
    print(f"🧩 embeddings shape: {embeddings.shape}")  # (N, D)

    save_embeddings(embeddings, EMB_OUTPUT_PATH)
    save_mapping(chunks, MAPPING_OUTPUT_PATH)
    print(f"💾 embeddings: {EMB_OUTPUT_PATH}")
    print(f"💾 mapping:    {MAPPING_OUTPUT_PATH}")

    build_faiss_index(embeddings, FAISS_INDEX_PATH)

    print("\n🎉 完事！你现在可以直接用 faiss.index + mapping 去检索了。")


if __name__ == "__main__":
    main()
