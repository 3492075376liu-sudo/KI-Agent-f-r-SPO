from pathlib import Path
import json
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ===== 路径配置 / Pfad-Konfiguration =====
CHUNK_INPUT_PATH = r"D:\KI-Agent\out\besser Chunk.jsonl"
EMB_OUTPUT_PATH = r"D:\KI-Agent\out\MM_SPO13_embeddings.npy"
MAPPING_OUTPUT_PATH = r"D:\KI-Agent\out\MM_SPO13_mapping.jsonl"
FAISS_INDEX_PATH = r"D:\KI-Agent\out\MM_SPO13_faiss.index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 这个模型支持多语言（包括德语），速度和效果都比较平衡。
# Dieses Modell unterstützt mehrere Sprachen (inkl. Deutsch) und ist relativ schnell.


def load_chunks(path: str) -> List[Dict]:
    """
    从 JSONL 文件中读取所有 chunks（每行一个 JSON）。
    Liest alle Chunks aus einer JSONL-Datei (eine Zeile = ein JSON-Objekt).
    """
    chunks: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks

# 这个函数负责读取你已经“修好的” besser Chunk 文件，返回一个字典列表。
# Diese Funktion lädt die bereinigte Chunk-Datei und gibt eine Liste von Dicts zurück.


def encode_chunks(chunks: List[Dict], model_name: str) -> np.ndarray:
    """
    使用 SentenceTransformer 模型，对每个 chunk["text"] 计算向量。
    Nutzt ein SentenceTransformer-Modell, um für jedes chunk["text"]
    einen Embedding-Vektor zu berechnen.
    """
    model = SentenceTransformer(model_name)
    texts = [ch.get("text", "") for ch in chunks]

    print(f"🧠 Embeddings werden berechnet... (Anzahl Texte: {len(texts)})")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 归一化，方便用 Inner Product≈Cosine
    )
    return embeddings

# 这个函数会加载预训练模型，把所有 chunk 的 text 转成一个 N×D 的向量矩阵。
# Die Funktion lädt ein vortrainiertes Modell und wandelt alle Texte der Chunks
# in eine N×D-Embedding-Matrix um.


def save_embeddings(embeddings: np.ndarray, path: str):
    """
    保存 embedding 矩阵为 .npy 文件。
    Speichert die Embedding-Matrix als .npy-Datei.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)

# 把所有向量存到磁盘，后面建索引或调试都可以直接载入这个 .npy。
# Speichert die Embeddings auf der Platte, damit sie später für den Index
# oder Debugging wieder geladen werden können.


def save_mapping(chunks: List[Dict], path: str):
    """
    保存 index → chunk 信息的映射表为 JSONL：
    每一行包含：idx, id, page, section, doc。

    Speichert die Zuordnung index → Chunk-Infos als JSONL:
    jede Zeile enthält: idx, id, page, section, doc.
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
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

# 这个函数会为每一行向量记录它对应哪个 chunk（id / page / section / doc），
# 以后检索时通过 idx 就能找到原始文本。
# Diese Funktion legt für jeden Embedding-Vektor fest, zu welchem Chunk er gehört,
# sodass man nach der FAISS-Suche über idx wieder auf den Text zugreifen kann.


def build_faiss_index(embeddings: np.ndarray, path: str):
    """
    使用 FAISS 建立向量索引，并保存到磁盘。
    Hier wird ein FAISS-Index mit den Embeddings aufgebaut und auf die Platte gespeichert.
    """
    num_vectors, dim = embeddings.shape
    print(f"📦 Baue FAISS-Index auf. Anzahl Vektoren: {num_vectors}, Dimension: {dim}")

    # 使用 Inner Product（内积）索引，配合归一化后的向量 ≈ 余弦相似度
    # Inner Product Index mit normalisierten Vektoren ≈ Kosinus-Ähnlichkeit
    index = faiss.IndexFlatIP(dim)

    # 向量已经在 encode 时归一化过，可以直接 add
    index.add(embeddings)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))

    print(f"✅ FAISS-Index gespeichert unter: {out_path}")

# 这个函数把 embedding 矩阵丢进 FAISS，创建一个基于内积的索引，并存成 .index 文件。
# Die Funktion erstellt mit FAISS einen Inner-Product-Index aus der Embedding-Matrix
# und speichert ihn als .index-Datei.


def main():
    # 1. 读取 besser Chunk / bereinigte Chunks laden
    chunks = load_chunks(CHUNK_INPUT_PATH)
    print(f"📥 Geladene Chunks: {len(chunks)}")

    if not chunks:
        print("❌ Keine Chunks geladen. Bitte Pfad prüfen.")
        return

    # 2. 计算所有 chunks 的 embeddings / Embeddings berechnen
    embeddings = encode_chunks(chunks, MODEL_NAME)
    print(f"🧩 Embeddings-Form: {embeddings.shape}")  # (N, D)

    # 3. 保存 embedding 矩阵 & mapping / Embeddings & Mapping speichern
    save_embeddings(embeddings, EMB_OUTPUT_PATH)
    save_mapping(chunks, MAPPING_OUTPUT_PATH)
    print(f"💾 Embeddings gespeichert: {EMB_OUTPUT_PATH}")
    print(f"💾 Mapping gespeichert:    {MAPPING_OUTPUT_PATH}")

    # 4. 构建并保存 FAISS 索引 / FAISS-Index aufbauen und speichern
    build_faiss_index(embeddings, FAISS_INDEX_PATH)

    print("\n🎉 Schritt 3 abgeschlossen!")
    print("   -> Chunks wurden vektorisiert und in einem FAISS-Index gespeichert.")


if __name__ == "__main__":
    main()
