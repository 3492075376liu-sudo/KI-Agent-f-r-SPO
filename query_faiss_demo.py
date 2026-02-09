from pathlib import Path
import json
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from dotenv import load_dotenv
import os

# ===== 环境变量 & OpenAI 客户端 / Umgebungsvariablen & OpenAI-Client =====
load_dotenv()  # 从 .env 读取 OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 路径配置 / Pfad-Konfiguration =====
BASE_DIR = Path(r"D:\KI-Agent")

CHUNK_INPUT_PATH = BASE_DIR / "out" / "besser Chunk.jsonl"
FAISS_INDEX_PATH = BASE_DIR / "out" / "MM_SPO13_faiss.index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 必须与构建 FAISS 索引时使用的模型保持一致


# ===== 输出语言规则 / Sprachregeln =====
LANG_RULES = {
    "de": "Antworte ausschließlich auf Deutsch.",
    "zh": "只用中文回答（模块/课程名称保留德语原名）。",
    "en": "Answer only in English (keep German module/course names).",
    "all": "Gib die Antwort dreisprachig in genau dieser Reihenfolge aus: [DE], [中文], [EN]."
}

NO_CONTEXT_MSG = {
    "de": "In den bereitgestellten Unterlagen wurden keine passenden Informationen für diese Frage gefunden.",
    "zh": "在提供的资料中没有找到能直接回答这个问题的内容。",
    "en": "No relevant information was found in the provided materials for this question.",
    "all": "[DE]\nIn den bereitgestellten Unterlagen wurden keine passenden Informationen für diese Frage gefunden.\n\n"
           "[中文]\n在提供的资料中没有找到能直接回答这个问题的内容。\n\n"
           "[EN]\nNo relevant information was found in the provided materials for this question."
}


def load_chunks(path: Path) -> List[Dict]:
    """读取 besser Chunk.jsonl，得到一个列表 dict（text/page/doc/section 等）"""
    chunks: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def load_faiss_index(path: Path) -> faiss.Index:
    """从磁盘读取已经建好的 FAISS 索引"""
    return faiss.read_index(str(path))


def load_model(model_name: str) -> SentenceTransformer:
    """加载 SentenceTransformer 模型（和建索引时同一个）"""
    return SentenceTransformer(model_name)


def search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = 5,
):
    """
    - 把问题编码成向量
    - 在 FAISS 中检索 top_k
    - 用下标取回 chunks 文本和元信息
    """
    query_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    D, I = index.search(query_emb, top_k)

    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        ch = chunks[idx]
        results.append(
            {
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "page": ch.get("page"),
                "section": ch.get("section"),
                "doc": ch.get("doc"),
                "text": ch.get("text", ""),
            }
        )
    return results


def build_context(results: List[Dict], max_chars: int = 3000) -> str:
    """
    把检索到的 chunk 拼成一个上下文字符串（带上 doc/page/section，方便 LLM 引用）
    """
    parts: List[str] = []
    length = 0

    for item in results:
        text = (item.get("text") or "").strip()
        if not text:
            continue

        header = f"[Dokument: {item.get('doc')} | Seite: {item.get('page')} | Abschnitt: {item.get('section')}]"
        block = f"{header}\n{text}"

        if length + len(block) > max_chars:
            break

        parts.append(block)
        length += len(block)

    return "\n\n".join(parts)


def translate_query_to_german(query: str) -> str:
    """
    （可选）把用户问题翻成德语，用于检索，提高召回
    注意：只返回翻译文本，不要解释
    """
    prompt = (
        "Übersetze die folgende Frage ins Deutsche. "
        "Gib NUR die Übersetzung aus, ohne Erklärungen:\n\n"
        f"{query}"
    )
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=120,
    )
    return (resp.output_text or "").strip()


def generate_answer_with_llm(question: str, context: str, output_lang: str) -> str:
    """
    使用 OpenAI LLM，根据“问题 + 上下文”生成回答（语言可切换）
    """
    output_lang = (output_lang or "de").lower()
    if output_lang not in LANG_RULES:
        output_lang = "de"

    if not context.strip():
        return NO_CONTEXT_MSG[output_lang]

    lang_rule = LANG_RULES[output_lang]

    prompt = f"""
Du bist ein Studienassistent. Du darfst NUR auf Basis des gegebenen Kontexts antworten.
Wenn der Kontext keine Antwort enthält, sage das klar (keine Erfindungen).

Wichtig (Ausgabe-Regeln):
- {lang_rule}
- Behalte Modul-/Kursnamen im Original (Deutsch).
- Antworte klar, präzise; wenn sinnvoll mit Stichpunkten.
- Wenn du dir unsicher bist, sag es.

[Kontext]
{context}

[Frage]
{question}
""".strip()

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=512,
    )

    return (response.output_text or "").strip()


def answer_question(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Dict],
    output_lang: str = "de",
    translate_for_retrieval: bool = False,
) -> Dict:
    """
    - (可选) 先翻译为德语用于检索
    - search() 找 chunks
    - build_context()
    - LLM 按 output_lang 输出
    """
    retrieval_query = query
    if translate_for_retrieval:
        try:
            retrieval_query = translate_query_to_german(query)
        except Exception:
            retrieval_query = query  # 翻译失败就退回原始问题

    hits = search(retrieval_query, model, index, chunks, top_k=5)
    context = build_context(hits, max_chars=3000)
    answer_text = generate_answer_with_llm(query, context, output_lang)

    return {
        "answer": answer_text,
        "hits": hits,
        "retrieval_query": retrieval_query,
    }


def main():
    print("📥 Lade Modell, Index und Chunks...")
    chunks = load_chunks(CHUNK_INPUT_PATH)
    print(f"   ➜ Anzahl der Chunks: {len(chunks)}")

    index = load_faiss_index(FAISS_INDEX_PATH)
    print("   ➜ FAISS-Index erfolgreich geladen")

    model = load_model(MODEL_NAME)
    print("   ➜ Embedding-Modell erfolgreich geladen")

    print("✅ Initialisierung abgeschlossen – du kannst jetzt Fragen stellen!\n")

    mode = input("Modus wählen: 1 = nur Suche, 2 = Frage-Antwort mit LLM (Standard 1): ").strip() or "1"

    output_lang = input("Ausgabe-Sprache wählen (de/zh/en/all) [de]: ").strip().lower() or "de"
    if output_lang not in LANG_RULES:
        output_lang = "de"

    translate_for_retrieval = False  # 默认关闭：需要就 /retrieval on

    print("\n💡 快捷指令：")
    print("   /lang de|zh|en|all    → 切换输出语言")
    print("   /retrieval on|off     → 是否把问题先翻成德语再检索（提升召回）")
    print("   /help                 → 查看指令")
    print("   q                      → 退出\n")

    while True:
        query = input("Bitte gib eine Frage ein (q zum Beenden): ").strip()
        if not query:
            continue
        if query.lower() == "q":
            print("👋 Suche wird beendet.")
            break

        # ===== 命令处理 =====
        if query.startswith("/help"):
            print("\nCommands:")
            print("  /lang de|zh|en|all")
            print("  /retrieval on|off")
            print("  q (quit)\n")
            continue

        if query.startswith("/lang"):
            parts = query.split()
            if len(parts) >= 2 and parts[1].lower() in LANG_RULES:
                output_lang = parts[1].lower()
                print(f"✅ 输出语言已切换为: {output_lang}\n")
            else:
                print("❌ 用法：/lang de|zh|en|all\n")
            continue

        if query.startswith("/retrieval"):
            parts = query.split()
            if len(parts) >= 2 and parts[1].lower() in ["on", "off"]:
                translate_for_retrieval = (parts[1].lower() == "on")
                state = "ON" if translate_for_retrieval else "OFF"
                print(f"✅ retrieval 翻译模式: {state}\n")
            else:
                print("❌ 用法：/retrieval on|off\n")
            continue

        print("\n" + "=" * 80)

        if mode == "2":
            result = answer_question(
                query=query,
                model=model,
                index=index,
                chunks=chunks,
                output_lang=output_lang,
                translate_for_retrieval=translate_for_retrieval,
            )

            print("【LLM Antwort / 回答 / Answer】")
            print(result["answer"])

            # 可选：展示检索实际用的 query（开了翻译时很有用）
            if translate_for_retrieval:
                print("\n[Retrieval-Query (DE)]")
                print(result["retrieval_query"])

            print("\nVerwendete Textausschnitte (Evidenz) / 使用到的原文片段（证据）:")
            print("-" * 80)
            for h in result["hits"]:
                print(
                    f"[{h['rank']}] Ähnlichkeit: {h['score']:.4f} | "
                    f"Dokument: {h.get('doc')} | Seite: {h.get('page')} | Abschnitt: {h.get('section')}"
                )
                print(h["text"])
                print("-" * 80)
        else:
            hits = search(query, model, index, chunks, top_k=5)
            if not hits:
                print("😶 Keine passenden Inhalte gefunden.")
            else:
                for h in hits:
                    print(
                        f"[{h['rank']}] Ähnlichkeit: {h['score']:.4f} | "
                        f"Dokument: {h.get('doc')} | Seite: {h.get('page')} | Abschnitt: {h.get('section')}"
                    )
                    print(h["text"])
                    print("-" * 80)

        print()


if __name__ == "__main__":
    main()
