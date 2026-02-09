from pathlib import Path
import json
import re
from typing import List, Dict

# ===== 路径配置 / Pfad-Konfiguration =====
INPUT_PATH = r"D:\KI-Agent\out\AG chunk.jsonl"          # 现有的粗 chunk 文件
OUTPUT_PATH = r"D:\KI-Agent\out\AG-besser chunk.jsonl"  # 输出更细的 chunk 文件

MAX_CHARS = 350  # 每个 chunk 最大字符数上限 / maximale Zeichenlänge pro Chunk


def load_chunks(path: str) -> List[Dict]:
    """
    从 JSONL 文件中读取所有 chunk（每行一个 JSON）。
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

# 这个函数负责把现有的 AG chunk 文件全部读进来，变成 Python 字典列表。
# Diese Funktion lädt die bestehende AG-Chunk-Datei und gibt eine Liste von Dicts zurück.


def preprocess_text(text: str) -> str:
    """
    在分句前做一点预处理：
    - 在 (1) (2) (3)... 这种编号前面强行断开一行，让它们单独成一句的开头。

    Kleine Vorverarbeitung vor dem Satz-Splitting:
    - Vor Nummerierungen wie (1) (2) (3) wird ein Zeilenumbruch eingefügt,
      damit sie einen eigenen Satzanfang bilden.
    """
    text = re.sub(r'\s*(\(\d+\))', r'\n\1', text)
    return text

# 这个函数的目的是把条款编号 (1)(2)(3)... 从上一句拆开，避免黏在前一句后面。
# Ziel dieser Funktion ist es, Nummerierungen (1)(2)(3)... vom vorherigen Satz zu trennen.


def split_into_sentences(text: str) -> List[str]:
    """
    把文本拆成句子列表：
    - 先预处理编号 (1)(2)...
    - 再按换行和 . ? ! 后的空格分句。

    Teilt den Text in eine Satzliste:
    - Vorverarbeitung der Nummerierungen (1)(2)...
    - Danach Split anhand von Zeilenumbrüchen und . ? ! gefolgt von Leerzeichen.
    """
    text = preprocess_text(text)

    rough_parts = re.split(r'\n+', text)
    sentences: List[str] = []

    for part in rough_parts:
        part = part.strip()
        if not part:
            continue
        sub = re.split(r'(?<=[\.\?\!])\s+', part)
        for s in sub:
            s = s.strip()
            if s:
                sentences.append(s)

    return sentences

# 这里实现了按标点切句的逻辑：先按换行粗分，再按 . ? ! 分成短句。
# Hier wird ein einfaches Satz-Splitting umgesetzt.


def split_long_sentence_by_length(sentence: str, max_chars: int = MAX_CHARS) -> List[str]:
    """
    如果某个句子太长（> max_chars），再按长度切一刀：
    - 尽量在空格处分段；
    - 实在没有空格，就硬切。

    Wenn ein Satz zu lang ist (> max_chars), wird er weiter aufgeteilt:
    - bevorzugt an Leerzeichen;
    - notfalls harter Cut.
    """
    s = sentence.strip()
    if len(s) <= max_chars:
        return [s]

    parts: List[str] = []
    start = 0
    L = len(s)
    while start < L:
        end = min(start + max_chars, L)
        # 在 start~end 范围内找最后一个空格
        split_pos = s.rfind(" ", start, end)
        if split_pos == -1 or split_pos <= start:
            split_pos = end
        part = s[start:split_pos].strip()
        if part:
            parts.append(part)
        start = split_pos
    return parts

# 这个函数是给“超长句子”用的二次切割，保证每一小块不会超过 MAX_CHARS 字符。
# Diese Funktion dient dazu, sehr lange Sätze weiter zu zerlegen.


def refine_chunk_text(text: str) -> List[str]:
    """
    对一个原始 chunk 的 text 做二次细切：
    1. 先按句子拆分；
    2. 对每个句子，如果太长，再按最大长度拆分。

    Verfeinert den Text eines ursprünglichen Chunks:
    1. in Sätze teilen;
    2. zu lange Sätze weiter nach max_chars aufsplitten.
    """
    sentences = split_into_sentences(text)
    refined: List[str] = []
    for sent in sentences:
        small_parts = split_long_sentence_by_length(sent, max_chars=MAX_CHARS)
        refined.extend(small_parts)
    return refined

# 这个函数实现了“再切一刀”的核心逻辑：先按句号切成句子，再把超长句子按长度切。
# Diese Funktion bildet das Herzstück der Verfeinerung.


def refine_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    把原来的 chunk 列表变成更细的 chunk 列表：
    - 每个旧 chunk 可能拆成多个新的 chunk；
    - 元信息 doc/page/section/type 继承；
    - id 在原 id 基础上加 _1/_2/...。

    Wandelt die ursprüngliche Chunk-Liste in eine feinere Liste um:
    - Ein alter Chunk kann in mehrere neue aufgeteilt werden;
    - Metadaten (doc/page/section/type) werden übernommen;
    - id bekommt Suffixe _1/_2/... je nach Teilstück.
    """
    new_chunks: List[Dict] = []

    for ch in chunks:
        base_id = ch.get("id", "chunk")
        doc = ch.get("doc")
        page = ch.get("page")
        section = ch.get("section")
        ctype = ch.get("type", "paragraph")
        text = ch.get("text", "")

        parts = refine_chunk_text(text)
        if len(parts) == 0:
            continue

        # 如果只切成 1 段，可以保持原 id；如果多段就加后缀
        if len(parts) == 1:
            new_chunks.append({
                "id": base_id,
                "doc": doc,
                "page": page,
                "type": ctype,
                "section": section,
                "text": parts[0],
            })
        else:
            for i, t in enumerate(parts, start=1):
                new_id = f"{base_id}_{i:02d}"
                new_chunks.append({
                    "id": new_id,
                    "doc": doc,
                    "page": page,
                    "type": ctype,
                    "section": section,
                    "text": t,
                })

    return new_chunks

# 这个函数遍历所有旧 chunk，对每一个做 refine_chunk_text，
# 然后生成带新 id 的小 chunk。
# Diese Funktion iteriert über alle alten Chunks und erzeugt daraus
# verfeinerte Chunks mit neuen IDs.


def main():
    # 0. 检查输入文件 / Eingabedatei prüfen
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        print(f"❌ Eingabedatei existiert nicht: {in_path}")
        return
    else:
        print(f"📄 Eingabedatei gefunden: {in_path} (Größe: {in_path.stat().st_size} Bytes)")

    # 1. 读取原始 chunks / ursprüngliche Chunks laden
    chunks = load_chunks(INPUT_PATH)
    orig_count = len(chunks)
    print(f"📥 Geladene Chunks (original): {orig_count}")

    if not chunks:
        print("❌ Keine Chunks geladen. Bitte Dateiinhalt prüfen.")
        return

    # 2. 做二次细切 / Chunks verfeinern
    better = refine_chunks(chunks)
    final_count = len(better)
    print(f"✨ Nach Verfeinerung: {final_count} Chunks")

    # 3. 写出到新的 JSONL / in neue JSONL-Datei schreiben
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in better:
            f.write(json.dumps(ch, ensure_ascii=False))
            f.write("\n")

    print("\n✅ Fertig!")
    print(f"   Ausgabedatei: {out_path}")
    print(f"\n📊 统计 / Statistik:")
    print(f"   原始 Chunk 数量 (original): {orig_count}")
    print(f"   细切后 Chunk 数量 (feiner): {final_count}")

    # 4. 看几个例子 / ein paar Beispiele anzeigen
    print("\n👀 Beispiel-Chunks (erste 10):\n")
    for ch in better[:10]:
        print("----")
        print("id:      ", ch.get("id"))
        print("Seite:   ", ch.get("page"))
        print("Abschnitt:", ch.get("section"))
        print("Länge:   ", len(ch.get("text", "")))
        print("Text:    ", ch.get("text", "")[:200], "...")
        print()


if __name__ == "__main__":
    main()
