from pathlib import Path
import json
from typing import List, Dict

# ===== 路径配置 / Pfad-Konfiguration =====
INPUT_PATH = r"D:\KI-Agent\out\Maschinenbau_und_Mechatronik_chunks.jsonl"
OUTPUT_PATH = r"D:\KI-Agent\out\besser Chunk.jsonl"

SHORT_THRESHOLD = 25   # 字符数 < 这个值就认为太短 / Chars < THRESHOLD => "zu kurz"
LONG_THRESHOLD = 500   # 字符数 > 这个值就认为太长 / Chars > THRESHOLD => "zu lang"


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

# 这个函数负责把现有的 chunk 文件全部读进来，变成 Python 字典列表。
# Diese Funktion lädt die bestehende Chunk-Datei und gibt eine Liste von Dicts zurück.


def merge_short_chunks(chunks: List[Dict],
                       short_threshold: int = SHORT_THRESHOLD) -> List[Dict]:
    """
    合并过短的 chunk：
    - 如果一个 chunk 的 text 很短（< short_threshold），
      优先把它并到“下一个同页同 section 的 chunk”前面；
      如果没有下一个，就并到前一个；
      实在都没有，就保留原样。

    Fügt zu kurze Chunks zusammen:
    - Wenn der Text eines Chunks sehr kurz ist (< short_threshold),
      wird er bevorzugt an den nächsten Chunk (gleiche Seite & section)
      vorn angehängt; falls nicht möglich, an den vorherigen.
      Wenn beides nicht geht, bleibt der Chunk unverändert.
    """
    new_chunks: List[Dict] = []
    i = 0
    n = len(chunks)

    while i < n:
        ch = chunks[i]
        text = ch.get("text", "")
        if len(text) < short_threshold:
            # 尝试合并到“下一个” / zuerst mit dem nächsten Chunk zusammenschieben
            merged = False
            if i + 1 < n:
                nxt = chunks[i + 1]
                if (nxt.get("page") == ch.get("page")
                        and nxt.get("section") == ch.get("section")):
                    # 把当前短文本加到下一条前面
                    # aktuellen kurzen Text vor den nächsten Chunk setzen
                    nxt["text"] = text + " " + nxt.get("text", "")
                    merged = True
                    # 当前 ch 不写入 new_chunks，直接跳过 / ch wird übersprungen

            if not merged and new_chunks:
                # 合并到“上一条” / an den vorherigen Chunk anhängen
                prev = new_chunks[-1]
                if (prev.get("page") == ch.get("page")
                        and prev.get("section") == ch.get("section")):
                    prev["text"] = prev.get("text", "") + " " + text
                    merged = True

            if not merged:
                # 实在找不到合并对象，就原样保留 / falls kein Merge möglich ist, Chunk behalten
                new_chunks.append(ch)

            i += 1
        else:
            # 正常长度的 chunk 直接保留 / normale Chunks einfach übernehmen
            new_chunks.append(ch)
            i += 1

    return new_chunks

# 这个函数会把特别短的 chunk 尝试和前后相邻的（同页、同 section）拼在一起，
# 避免出现只有 “- 2.”、“(5) Ende des 4.” 这类信息量过小的块。
# Diese Funktion versucht, extrem kurze Chunks mit ihren Nachbarn
# (gleiche Seite & section) zusammenzuführen, damit keine Mini-Chunks
# wie „- 2.“ oder „(5) Ende des 4.“ alleine stehen bleiben.


def split_long_chunk(chunk: Dict,
                     long_threshold: int = LONG_THRESHOLD) -> List[Dict]:
    """
    把一个过长的 chunk 按长度拆成多个小块：
    - 以 long_threshold 为上限，尽量在空格处切分；
    - 每一段保留原 chunk 的元信息，只是 text 缩短，id 后面加 _1, _2, _3...

    Teilt einen zu langen Chunk in mehrere kleinere:
    - nutzt long_threshold als maximale Länge und splittet möglichst an Leerzeichen;
    - Metadaten bleiben erhalten, nur der Text wird aufgeteilt,
      die id bekommt Suffixe _1, _2, _3...
    """
    text = chunk.get("text", "").strip()
    if len(text) <= long_threshold:
        return [chunk]

    parts: List[Dict] = []
    base_id = chunk.get("id", "chunk")

    start = 0
    part_idx = 1
    L = len(text)

    while start < L:
        # 先取一个最大窗口 / zunächst ein Fenster in Maximalgröße
        end = min(start + long_threshold, L)

        # 尝试在窗口中间到 end 范围内找最后一个空格作为切分点
        # Versuche, zwischen Mitte des Fensters und end das letzte Leerzeichen zu finden
        mid = start + long_threshold // 2
        split_pos = text.rfind(" ", mid, end)

        if split_pos == -1 or split_pos <= start:
            # 找不到合适空格，就硬切 / wenn kein Leerzeichen gefunden, hart splitten
            split_pos = end

        part_text = text[start:split_pos].strip()
        if part_text:
            new_chunk = dict(chunk)  # 浅拷贝 / Shallow Copy
            new_chunk["id"] = f"{base_id}_{part_idx}"
            new_chunk["text"] = part_text
            parts.append(new_chunk)
            part_idx += 1

        start = split_pos

    return parts

# 这个函数专门处理特别长的 chunk，会按大约 long_threshold 的长度切块，
# 优先在空格处分段，切出来的每一段都继承原来的元信息（page/section 等）。
# Diese Funktion behandelt sehr lange Chunks, indem sie den Text in Stücke
# von ungefähr long_threshold Zeichen aufteilt, möglichst an Leerzeichen.
# Jede Teilstück behält die ursprünglichen Metadaten wie page/section.


def split_long_chunks(chunks: List[Dict],
                      long_threshold: int = LONG_THRESHOLD) -> List[Dict]:
    """
    对列表里的每一个 chunk 进行“如果太长就拆分”的处理。

    Wendet die Split-Logik auf alle Chunks an:
    - wenn ein Chunk zu lang ist, wird er in mehrere Teile zerlegt.
    """
    new_chunks: List[Dict] = []
    for ch in chunks:
        text = ch.get("text", "")
        if len(text) > long_threshold:
            parts = split_long_chunk(ch, long_threshold=long_threshold)
            new_chunks.extend(parts)
        else:
            new_chunks.append(ch)
    return new_chunks

# 这个函数会遍历所有 chunk，对长度超过 long_threshold 的调用 split_long_chunk 做拆分。
# Diese Funktion iteriert über alle Chunks und ruft für zu lange Chunks
# split_long_chunk auf, um sie aufzuteilen.


def main():
    # 1. 读取原始 chunk 文件 / Original-Chunk-Datei laden
    chunks = load_chunks(INPUT_PATH)
    orig_count = len(chunks)
    print(f"📥 Geladene Chunks (Original): {orig_count}")

    # 2. 合并过短的 chunks / zu kurze Chunks zusammenführen
    merged = merge_short_chunks(chunks, short_threshold=SHORT_THRESHOLD)
    merged_count = len(merged)
    print(f"🩹 Nach Merge der kurzen Chunks: {merged_count} Chunks")

    # 3. 拆分过长的 chunks / zu lange Chunks aufteilen
    better = split_long_chunks(merged, long_threshold=LONG_THRESHOLD)
    final_count = len(better)
    print(f"✂️  Nach Split der langen Chunks: {final_count} Chunks")

    # 4. 写出到新的 JSONL 文件 / in neue JSONL-Datei schreiben
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in better:
            f.write(json.dumps(ch, ensure_ascii=False))
            f.write("\n")

    print("\n✅ Fertig!")
    print(f"   Ausgabedatei: {out_path}")
    print(f"\n📊 统计 / Statistik:")
    print(f"   原始 Chunk 数量 (Original): {orig_count}")
    print(f"   合并短 Chunk 之后 (nach Merge): {merged_count}")
    print(f"   最终输出 Chunk 数量 (final): {final_count}")

    # 5. 随便看几条结果 / ein paar Beispiel-Chunks anzeigen
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
