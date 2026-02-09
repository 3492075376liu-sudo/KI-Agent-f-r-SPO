from pathlib import Path
import json
import re
from typing import List

# ===== 路径配置 / Pfad-Konfiguration =====
INPUT_PATH = r"D:\KI-Agent\out\Maschinenbau_und_Mechatronik_clean.json.txt"
OUTPUT_PATH = r"D:\KI-Agent\out\Maschinenbau_und_Mechatronik_chunks.jsonl"
DOC_SHORT_NAME = "MM_SPO13"  # 文档短名 / Kurzname des Dokuments

# 上面三行是你需要改的配置：输入清洗后文件路径、输出 chunks 文件路径、文档简称。
# Die drei Zeilen oben sind die wichtigsten Einstellungen:
# Pfad zur bereinigten Eingabedatei, Pfad zur Ausgabedatei und Dokument-Kurzname.


def load_pages(path: str):
    """
    读取清洗后的 JSONL 文件。
    一行一页，返回一个 dict 列表，每个 dict 代表一页。
    需要文件里每行至少有: page, text 这两个字段。

    Lädt die bereinigte JSONL-Datei.
    Eine Zeile entspricht einer Seite als Dict.
    Jede Zeile braucht mindestens die Felder: page, text.
    """
    pages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pages.append(json.loads(line))
    return pages

# 这个函数负责把 JSONL 读进来，每一行（一页）变成一个 Python 字典。
# Diese Funktion liest die JSONL-Datei ein und wandelt jede Zeile (eine Seite)
# in ein Python-Dict um.


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
# Ziel dieser Funktion ist es, Nummerierungen (1)(2)(3)... vom vorherigen Satz zu trennen,
# damit sie sauber als eigener Satzanfang behandelt werden.


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

# 这里实现了一个简单的“按句子切分”的逻辑：先按换行粗分，再按 . ? ! 分成短句。
# Hier wird ein einfaches Satz-Splitting umgesetzt: zuerst grob per Zeilenumbruch,
# dann feiner per . ? ! in einzelne Sätze.


def make_sentence_chunks(
    sentences: List[str],
    max_sent_per_chunk: int = 1,
    overlap: int = 0
) -> List[str]:
    """
    按句子生成 chunk。
    默认：max_sent_per_chunk=1，一句一个 chunk，不重叠。

    Erzeugt Chunks aus der Satzliste.
    Standard: max_sent_per_chunk=1, ein Satz pro Chunk, ohne Überlappung.
    """
    chunks = []
    if not sentences:
        return chunks

    step = max_sent_per_chunk - overlap
    if step <= 0:
        raise ValueError("max_sent_per_chunk 必须大于 overlap")

    start = 0
    while start < len(sentences):
        end = start + max_sent_per_chunk
        chunk_sents = sentences[start:end]
        if len(chunk_sents) == 0:
            break
        chunk_text = " ".join(chunk_sents)
        chunks.append(chunk_text)
        start += step

    return chunks

# 这个函数把句子数组重新打包成 chunk，现在设置为“一句就是一个 chunk”。
# Diese Funktion packt die Satzliste in Chunks. Momentan wird jeder einzelne Satz
# direkt zu einem Chunk gemacht.


def guess_section(full_text: str, page: int) -> str:
    """
    粗略猜测这一页属于哪个 section：
    - 包含 § 41 / Tabelle 1 / Tabelle 2 / Tabelle 3 就返回对应标题；
    - 否则返回 'Seite X'。

    Grobe Schätzung, zu welchem Abschnitt die Seite gehört:
    - Wenn der Text § 41 / Tabelle 1 / Tabelle 2 / Tabelle 3 enthält,
      wird der entsprechende Titel genutzt;
    - sonst 'Seite X'.
    """
    if "§ 41" in full_text:
        return "§ 41 Bachelorstudiengang Maschinenbau und Mechatronik"
    if "Tabelle 1" in full_text:
        return "Tabelle 1 Modulstruktur"
    if "Tabelle 2" in full_text:
        return "Tabelle 2 Grundstudium"
    if "Tabelle 3" in full_text:
        return "Tabelle 3 Hauptstudium"
    return f"Seite {page}"

# 这个函数只是给每一页打个大概的标签：属于哪个条款/表格，或者直接“第 X 页”。
# Diese Funktion vergibt einen groben Abschnitts-Namen für jede Seite:
# entweder ein Paragraf/Tabelle oder einfach 'Seite X'.


def make_chunks_for_page(
    page_rec: dict,
    doc_short_name: str,
    max_sent_per_chunk: int = 1,
    overlap: int = 0
):
    """
    把“一页”的记录切成多个 chunk dict（默认一句一个 chunk）。

    Teilt einen Seiten-Datensatz in mehrere Chunk-Dicts
    (Standard: ein Satz pro Chunk).
    """
    page = page_rec.get("page")
    full_text = page_rec.get("text", "")

    sentences = split_into_sentences(full_text)
    chunk_texts = make_sentence_chunks(
        sentences,
        max_sent_per_chunk=max_sent_per_chunk,
        overlap=overlap
    )

    section = guess_section(full_text, page)

    chunks = []
    for i, ctext in enumerate(chunk_texts, start=1):
        chunk_id = f"{doc_short_name}_p{page}_c{i:02d}"
        chunk = {
            "id": chunk_id,
            "doc": doc_short_name,
            "page": page,
            "type": "paragraph",  # 目前统一标记为 paragraph
            "section": section,
            "text": ctext
        }
        chunks.append(chunk)

    return chunks

# 这个函数把“某一页”的 text 拆成很多条 chunk，每条 chunk 变成一个带字段的字典。
# Diese Funktion zerlegt den Text einer Seite in viele Chunks und baut
# für jeden Chunk ein Dict mit allen wichtigen Feldern.


def main():
    # 1. 读取所有页面 / alle Seiten laden
    pages = load_pages(INPUT_PATH)
    if not pages:
        print("❌ Es wurden keine Seiten geladen. Bitte prüfe den Pfad in INPUT_PATH.")
        return

    all_chunks = []

    # 2. 处理整个文档的每一页 / jede Seite des Dokuments verarbeiten
    for page_rec in pages:
        page = page_rec.get("page")
        print(f"🔧 Verarbeite Seite {page} ...")
        page_chunks = make_chunks_for_page(
            page_rec,
            doc_short_name=DOC_SHORT_NAME,
            max_sent_per_chunk=1,  # 一句一个 chunk / ein Satz pro Chunk
            overlap=0
        )
        all_chunks.extend(page_chunks)

    # 3. 写出所有 chunks 到 JSONL / alle Chunks in eine JSONL-Datei schreiben
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False))
            f.write("\n")

    print(f"\n✅ Insgesamt wurden {len(all_chunks)} Chunks erzeugt.")
    print(f"   Ausgabedatei: {out_path}")

    # 4. 控制台上顺便看几条 / ein paar Beispiel-Chunks in der Konsole anzeigen
    print("\n👀 Beispiel-Chunks:\n")
    for ch in all_chunks[:10]:
        print("----")
        print("id:      ", ch["id"])
        print("Seite:   ", ch["page"])
        print("Abschnitt:", ch["section"])
        print("Text:    ", ch["text"])
        print()

# main() 负责把所有步骤串起来：读文件 → 每页切 chunk → 写出 JSONL → 打印前几个示例。
# Die main()-Funktion verbindet alle Schritte:
# Datei laden → pro Seite Chunks erzeugen → JSONL schreiben → einige Beispiele ausgeben.


if __name__ == "__main__":
    main()
