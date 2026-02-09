from pathlib import Path
import re, json
from datetime import datetime
from typing import Dict, Any, List


# ====== 路径配置（按你的 Windows 项目路径）======
BASE = Path(r"D:\KI-Agent")

PDF_PATH = BASE / "data" / "spo_allgemein" / "AllgemeinerTeilBachelor_aktuell_13.12.2023.pdf"

# ✅ 输出也放回 data/spo_allgemein
OUT_DIR = BASE / "data" / "spo_allgemein"
OUT_RAW = OUT_DIR / "allgemeinerteil_bachelor_raw.jsonl"
OUT_CLEAN = OUT_DIR / "allgemeinerteil_bachelor_clean.jsonl"


def read_pdf_text(pdf_path: Path) -> List[Dict[str, Any]]:
    """优先 pypdf；失败回退 pdfplumber。按页返回 [{"page":1,"text":"..."}, ...]"""
    pages: List[Dict[str, Any]] = []

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            pages.append({"page": i, "text": txt})
        return pages

    except Exception:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                pages.append({"page": i, "text": txt})
        return pages


def normalize_whitespace(s: str) -> str:
    # 清掉 PDF 抽取常见脏字符（如“￾”导致的乱码）
    s = (
        s.replace("\xa0", " ")
         .replace("\u00ad", "")   # soft hyphen
         .replace("\ufffe", "")   # noncharacter（常见为“￾”）
         .replace("\uffff", "")
         .replace("\ufffd", "")   # replacement char
    )

    # 断词修复：Studien-\nordnung -> Studienordnung
    s = re.sub(r"-\s*\n\s*", "", s)

    # 换行合并成空格
    s = re.sub(r"\s*\n\s*", " ", s)

    # 多空格合一
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()


def remove_headers_footers(s: str) -> str:
    # 这份 PDF 页眉/页脚典型：Inkrafttreten... Allg. B SPO 1/30
    s = re.sub(
        r"Inkrafttreten:\s*.*?\bAllg\.\s*B\s*SPO\s*\d+/\d+",
        "",
        s,
        flags=re.IGNORECASE
    )

    # 只剩 “Allg. B SPO 12/30” 这种也干掉
    s = re.sub(r"\bAllg\.\s*B\s*SPO\s*\d+/\d+\b", "", s, flags=re.IGNORECASE)

    # 传统页码（保险）
    s = re.sub(r"\bSeite\s+\d+(\s+von\s+\d+)?\b", "", s, flags=re.IGNORECASE)

    # 版本日期（有些页可能出现）
    s = re.sub(r"\bStand:\s*\d{1,2}\.\d{1,2}\.\d{2,4}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bLetzte\s+Änderung:\s*\d{1,2}\.\d{1,2}\.\d{2,4}\b", "", s, flags=re.IGNORECASE)

    # 版权尾巴（如果有）
    s = re.sub(r"©\s?\d{4}.*", "", s)

    return s.strip()


def clean_text(s: str) -> str:
    s = normalize_whitespace(s)
    s = remove_headers_footers(s)

    # 条款号后补空格：3.2.Prüfung -> 3.2. Prüfung
    s = re.sub(r"(\d+\.\d+)\.(?=[A-ZÄÖÜ])", r"\1. ", s)

    # 给“§”前面留个断点（方便检索/切块）
    s = re.sub(r"\s*(§\s*\d+)", r" \1", s)

    # 最终再收一下多空格
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF 不存在：{PDF_PATH}")

    pages = read_pdf_text(PDF_PATH)

    n_raw, n_clean = 0, 0
    with OUT_RAW.open("w", encoding="utf-8") as f_raw, OUT_CLEAN.open("w", encoding="utf-8") as f_clean:
        for p in pages:
            record_raw = {
                "doc": PDF_PATH.name,
                "typ": "Allgemeine",
                "page": p["page"],
                "text": p["text"],
                "source_path": str(PDF_PATH),
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            }
            f_raw.write(json.dumps(record_raw, ensure_ascii=False) + "\n")
            n_raw += 1

            text_clean = clean_text(p["text"] or "")
            record_clean = {
                "doc": PDF_PATH.name,
                "typ": "Allgemeine",
                "page": p["page"],
                "text": text_clean,
            }
            f_clean.write(json.dumps(record_clean, ensure_ascii=False) + "\n")
            n_clean += 1

    print(f"[OK] RAW pages:   {n_raw} -> {OUT_RAW}")
    print(f"[OK] CLEAN pages: {n_clean} -> {OUT_CLEAN}")


if __name__ == "__main__":
    main()
