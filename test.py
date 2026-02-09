import os
from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset

# ========= Konfiguration (bitte anpassen) =========
FRAGEN_DATEI = r"questions.txt"          # 一行一个问题
AUSGABE_ORDNER = r"./ragas_out"
MAX_FRAGEN = None                       # 例如 50；不限制就 None

# Bewertungs-Modelle (OpenAI als Beispiel)
EVAL_LLM_MODEL = "gpt-4o-mini"
EVAL_EMBED_MODEL = "text-embedding-3-small"


# ========= HIER musst du dein eigenes RAG-System anschließen =========
def call_your_rag(frage: str) -> Tuple[str, List[str]]:
    """
    Rückgabe:
      antwort: str
      kontexte: List[str]  # genau die Chunks/Passagen, die dein Generator wirklich bekommen hat
    """
    # TODO: Ersetze das durch deinen Code (Retriever -> Prompt -> Generator)
    # Beispiel (Pseudo):
    # antwort, docs = rag.ask(frage)
    # kontexte = [d.text for d in docs]
    raise NotImplementedError("Bitte hier dein RAG-System anschließen und (antwort, kontexte) zurückgeben.")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def radar_plot(scores: Dict[str, float], titel: str, pfad: str):
    labels = list(scores.keys())
    values = [float(scores[k]) for k in labels]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.array(values)
    angles = np.concatenate([angles, angles[:1]])
    values = np.concatenate([values, values[:1]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.20)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 1)
    ax.set_title(titel, pad=20)
    fig.savefig(pfad, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(AUSGABE_ORDNER)

    # --- Fragen laden ---
    with open(FRAGEN_DATEI, "r", encoding="utf-8") as f:
        fragen = [line.strip() for line in f if line.strip()]
    if MAX_FRAGEN:
        fragen = fragen[:MAX_FRAGEN]
    if not fragen:
        raise ValueError("Die Datei questions.txt ist leer. Bitte eine Frage pro Zeile eintragen.")

    print(f"[Info] Anzahl Fragen: {len(fragen)}")

    # --- RAG laufen lassen und Datensatz bauen ---
    rows = []
    for i, frage in enumerate(fragen, start=1):
        antwort, kontexte = call_your_rag(frage)
        if not isinstance(kontexte, list) or not all(isinstance(x, str) for x in kontexte):
            raise TypeError("retrieved_contexts muss eine List[str] sein.")

        rows.append(
            {
                "user_input": frage,
                "response": antwort,
                "retrieved_contexts": kontexte,
            }
        )

        if i % 10 == 0:
            print(f"[Info] {i}/{len(fragen)} Fragen verarbeitet...")

    df = pd.DataFrame(rows)
    jsonl_path = os.path.join(AUSGABE_ORDNER, "rag_outputs.jsonl")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"[OK] Rohdaten gespeichert: {jsonl_path}")

    dataset = Dataset.from_pandas(df)

    # --- RAGAS Setup (OpenAI as Judge) ---
    # Voraussetzung: OPENAI_API_KEY ist gesetzt
    from openai import AsyncOpenAI, OpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    from ragas import evaluate

    llm = llm_factory(EVAL_LLM_MODEL, client=AsyncOpenAI())
    embeddings = embedding_factory(
        provider="openai",
        model=EVAL_EMBED_MODEL,
        client=OpenAI(),
        interface="modern",
    )

    from ragas.metrics.collections import (
        AnswerRelevancy,
        Faithfulness,
        ContextRelevance,
        ResponseGroundedness,
    )

    metrics = [
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        Faithfulness(llm=llm),
        ContextRelevance(llm=llm),
        ResponseGroundedness(llm=llm),
    ]

    print("[Info] Starte RAGAS-Bewertung (vier Metriken, ohne Ground Truth)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
    )

    result_dict = dict(result)
    print("\n[Ergebnis] Gesamt-Scores:")
    for k, v in result_dict.items():
        print(f" - {k}: {v}")

    # --- hübsche Namen für Radar ---
    # (Keys können je nach Version minimal variieren; wir mappen defensiv)
    def pick_key(candidates):
        for c in candidates:
            if c in result_dict:
                return c
        raise KeyError(f"Keiner der Keys gefunden: {candidates}. Verfügbar: {list(result_dict.keys())}")

    scores = {
        "Antwort-Relevanz": float(result_dict[pick_key(["answer_relevancy"])]),
        "Faktentreue": float(result_dict[pick_key(["faithfulness"])]),
        "Kontext-Relevanz": float(result_dict[pick_key(["context_relevance"])]),
        "Antwort-Begründetheit": float(result_dict[pick_key(["response_groundedness"])]),
    }

    csv_path = os.path.join(AUSGABE_ORDNER, "ragas_scores.csv")
    pd.DataFrame([scores]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Scores gespeichert: {csv_path}")

    radar_path = os.path.join(AUSGABE_ORDNER, "ragas_radar.png")
    radar_plot(scores, titel="RAGAS Radar (4 Kernmetriken, ohne Ground Truth)", pfad=radar_path)
    print(f"[OK] Radar-Plot gespeichert: {radar_path}")

    print("\n[Fertig] Bewertung abgeschlossen ✅")


if __name__ == "__main__":
    main()
