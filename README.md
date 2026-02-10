Inhaltsverzeichnis

Hinweis: Falls das Inhaltsverzeichnis beim Öffnen nicht automatisch aktualisiert wird, bitte in Word 'F9' drücken oder über 'Referenzen → Inhaltsverzeichnis aktualisieren'.
Manuelle Übersicht (wird durch das automatische Inhaltsverzeichnis ersetzt):
•	1. Einleitung und Systemarchitektur
•	2. Entwicklungsprozess: Backend und Datenpipeline
•	3. Frontend-Implementierung (index.html)
•	4. Qualitätssicherung und Test-Evaluation (test.py)
•	5. Nutzung und Reproduzierbarkeit
•	6. Fazit und Ausblick
 
1. Einleitung und Systemarchitektur
1.1 Zielsetzung
Ziel dieses Projekts ist die Entwicklung eines zuverlässigen, KI-gestützten Assistenten, der komplexe Fragen zur Studien- und Prüfungsordnung (SPO) der Hochschule Furtwangen präzise und nachvollziehbar beantwortet. Aufgrund der juristischen Formulierungen und der damit verbundenen Anforderungen an Faktentreue wurde ein Retrieval-Augmented-Generation (RAG)-Ansatz umgesetzt. Dieser reduziert Halluzinationen, indem Antworten konsequent auf retrievte Textpassagen aus den offiziellen Dokumenten gestützt werden.
1.2 Anforderungen
•	Nachvollziehbarkeit: Ausgabe von Referenzen (Dokument/Seite) zur schnellen Überprüfung.
•	Robustheit: Stabile Retrieval-Qualität durch Textbereinigung und konsistente Chunk-Metadaten.
•	Modularität: Trennung der Wissensbasen, um fachübergreifende und fachspezifische Inhalte sauber zu isolieren.
•	Reproduzierbarkeit: Klare Pipeline-Schritte (Cleaning → Chunking → Indexing) und dokumentierte Start-/Build-Kommandos.
1.3 Systemüberblick
Das System verwendet eine Dual-Engine-Architektur, um zwischen dem Allgemeinen Teil der SPO (AG) und fachspezifischen Regelungen (z. B. Mechatronik – MM) zu unterscheiden. Die Wissensbasen werden getrennt indiziert und zur Laufzeit dynamisch ausgewählt.
•	Wissensbasis: Zwei separate FAISS-Vektordatenbanken (AG und MM).
•	Routing: Engine-Auswahl durch das Frontend über den Request-Parameter target.
•	LLM-Synthese: OpenAI gpt-4o-mini zur sprachlichen Zusammenfassung der gefundenen Evidenzstellen.
•	Transparenz: Anzeige einer Referenz (Dokument/Seite) im Frontend auf Basis der Top-Hits.
1.4 Architektur und Datenfluss
Der Datenfluss kann in vier Stufen beschrieben werden:
1.	Offline-Pipeline: PDF-Extraktion und Bereinigung; Erzeugung von Chunk-Dateien (JSONL).
2.	Index-Build: Embedding-Berechnung (SentenceTransformer) und Aufbau eines FAISS-Index.
3.	Online-Retrieval: Anfrage-Embedding und Top-k Suche im passenden Index (AG/MM).
4.	Antwortgenerierung: Übergabe der Top-k Passagen an das LLM; Ausgabe von Antwort + Referenzen.
1.5 Technologie-Stack
•	Backend: Python, FastAPI, Uvicorn, FAISS, sentence-transformers, python-dotenv, OpenAI SDK
•	Frontend: HTML, JavaScript, Tailwind CSS, Prism.js (Code-Highlighting)
•	Persistenz/Artefakte: JSONL (Pages/Chunks/Mapping), NPY (Embeddings), FAISS Index-Datei
2. Entwicklungsprozess: Backend und Datenpipeline
2.1 Heuristische Textreinigung (ag wash.py)
PDF-Dokumente enthalten häufig störende Elemente wie Kopf-/Fußzeilen, Seitenzählungen und unsichtbare Sonderzeichen. Diese Artefakte verschlechtern Embeddings und führen zu schwächerem Retrieval. Daher wurde eine Bereinigungslogik implementiert, die insbesondere Worttrennungen am Zeilenende korrigiert und typische Footer-Patterns entfernt.
Listing 1: Normalisierung und Bereinigung von PDF-Extraktionen (Auszug).
def normalize_whitespace(s: str) -> str:
    # Entfernung von Sonderzeichen und Soft-Hyphens
    s = s.replace("\xa0", " ").replace("\u00ad", "")
    # Reparatur von Worttrennungen: Studien-\nordnung -> Studienordnung
    s = re.sub(r"-\s*\n\s*", "", s)
    # Konsolidierung von Whitespaces
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()
Ergänzend werden Kopf-/Fußzeilenmuster entfernt und finale Normalisierungen durchgeführt (z. B. einheitliche Abstände nach Nummerierungen). In der Pipeline werden sowohl eine RAW-als auch eine CLEAN-JSONL-Datei erzeugt, um Debugging und Rückverfolgbarkeit zu unterstützen.
Listing 2: Entfernen typischer Header-/Footer-Muster und finale Bereinigung (Auszug).
def remove_headers_footers(s: str) -> str:

    s = re.sub(
        r"Inkrafttreten:\s*.*?\bAllg\.\s*B\s*SPO\s*\d+/\d+",
        "",
        s,
        flags=re.IGNORECASE
    )


    s = re.sub(r"\bAllg\.\s*B\s*SPO\s*\d+/\d+\b", "", s, flags=re.IGNORECASE)

    s = re.sub(r"\bSeite\s+\d+(\s+von\s+\d+)?\b", "", s, flags=re.IGNORECASE)

    return s.strip()

def clean_text(s: str) -> str:
    s = normalize_whitespace(s)
    s = remove_headers_footers(s)

    s = re.sub(r"(\d+\.\d+)\.(?=[A-ZÄÖÜ])", r"\1. ", s)
    
    s = re.sub(r"\s*(§\s*\d+)", r" \1", s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()
2.2 Strukturerhaltendes Chunking (AG chunk.py)
Um juristische Paragraphen und Aufzählungen im Kontext zu halten, wird ein satzbasiertes Chunking-Verfahren verwendet. Nummerierungen (z. B. “(1)”) werden mittels RegEx als eigene Satzanfänge separiert, damit sie nicht mit dem vorherigen Satz verschmelzen. Zusätzlich werden Metadaten wie Seite und Abschnitt gespeichert, um die spätere Quellenanzeige zu ermöglichen.
Listing 3: Vorverarbeitung und Satz-Splitting (Auszug).
def preprocess_text(text: str) -> str:
    # Erzwingt Zeilenumbrüche vor Absatznummerierungen zur sauberen Trennung
    text = re.sub(r'\s*(\(\d+\))', r'\n\1', text)
    return text

def split_into_sentences(text: str) -> List[str]:
    text = preprocess_text(text)
    rough_parts = re.split(r'\n+', text)
    sentences = []
    for part in rough_parts:
        sub = re.split(r'(?<=[\.\?\!])\s+', part)
        sentences.extend([s.strip() for s in sub if s.strip()])
    return sentences
Im Anschluss werden die Sätze in Chunks überführt. Die Chunk-IDs sind deterministisch und enthalten Dokumentkürzel, Seitenzahl und Chunk-Index. Dies erleichtert Debugging und Versionierung der Index-Artefakte.
Listing 4: Chunk-Erzeugung mit Metadaten (Auszug).
chunk_id = f"{doc_short_name}_p{page}_c{i:02d}"
chunk = {"id": chunk_id, "page": page, "section": section, "text": ctext}
2.3 Verfeinerung der Chunks (AG-besser chunk.py)
SPO-Texte enthalten teilweise sehr lange Sätze. Um das Kontextfenster effizient zu nutzen und die semantische Dichte zu kontrollieren, wird in einer zweiten Stufe ein Längenlimit eingeführt. Zu lange Einheiten werden bevorzugt an Wortgrenzen aufgeteilt, wobei die Metadaten vollständig übernommen werden.
Listing 5: Längenbasierte Segmentierung und Refinement (Auszug).
def split_long_sentence_by_length(sentence: str, max_chars: int = 350) -> List[str]:
    if len(sentence) <= max_chars:
        return [sentence.strip()]
    parts, start = [], 0
    while start < len(sentence):
        end = min(start + max_chars, len(sentence))
        split_pos = sentence.rfind(" ", start, end)
        if split_pos == -1 or split_pos <= start:
            split_pos = end
        parts.append(sentence[start:split_pos].strip())
        start = split_pos
    return [p for p in parts if p]
2.4 Vektor-Indizierung (build-index AG.py)
Die Chunks werden mit dem Modell paraphrase-multilingual-MiniLM-L12-v2 in Vektoren umgewandelt. Zur effizienten Ähnlichkeitssuche wird ein FAISS IndexFlatIP verwendet. Durch vorherige Normalisierung der Embeddings entspricht der Inner Product der Cosine Similarity.
Listing 6: Aufbau und Persistierung des FAISS-Index (Auszug).
def build_faiss_index(embeddings: np.ndarray, path: str) -> None:
    num_vectors, dim = embeddings.shape
    # Inner Product auf normalisierten Vektoren entspricht Cosine Similarity
    index = faiss.IndexFlatIP(dim) 
    index.add(embeddings)
    faiss.write_index(index, str(Path(path)))
Für Nachvollziehbarkeit wird zusätzlich eine Mapping-Datei (Indexposition → Chunk-Metadaten) geschrieben. Dies erleichtert die Diagnose von Retrieval-Ergebnissen und stellt sicher, dass Index und Chunk-Datei konsistent zusammengehören.
Listing 7: Speicherung einer Index-zu-Chunk Mapping-Datei (Auszug).
def save_mapping(chunks: List[Dict], path: str) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for idx, ch in enumerate(chunks):
            rec = {"idx": idx, "id": ch.get("id"), "page": ch.get("page"),
                   "section": ch.get("section"), "doc": ch.get("doc")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
2.5 API-Orchestrierung (main_api.py)
Das Backend basiert auf FastAPI und übernimmt die Orchestrierung der Pipeline zur Laufzeit: Routing auf die korrekte Wissensbasis, Retrieval (Top-k), Kontextaufbau sowie die Übergabe an das LLM. Der Parameter target wird vom Frontend gesetzt und steuert die Indexauswahl.
Listing 8: Dual-Engine Routing und Retrieval (Auszug).
@app.post("/chat")
async def chat(request: ChatRequest):
    # Auswahl der Wissensbasis
    if request.target == "AG":
        current_index, current_chunks = ag_index, ag_chunks
    else:
        current_index, current_chunks = mm_index, mm_chunks

    # Vektor-Retrieval
    query_vector = model.encode([request.query], normalize_embeddings=True)
    D, I = current_index.search(query_vector, k=5)
    # ... LLM Generierung folgt ...
Die Antwort wird zusammen mit einer Hit-Liste zurückgegeben. Jeder Hit enthält Metadaten (Dokument, Seite, Score), die im Frontend als Referenz angezeigt werden können. Die Wahl von k=5 ist ein pragmatischer Kompromiss aus Kontextabdeckung und Tokenbudget.
Listing 9: Kontextaufbau und OpenAI-Call (Auszug).
context = "\n\n".join(context_parts)
prompt = f"Du bist ein Studienassistent. Antworte auf {request.lang}.\nContext:\n{context}\nFrage: {request.query}"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "Assistant for HFU students."},
              {"role": "user", "content": prompt}]
)
3. Frontend-Implementierung (index.html)
Das Frontend wurde mit Tailwind CSS umgesetzt und legt den Fokus auf Bedienbarkeit und Transparenz. Zentrale UI-Elemente sind die Modusumschaltung (AG/MM), ein klarer Chat-Verlauf sowie die Anzeige einer Referenzzeile (Dokumentname und Seitenzahl) aus den Retrieval-Hits.
Listing 10: API-Request aus dem Frontend (Auszug).
async function sendQuery() {
    const res = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ 
            query: question, 
            target: currentMode // Wechsel zwischen AG und MM
        })
    });
    const data = await res.json();
    appendMsg(data.answer, 'ai', data.hits); // Anzeige inkl. Metadaten
}
Die Referenzausgabe ist bewusst kompakt gehalten, um die Lesbarkeit der Antwort nicht zu beeinträchtigen, gleichzeitig aber eine unmittelbare Plausibilitätsprüfung zu ermöglichen.
Listing 11: Referenzanzeige im Chat (Auszug).
if (hits.length > 0) {
    bubble.innerHTML += `<div class="mt-4 pt-3 border-t text-[11px] font-bold opacity-60 uppercase">Referenz: ${hits[0].doc} · S.${hits[0].page}</div>`;
}
4. Qualitätssicherung und Test-Evaluation (test.py)
Zur Absicherung der Systemqualität wurde ein automatisierbares Evaluationsskript vorbereitet. Es baut aus einer Fragenliste (questions.txt) einen Datensatz aus Nutzerfrage, Systemantwort und tatsächlich genutzten Retrieval-Kontexten. Anschließend wird mit RAGAS eine metrische Bewertung ohne Ground-Truth durchgeführt.
4.1 Bewertete Kriterien
•	Faithfulness (Faktentreue): Misst, ob die Antwort durch den gelieferten Kontext inhaltlich gedeckt ist.
•	Answer Relevancy (Antwort-Relevanz): Bewertet, ob die Antwort die Nutzerfrage direkt adressiert.
•	Context Precision (Kontext-Präzision): Bewertet, wie viel des zurückgegebenen Kontexts tatsächlich relevant ist (wenig Rauschen).
•	Context Recall (Kontext-Abdeckung): Misst, ob die relevanten Textstellen im Retrieval enthalten sind (möglichst vollständige Abdeckung).
Anmerkung: Die Aussagekraft einzelner Metriken hängt von der Verfügbarkeit geeigneter Testfragen und ggf. Referenzkontexte ab. In der Praxis empfiehlt sich eine Kombination aus metrischer Evaluation und stichprobenartiger manueller Prüfung der Quellenstellen.
4.2 Evaluationsskript (Auszug)
Listing 12: RAGAS-Evaluation und Radar-Plot (Auszug).
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=eval_llm,
    embeddings=eval_embeddings
)
# Visualisierung der Scores als Radar-Plot für die Performance-Analyse
Hinweis: In der aktuellen Projektfassung dient test.py als Evaluations-Framework. Für den produktiven Testlauf muss die Funktion call_your_rag(frage) so implementiert werden, dass sie die Systemantwort und exakt die Passagen zurückliefert, die an das LLM übergeben wurden. Damit wird sichergestellt, dass die Metriken tatsächlich das reale Systemverhalten messen.
5. Nutzung und Reproduzierbarkeit
5.1 Daten-Aktualisierung
Die Index-Artefakte werden offline erzeugt. Bei Änderungen an den SPO-Dokumenten sollten die folgenden Schritte in der angegebenen Reihenfolge erneut ausgeführt werden:
5.	python ag wash.py
6.	python AG chunk.py
7.	python AG-besser chunk.py
8.	python build-index AG.py
Die Skripte verwenden in der vorliegenden Version Windows-spezifische Pfade (z. B. D:\KI-Agent). Für andere Umgebungen sind die Pfadvariablen in den jeweiligen Dateien anzupassen.
5.2 Systemstart
Backend starten: python main_api.py (Standard: 127.0.0.1:8000). Anschließend index.html im Browser öffnen. Im UI kann zwischen Allgemein (AG) und Mechatronik (MM) umgeschaltet und eine Anfrage gestellt werden.
5.3 Konfiguration
Für die LLM-Komponente ist ein OpenAI API-Key erforderlich. Dieser wird über eine .env-Datei bereitgestellt und beim Start des Backends per load_dotenv() geladen. Empfohlen ist eine minimale Konfiguration mit OPENAI_API_KEY und optionalen Proxy-/Timeout-Einstellungen gemäß lokaler IT-Richtlinien.
6. Fazit und Ausblick
Das entwickelte System demonstriert eine funktionsfähige RAG-Architektur für die SPO-Beratung. Die klare Trennung der Wissensbasen (AG/MM) reduziert thematische Vermischungen und unterstützt präzises Retrieval. Durch die Referenzanzeige im Frontend wird die Überprüfbarkeit der Antworten verbessert.
6.1 Grenzen des Prototyps
•	Der Assistent ist ein Informationssystem und ersetzt keine verbindliche Rechtsauskunft; maßgeblich bleibt die Original-SPO.
•	Die Antwortqualität hängt von der Aktualität der indizierten Dokumente sowie von Chunking- und Retrieval-Parametern ab.
•	Bei sehr spezifischen Begriffen kann eine rein vektorbasierte Suche einzelne Treffer übersehen (Synonyme/Schreibvarianten).
6.2 Weiterentwicklung
•	Hybrid Search (Vektor + Keyword/BM25), um exakte Fachbegriffe und Paragraphenzitate noch zuverlässiger zu treffen.
•	Re-Ranking (z. B. Cross-Encoder) nach FAISS Top-k, um die Präzision der Kontextauswahl zu erhöhen.
•	Erweiterte Quellenanzeige (Top-3 Passagen mit Score) und optionaler Link auf die PDF-Seite zur direkten Verifikation.
•	Automatisierte Regressionstests: feste Fragebatterie + RAGAS-Auswertung pro Index-Version.
Dokumentation erstellt für die Hochschule Furtwangen (HFU).
