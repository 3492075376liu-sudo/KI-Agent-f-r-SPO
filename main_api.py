from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

# 1. 初始化
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 路径配置
BASE_DIR = Path(r"D:\KI-Agent")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# MM 路径 (原来的)
MM_CHUNK_PATH = BASE_DIR / "out" / "besser Chunk.jsonl"
MM_INDEX_PATH = BASE_DIR / "out" / "MM_SPO13_faiss.index"

# AG 路径 (新加的)
AG_CHUNK_PATH = BASE_DIR / "out" / "AG-besser chunk.jsonl"
AG_INDEX_PATH = BASE_DIR / "out" / "AG_besser_faiss.index"

# 3. 资源预加载
print("🚀 Doppelmotorige Daten werden gerade geladen. (MM + AG)...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer(MODEL_NAME)

# 加载 MM
with open(MM_CHUNK_PATH, "r", encoding="utf-8") as f:
    mm_chunks = [json.loads(line) for line in f]
mm_index = faiss.read_index(str(MM_INDEX_PATH))

# 加载 AG
with open(AG_CHUNK_PATH, "r", encoding="utf-8") as f:
    ag_chunks = [json.loads(line) for line in f]
ag_index = faiss.read_index(str(AG_INDEX_PATH))

print("✅ Das Dual-Boot-System ist erfolgreich gestartet.")

# 定义请求格式，增加一个 target 参数
class ChatRequest(BaseModel):
    query: str
    lang: str = "de"
    target: str = "MM"  # 默认为 Mechatronik

@app.post("/chat")
async def chat(request: ChatRequest):
    # 根据前端选择切换数据源
    if request.target == "AG":
        current_index = ag_index
        current_chunks = ag_chunks
        doc_label = "Allgemeine SPO"
    else:
        current_index = mm_index
        current_chunks = mm_chunks
        doc_label = "Mechatronik SPO"

    # 1. 向量检索
    query_vector = model.encode([request.query], normalize_embeddings=True)
    D, I = current_index.search(query_vector, k=5)
    
    hits = []
    context_parts = []
    for rank, idx in enumerate(I[0]):
        if idx != -1:
            ch = current_chunks[idx]
            hits.append({
                "rank": rank + 1,
                "doc": ch.get("doc", doc_label),
                "page": ch.get("page"),
                "score": float(D[0][rank])
            })
            context_parts.append(f"[{doc_label} P.{ch.get('page')}]\n{ch.get('text')}")

    # 2. 大模型生成
    context = "\n\n".join(context_parts)
    prompt = f"Du bist ein Studienassistent. Antworte auf {request.lang}.\nContext:\n{context}\nFrage: {request.query}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Assistant for HFU students."},
                  {"role": "user", "content": prompt}]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "hits": hits
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)