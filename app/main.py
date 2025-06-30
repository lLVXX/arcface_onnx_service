# app/main.py
import os
import json
import numpy as np
import psycopg2
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from io import BytesIO
from PIL import Image

# ----------------------------
# Configuración desde ENV
# ----------------------------
PG_CONFIG = {
    'dbname':   os.getenv('PG_DB',       'SCOUT_DB'),
    'user':     os.getenv('PG_USER',     'postgres'),
    'password': os.getenv('PG_PASSWORD', '12345678'),
    'host':     os.getenv('PG_HOST',     'db'),
    'port':     int(os.getenv('PG_PORT', '5432')),
}
THRESHOLD = float(os.getenv('THRESHOLD', '0.5'))

# ----------------------------
# FastAPI & CORS
# ----------------------------
app = FastAPI(title="ArcFace ONNX Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"],  allow_headers=["*"],
)

# ----------------------------
# Modelo ArcFace
# ----------------------------
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# Embeddings cache
# ----------------------------
embeddings_dict: dict[int, list[np.ndarray]] = {}

def load_embeddings() -> dict[int, list[np.ndarray]]:
    """Lee todos los embeddings guardados en Postgres."""
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT estudiante_id, embedding FROM personas_estudiantefoto WHERE embedding IS NOT NULL"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    embs = defaultdict(list)
    for est_id, emb_val in rows:
        if isinstance(emb_val, (bytes, bytearray, memoryview)):
            raw = emb_val.tobytes() if isinstance(emb_val, memoryview) else emb_val
            emb = np.frombuffer(raw, dtype=np.float32)
        else:
            arr = json.loads(emb_val)
            emb = np.array(arr, dtype=np.float32)
        embs[est_id].append(emb)
    return embs

@app.on_event("startup")
def startup_event():
    global embeddings_dict
    try:
        embeddings_dict = load_embeddings()
        total = sum(len(v) for v in embeddings_dict.values())
        print(f"✅ Cargados {total} embeddings desde la tabla personas_estudiantefoto")
    except Exception as e:
        embeddings_dict = {}
        print(f"⚠️ No se pudieron cargar embeddings: {e}")
    print(f"▶ Conectado a PostgreSQL {PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}")
    print(f"▶ Threshold = {THRESHOLD}")

# ----------------------------
# Endpoints HTTP
# ----------------------------
@app.post("/generar_embedding/")
async def generar_embedding(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo imagen: {e}")

    faces = face_app.get(np.array(img))
    if not faces:
        return {"ok": False, "msg": "No se detectaron rostros"}
    return {"ok": True, "embedding": faces[0].embedding.tolist()}

@app.post("/match_faces/")
async def match_faces(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo imagen: {e}")

    faces = face_app.get(np.array(img))
    if not faces:
        return {"ok": False, "msg": "No se detectaron rostros"}

    results = []
    for f in faces:
        q = f.embedding.astype(np.float32)
        best_id, best_score = None, -1.0
        for est_id, em_list in embeddings_dict.items():
            for emb in em_list:
                sim = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-8))
                if sim > best_score:
                    best_score, best_id = sim, est_id
        results.append({
            "face_box": f.bbox.tolist(),
            "estudiante_id": best_id if best_score > THRESHOLD else None,
            "similarity": best_score,
            "match": best_score > THRESHOLD
        })
    return {"ok": True, "results": results}

# ----------------------------
# WebSocket para streaming
# ----------------------------
@app.websocket("/stream/")
async def stream_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            img = Image.open(BytesIO(data)).convert("RGB")
            faces = face_app.get(np.array(img))
            out = []
            for f in faces:
                q = f.embedding.astype(np.float32)
                best_id, best_score = None, -1.0
                for est_id, em_list in embeddings_dict.items():
                    for emb in em_list:
                        sim = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-8))
                        if sim > best_score:
                            best_score, best_id = sim, est_id
                out.append({
                    "match_id": best_id if best_score > THRESHOLD else None,
                    "similarity": best_score
                })
            await ws.send_json({"faces": out})
    except WebSocketDisconnect:
        pass
