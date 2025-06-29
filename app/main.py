
# app/main.py
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from io import BytesIO
from PIL import Image
import psycopg2
from collections import defaultdict

# CONFIG (ideal: usar variables de entorno)
PG_CONFIG = {
    'dbname': 'SCOUT_DB',
    'user': 'postgres',
    'password': '12345678',
    'host': 'localhost',
    'port': 5432,
}
THRESHOLD = 0.5

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar modelo y embeddings en memoria
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Carga embeddings robusta

def load_embeddings():
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT estudiante_id, embedding FROM personas_estudiantefoto WHERE embedding IS NOT NULL"
    )
    data = cur.fetchall()
    cur.close()
    conn.close()

    embs = defaultdict(list)
    for est_id, emb_val in data:
        # emb_val puede ser bytes o string JSON
        if isinstance(emb_val, (bytes, bytearray, memoryview)):
            raw = emb_val.tobytes() if isinstance(emb_val, memoryview) else emb_val
            emb = np.frombuffer(raw, dtype=np.float32)
        else:
            # asumir JSON almacenado como text
            arr = json.loads(emb_val)
            emb = np.array(arr, dtype=np.float32)
        embs[est_id].append(emb)
    return embs

embeddings_dict = load_embeddings()

@app.post("/generar_embedding/")
async def generar_embedding(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    faces = face_app.get(np.array(img))
    if not faces:
        return {"ok": False, "msg": "No se detectaron rostros"}
    return {"ok": True, "embedding": faces[0].embedding.tolist()}

@app.post("/match_faces/")
async def match_faces(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    faces = face_app.get(np.array(img))
    if not faces:
        return {"ok": False, "msg": "No se detectaron rostros"}

    results = []
    for f in faces:
        q = f.embedding
        # buscar mejor match
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
                q = f.embedding
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
