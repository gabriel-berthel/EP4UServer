from io import BytesIO
import signal
import pickle
import httpx
import torch

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import Response, StreamingResponse

from .piper_tts import PiperTTS
from .docling_converter import DoclingParser

# ---------------------------
# Helpers
# ---------------------------

def hash_file_contents(contents: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(contents)
    return h.hexdigest()

def cleanup():
    print("\nCleaning up GPU memory...")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("Cleanup done.")
    
def wav_to_mp3_bytes(wav_bytes: bytes) -> bytes:
    import subprocess

    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-i", "pipe:0",
            "-f", "mp3",
            "-codec:a", "libmp3lame",
            "-b:a", "192k",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    out, err = p.communicate(input=wav_bytes)
    if p.returncode != 0:
        raise RuntimeError(err.decode())

    return out

# ---------------------------
# Lifespan
# ---------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting service...")

    # Initialize shared services
    app.state.converter = DoclingParser()
    app.state.http_client = httpx.AsyncClient(timeout=60)

    yield

    print("Shutting down service...")
    await app.state.http_client.aclose()
    cleanup()

app = FastAPI(lifespan=lifespan)

# ---------------------------
# Signal handling (still useful)
# ---------------------------

def handle_signal(signum, frame):
    print(f"\nSignal {signum} received")
    cleanup()
    exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ---------------------------
# TTS Endpoint
# ---------------------------

@app.post("/tts")
async def tts(request: Request):
    data = await request.json()

    text = data.get("text")
    voice = data.get("voice")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")
    if not voice:
        raise HTTPException(status_code=400, detail="Missing 'voice'")

    model_path = f"./voices/{voice}.onnx"
    
    tts_engine = PiperTTS(model_path=model_path)
    audio_bytes = tts_engine.synthesize(text)
    mp3_bytes = wav_to_mp3_bytes(audio_bytes)
    
    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="speech.mp3"'}
    )

# ---------------------------
# Parse Endpoint
# ---------------------------

@app.post("/parse")
async def parse_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    contents = await file.read()
    filename = "document_tmp.pdf"

    with open(filename, "wb") as f:
        f.write(contents)

    try:
        result = app.state.converter.parse(filename)

        mem_file = BytesIO()
        pickle.dump(result, mem_file)
        mem_file.seek(0)

        return StreamingResponse(
            mem_file,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={filename}.pickle"
            }
        )

    except Exception as e:
        print("Error parsing:", e)
        raise HTTPException(status_code=500, detail="Error parsing file")

# ---------------------------
# Ollama Forwarding Endpoint
# ---------------------------

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

@app.post("/ollama")
async def forward_to_ollama(request: Request):
    """
    Transparent proxy to Ollama/OpenAI-compatible API.
    Just forwards JSON payload and returns response.
    """
    payload = await request.json()

    try:
        client: httpx.AsyncClient = app.state.http_client

        resp = await client.post(OLLAMA_URL, json=payload)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Entry point
# ---------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
if __name__ == "__main__":
    main()
