"""Static-file backend for the physics-llm Space.

The model now runs in the user's browser via wllama (WASM). This server only:
  - serves the built React app from /dist
  - exposes /api/scenarios (reads bundled JSONL examples)
  - sets COOP/COEP so the page can use SharedArrayBuffer for multi-threaded WASM
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

DIST_DIR = Path(__file__).parent / "dist"
EXAMPLES_DIR = Path(__file__).parent / "examples"

app = FastAPI(title="physics-llm")


class CrossOriginIsolationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        resp.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return resp


app.add_middleware(CrossOriginIsolationMiddleware)


@app.get("/api/scenarios")
async def list_scenarios():
    items = []
    for p in sorted(EXAMPLES_DIR.glob("*.jsonl")):
        try:
            with open(p) as f:
                lines = [ln for ln in f if ln.strip()]
            header = json.loads(lines[0])
            initial = [json.loads(ln) for ln in lines[1:5] if ln.startswith("{")]
            items.append({
                "name": p.stem,
                "header": header,
                "initial_frames": initial,
            })
        except Exception as exc:
            print(f"[scenarios] skip {p.name}: {exc}", flush=True)
    return JSONResponse(items)


@app.get("/api/health")
async def health():
    return {"ok": True, "mode": "client-side-wasm"}


if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(DIST_DIR / "index.html")

    @app.get("/{path:path}")
    async def spa(path: str):
        candidate = DIST_DIR / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")
