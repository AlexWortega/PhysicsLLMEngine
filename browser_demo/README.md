---
title: Physics LLM
emoji: 🪀
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Draw physics scenes — let a fine-tuned LFM2 simulate them
---

# Physics LLM 🪀

Pick a physics scenario, optionally drop a few extra circles/boxes/walls into the scene, then let a fine-tuned `LFM2-350M` predict the next frames token-by-token. The simulation is streamed live to a Konva canvas as it's generated.

## Stack

- **Frontend**: React + Vite + react-konva (vector canvas)
- **Backend**: FastAPI + `llama-cpp-python`
- **Model**: [`AlexWortega/lfm2-scenarios-GGUF`](https://huggingface.co/AlexWortega/lfm2-scenarios-GGUF) · Q4_K_M (216 MB)
- **Streaming**: NDJSON over POST `/simulate`
- **Runtime**: pure CPU (free Space tier)

## Endpoints

- `GET  /` — React UI
- `GET  /api/scenarios` — bundled example scenes
- `POST /simulate` → `application/x-ndjson` stream of `{type: "frame", …}` events

## Local dev

```bash
# 1. Build the React app
npm install
npm run build

# 2. Run the backend (serves /dist + /simulate on port 7860)
pip install -r backend/requirements.txt
uvicorn backend.server:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860 .

## Notes

- First request waits ~30 s while the GGUF (≈216 MB) downloads from the Hub. Subsequent requests are instant.
- 50-frame rollouts in ~30–60 s on a 2-vCPU CPU Space.
- 6 of the 30 scenarios (`pong`, `bowling`, `ramp_roll`, `angry_birds`, `hourglass`, `newtons_cradle`) were never seen at training time — try them to probe out-of-distribution generalization.
