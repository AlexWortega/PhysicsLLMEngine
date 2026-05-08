# --- stage 1: build the React app (with wllama bundled into /assets) ---
FROM node:20-bookworm-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund --include=optional
COPY tsconfig*.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

# --- stage 2: tiny FastAPI server that serves /dist and /api/scenarios ---
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY backend/server.py ./server.py
COPY backend/examples ./examples
COPY --from=frontend /app/dist ./dist

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
