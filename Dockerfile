# RescueLens: one container = React UI + FastAPI + ONNX models.
#   docker build -t rescuelens .
#   docker run -p 7860:7860 rescuelens        -> http://localhost:7860
# Models are fetched at build time; after uploading new models use "Factory rebuild" on the Space (cache would keep old ones).

FROM node:22-slim AS ui
WORKDIR /ctx
COPY . .
RUN mkdir -p /ui/dist && if [ -f frontend/package.json ]; then \
      cd frontend && (npm ci || npm install) && npm run build && cp -r dist/. /ui/dist/; fi

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/srv/models FRONTEND_DIST=/srv/frontend/dist
WORKDIR /srv
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
ARG MODEL_BASE_URL="https://huggingface.co/PriyanshuSenapati/rescuelens-models/resolve/main"
COPY backend/fetch_models.py backend/fetch_models.py
RUN MODEL_BASE_URL="$MODEL_BASE_URL" python backend/fetch_models.py
COPY data/ data/
COPY docs/fixtures/ docs/fixtures/
COPY backend/app/ backend/app/
COPY --from=ui /ui/dist frontend/dist
RUN useradd -m -u 1000 user && chown -R user /srv
USER user
WORKDIR /srv/backend
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
