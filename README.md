# Restaurant Recommendation System

Project duoc tach thanh 2 phan:

- `backend/`: FastAPI API + ranker
- `restaurant_web/`: frontend React + Vite

## Cai dat

### Backend

```bash
pip install -r backend/requirements.txt
```

### Frontend

```bash
cd restaurant_web
npm install
```

## Cau hinh moi truong

Khong can commit `.env` vao repo. Hay giu file `.env` that ben ngoai repo khi deploy.

Backend se doc bien moi truong theo thu tu:

1. Bien he thong co san tren server/container
2. File duoc chi dinh boi `BACKEND_ENV_FILE`
3. `./.env` o root repo
4. `backend/.env`

Neu `BACKEND_ENV_FILE` tro toi mot file `.env` nam ben ngoai repo, cac duong dan tuong doi nhu `RANKER_ARTIFACT_PATH=./restaurant_ranker.joblib` se duoc resolve theo thu muc chua file `.env` do.

Frontend khong con hardcode `http://localhost:8000`.

- Local dev: Vite tu proxy `/rank` va `/health` sang backend `http://localhost:8000`
- Production: neu frontend va backend khac domain, set `VITE_API_BASE_URL=https://your-backend-domain` luc build frontend

## Chay du an

### Backend

Tu root repo, dung mot trong cac lenh sau:

```bash
python -m backend.api
```

```bash
python backend/api.py
```

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd restaurant_web
npm run dev
```

## Khi deploy can push gi

Neu chi deploy app runtime, thong thuong chi can:

- `backend/`
- `restaurant_web/`
- `README.md`
- file dependency manifests nhu `backend/requirements.txt`, `restaurant_web/package.json`, `restaurant_web/package-lock.json`

Khong nen push:

- `.env`
- notebooks, crawler, file dataset lon
- artifact neu muon de ben ngoai repo; khi do set `RANKER_ARTIFACT_PATH` trong env

## Luu y

- Backend can co `RANKER_ARTIFACT_PATH` hop le de load model.
- Backend can `PINECONE_API_KEY` neu dung retrieval tu Pinecone.
- Neu lay metadata nha hang tu Postgres, can bat `RESTAURANT_DB_ENABLED=true` va cau hinh `DATABASE_URL` hoac `DB_*`.
