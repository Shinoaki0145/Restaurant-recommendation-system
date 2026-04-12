# Restaurant Recommendation System

Repo nay duoc tach thanh 2 phan:

- `restaurant_web/`: frontend React + Vite
- `backend/`: FastAPI + ranker

## Cau truc

```text
.
|-- backend/
|-- dataset/
|-- notebooks/
`-- restaurant_web/
```

## Lenh can chay truoc khi start

### 1. Cai frontend

```bash
cd restaurant_web
npm install
```

### 2. Cai backend

```bash
pip install -r backend/requirements.txt
```

## Chay du an

### Chay backend

Tu root repo:

```bash
python -m backend.api
```

Hoac:

```bash
python backend/api.py
```

### Chay frontend

Tu `restaurant_web/`:

```bash
npm run dev
```

## Ghi chu

- Backend doc bien moi truong tu `.env` o root repo. Neu can, no van fallback sang `restaurant_web/.env` de tuong thich voi cau truc cu.
- Artifact model duoc luu trong `backend/artifacts/`.
- Dataset van nam trong `dataset/`.
