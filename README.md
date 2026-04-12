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

Backend dung file cau hinh rieng tai `backend/.env`.
Frontend neu can thi dung file rieng tai `restaurant_web/.env`.

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

- Backend va frontend da duoc tach rieng.
- Backend chi doc bien moi truong tu `backend/.env`.
- Duong dan tuong doi trong `backend/.env` se uu tien resolve theo thu muc `backend/`.
- Frontend chi nam trong `restaurant_web/`.
- Artifact model duoc luu trong `backend/artifacts/`.
- Dataset van nam trong `dataset/`.
