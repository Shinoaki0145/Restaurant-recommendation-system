# Restaurant Web

Frontend React/Vite lives here. Backend API da duoc dua ra root repo tai [../backend/api.py](../backend/api.py).

## Run frontend

```bash
npm install
npm run dev
```

## Run backend API

From the repo root:

```bash
python -m backend.api
```

Or from inside `restaurant_web/`:

```bash
npm run api
```

## Notes

- Frontend local env can stay in [`.env`](./.env) if needed.
- Backend uu tien doc [`.env`](../.env), sau do moi fallback sang `restaurant_web/.env`.
- Model artifacts are stored in [`../backend/artifacts/`](../backend/artifacts/).
- Training data still stays in [`../dataset/`](../dataset/) to avoid duplicating large CSV files.
- Inference after Pinecone is configured as Supabase-only for restaurant metadata.
