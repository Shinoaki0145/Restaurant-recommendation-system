# Restaurant Web

Frontend React/Vite lives here, and the restaurant ranking API now lives in [backend/api.py](./backend/api.py).

## Run frontend

```bash
npm install
npm run dev
```

## Run backend API

From inside `restaurant_web/`:

```bash
npm run api
```

Or directly:

```bash
python backend/api.py
```

## Notes

- Backend config is read from [`.env`](./.env).
- Model artifacts are stored in [`artifacts/`](./artifacts).
- Training data still stays in `../dataset/` to avoid duplicating large CSV files.
- Inference after Pinecone is configured as Supabase-only for restaurant metadata.
