# Restaurant Web

Frontend React/Vite lives here. Backend da tach rieng hoan toan sang thu muc [../backend](../backend).

## Run frontend

```bash
npm install
npm run dev
```

## Notes

- Frontend local env can stay in [`.env`](./.env) if needed.
- Backend co env rieng trong [`../backend/.env`](../backend/.env).
- Model artifacts are stored in [`../backend/artifacts/`](../backend/artifacts/).
- Training data still stays in [`../dataset/`](../dataset/) to avoid duplicating large CSV files.
- Inference after Pinecone is configured as Supabase-only for restaurant metadata.
