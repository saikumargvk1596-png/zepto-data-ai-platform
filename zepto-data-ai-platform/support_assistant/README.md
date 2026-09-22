# Support Assistant

Architecture: `8 policy documents -> Sentence Transformers -> ChromaDB -> classify_intent -> retrieve_and_answer/direct_answer -> Pydantic -> FastAPI /ask`.

The default `MOCK_LLM` mode is fully local and deterministic. Policy queries are identified using the required keyword heuristic. Policy queries retrieve top-3 ChromaDB chunks and produce `Based on the retrieved context: ...`; general queries return the fixed policy-only message.

Run:
```bash
python ingest.py
uvicorn main:app --reload
```

Examples:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the delivery fee?\"}"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"Tell me a joke.\"}"
```

Docker:
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

Optional real mode uses Groq when `MOCK_LLM=0` and `GROQ_API_KEY` is supplied. The real generation path retries up to two additional times after validation/provider failures and returns a marked error response if it still fails.
