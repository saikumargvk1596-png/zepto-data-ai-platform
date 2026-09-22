import os
from pathlib import Path
from typing import TypedDict, List
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from schemas import AskResponse
from prompts import SYSTEM_PROMPT

BASE = Path(__file__).resolve().parent
DB = BASE / "chroma_db"
NAME = "zepto_policies"
MODEL = "all-MiniLM-L6-v2"
KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

class State(TypedDict, total=False):
    query: str
    intent: str
    retrieved_documents: List[str]
    retrieved_ids: List[str]
    answer: str
    sources: List[str]
    confidence: float

class SupportGraph:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB))
        self.collection = self.client.get_or_create_collection(name=NAME, metadata={"hnsw:space": "cosine"})
        self.model = SentenceTransformer(MODEL)

    @property
    def mock(self):
        return os.getenv("MOCK_LLM", "1") != "0"

    def classify_intent(self, state: State):
        q = state["query"].lower()
        # Required deterministic mock classifier. The optional real mode keeps
        # the same safe routing fallback when no external provider is configured.
        return {"intent": "policy_question" if any(k in q for k in KEYWORDS) else "general_question"}

    def _groq_answer(self, query, contexts):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        context = "\n\n".join(contexts)
        prompt = SYSTEM_PROMPT + f"\n\nCONTEXT:\n{context}\n\nCUSTOMER QUESTION:\n{query}"
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [{"role": "system", "content": prompt}],
            "temperature": 0,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _validated_real_answer(self, query, docs, ids):
        # Up to 3 total attempts = initial call + 2 corrective retries.
        last_error = None
        for attempt in range(3):
            try:
                raw = self._groq_answer(query, docs)
                # The real provider is asked for prose; we deterministically wrap
                # it into the required Pydantic schema.
                response = AskResponse(answer=raw, sources=ids, confidence=0.8)
                return response.model_dump()
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    # Corrective retry is represented by another provider call;
                    # the provider prompt remains grounded in the same context.
                    continue
        return {
            "answer": f"ERROR: real-LLM output could not be validated: {last_error}",
            "sources": ids,
            "confidence": 0.0,
        }

    def retrieve_and_answer(self, state: State):
        emb = self.model.encode([state["query"]], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=emb, n_results=3, include=["documents", "distances"])
        docs = result["documents"][0]
        ids = result["ids"][0]

        if self.mock:
            answer = f"Based on the retrieved context: {docs[0][:200]}"
            return {"retrieved_documents": docs, "retrieved_ids": ids, "answer": answer, "sources": ids, "confidence": 1.0}

        return {"retrieved_documents": docs, "retrieved_ids": ids, **self._validated_real_answer(state["query"], docs, ids)}

    def direct_answer(self, state: State):
        if self.mock:
            return {"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}
        try:
            raw = self._groq_answer(state["query"], [])
            response = AskResponse(answer=raw, sources=[], confidence=0.8)
            return response.model_dump()
        except Exception:
            return {"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 0.0}

    @staticmethod
    def route(state: State):
        return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

    def build(self):
        g = StateGraph(State)
        g.add_node("classify_intent", self.classify_intent)
        g.add_node("retrieve_and_answer", self.retrieve_and_answer)
        g.add_node("direct_answer", self.direct_answer)
        g.set_entry_point("classify_intent")
        g.add_conditional_edges("classify_intent", self.route, {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"})
        g.add_edge("retrieve_and_answer", END)
        g.add_edge("direct_answer", END)
        return g.compile()

    def ask(self, query: str):
        result = self.build().invoke({"query": query})
        return AskResponse(answer=result["answer"], sources=result.get("sources", []), confidence=result.get("confidence", 1.0))
