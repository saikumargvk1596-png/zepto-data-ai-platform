from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
BASE=Path(__file__).resolve().parent; DOCS=BASE/"docs"; DB=BASE/"chroma_db"; NAME="zepto_policies"; MODEL="all-MiniLM-L6-v2"
def build_collection():
    client=chromadb.PersistentClient(path=str(DB)); col=client.get_or_create_collection(name=NAME,metadata={"hnsw:space":"cosine"}); model=SentenceTransformer(MODEL)
    paths=sorted(DOCS.glob("doc_*.txt")); docs=[p.read_text(encoding="utf-8").strip() for p in paths]; ids=[p.stem for p in paths]
    em=model.encode(docs,normalize_embeddings=True).tolist(); col.upsert(ids=ids,documents=docs,embeddings=em); print(f"Indexed {len(docs)} documents")
    return col
if __name__=="__main__": build_collection()
