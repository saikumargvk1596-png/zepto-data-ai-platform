from fastapi import FastAPI
from schemas import AskRequest,AskResponse
from graph import SupportGraph
app=FastAPI(title="Zepto Support Assistant",version="1.0")
graph=SupportGraph()
@app.get("/")
def root(): return {"service":"Zepto Support Assistant","mock_llm":graph.mock,"docs":"/docs"}
@app.post("/ask",response_model=AskResponse)
def ask(request:AskRequest): return graph.ask(request.query)
