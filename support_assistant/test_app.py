from fastapi.testclient import TestClient
from main import app
client=TestClient(app)
def test_policy():
    r=client.post("/ask",json={"query":"What is the delivery fee?"}); assert r.status_code==200; assert r.json()["sources"]
def test_general():
    r=client.post("/ask",json={"query":"Tell me a joke."}); assert r.status_code==200; assert r.json()["sources"]==[]
