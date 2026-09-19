from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from agents import graph

load_dotenv()

app = FastAPI()

class RunRequest(BaseModel):
    project_id: str

@app.post("/run_graph")
async def run_graph(req: RunRequest):
    try:
        result = graph.invoke({"project_id": req.project_id})
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)