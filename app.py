from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import yaml
import torch
import time
import json
from torch.utils.data import DataLoader
from src.RagPipeline import RagPipeline
from src.QAdataset import QuestionsDataset

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list

#os.chdir("Multimodal-RAG-opensource")
with open('src/configs.yml', 'r') as f:
    config = yaml.safe_load(f)

conversational_chain = RagPipeline(config)

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    question = request.question
    try:
        result = conversational_chain.rag_chain_with_source.invoke(question)
        context = [doc.page_content for doc in result['context']]
        response = result['answer']
        
        return QueryResponse(
            question=question,
            answer=response,
            contexts=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
