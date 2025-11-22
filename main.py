from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
service = None  # não carrega imediatamente

class PredictInput(BaseModel):
    x_all: list
    edge_index: list
    mask: list | None = None

class PredictOutput(BaseModel):
    predictions: list

@app.post("/predict", response_model=PredictOutput)
def predict(payload: PredictInput):
    global service
    if service is None:
        from model_service import ModelService
        service = ModelService()  # carrega só na primeira chamada

    x_tensor = torch.tensor(payload.x_all, dtype=torch.float)
    edge_tensor = torch.tensor(payload.edge_index, dtype=torch.long)
    mask_tensor = torch.tensor(payload.mask, dtype=torch.bool) if payload.mask else None

    preds = service.predict(x_tensor, edge_tensor, mask_tensor)
    return PredictOutput(predictions=preds.tolist())
