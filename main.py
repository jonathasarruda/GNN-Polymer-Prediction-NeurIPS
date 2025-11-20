from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model_service import ModelService

app = FastAPI()
service = ModelService()

class PredictInput(BaseModel):
    x_all: list  # matriz de features dos nós
    edge_index: list  # lista de arestas (pares de índices)
    mask: list | None = None  # máscara opcional para seleção de nós

class PredictOutput(BaseModel):
    predictions: list  # lista de predições do modelo

@app.post("/predict", response_model=PredictOutput)
def predict(payload: PredictInput):
    # Converter entrada para tensores PyTorch
    x_tensor = torch.tensor(payload.x_all, dtype=torch.float)
    edge_tensor = torch.tensor(payload.edge_index, dtype=torch.long)
    mask_tensor = torch.tensor(payload.mask, dtype=torch.bool) if payload.mask else None

    # Fazer predição usando o serviço
    preds = service.predict(x_tensor, edge_tensor, mask_tensor)

    # Retornar como lista JSON
    return PredictOutput(predictions=preds.tolist())
