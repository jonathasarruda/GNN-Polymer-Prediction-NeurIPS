from fastapi import FastAPI
from pydantic import BaseModel
import torch

print("🚀 Iniciando FastAPI...")

app = FastAPI()
service = None  # só carrega quando necessário

class PredictInput(BaseModel):
    mask: list   # agora só precisa disso

class PredictOutput(BaseModel):
    predictions: list

@app.get("/")
def root():
    return {"status": "ok", "message": "API está rodando"}

@app.post("/predict", response_model=PredictOutput)
def predict(payload: PredictInput):
    global service
    if service is None:
        print("📦 Carregando ModelService pela primeira vez...")
        from model_service import ModelService
        service = ModelService()

        # Carrega o grafo completo já salvo
        service.x_all = torch.load("model/x_all.pt")
        service.edge_index = torch.load("model/edge_index.pt")

    try:
        mask_tensor = torch.tensor(payload.mask, dtype=torch.long)

        preds = service.predict(service.x_all, service.edge_index, mask_tensor)
        return PredictOutput(predictions=preds.tolist())

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
