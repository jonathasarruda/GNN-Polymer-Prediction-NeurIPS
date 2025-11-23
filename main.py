from fastapi import FastAPI
from pydantic import BaseModel
import torch

print("🚀 Iniciando FastAPI...")

app = FastAPI()
service = None  # carregado apenas na primeira chamada

class PredictInput(BaseModel):
    mask: list  # índices dos nós que você quer prever

class PredictOutput(BaseModel):
    predictions: list


@app.get("/")
def root():
    return {"status": "ok", "message": "API está rodando"}


@app.post("/predict", response_model=PredictOutput)
def predict(payload: PredictInput):
    global service

    # Carrega o serviço e o grafo completo apenas na primeira chamada
    if service is None:
        print("📦 Carregando ModelService e grafo completo pela primeira vez...")

        from model_service import ModelService
        service = ModelService()

        # Carrega o grafo que o modelo realmente usa
        service.x_all = torch.load("model/x_all.pt")          # shape [N, 17]
        service.edge_index = torch.load("model/edge_index.pt") # shape [2, E]

        print("✅ Grafo carregado.")

    try:
        # Converte máscara vinda da requisição
        mask_tensor = torch.tensor(payload.mask, dtype=torch.long)

        # Chama a predição real
        preds = service.predict(service.x_all, service.edge_index, mask_tensor)

        return PredictOutput(predictions=preds.tolist())

    except Exception as e:
        import traceback
        print("❌ Erro durante a predição:")
        traceback.print_exc()
        raise e
