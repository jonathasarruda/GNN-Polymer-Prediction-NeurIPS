import torch
from gnn_polymer_predictor import SimpleGNN

class ModelService:
    def __init__(self):
        # Carrega o checkpoint salvo no Kaggle
        checkpoint = torch.load("model/simple_gnn.pt", map_location="cpu")

        # Se o arquivo tiver metadados, usa eles
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            in_channels = checkpoint.get("in_channels")
            hidden_channels = checkpoint.get("hidden_channels", 64)
            out_channels = checkpoint.get("out_channels")
            state_dict = checkpoint["state_dict"]
        else:
            # Caso seja apenas state_dict puro (sem metadados)
            # ⚠️ Aqui você precisa colocar os valores reais usados no treino
            in_channels = 17
            hidden_channels = 64
            out_channels = 5
            state_dict = checkpoint

        # Instancia o modelo com os parâmetros corretos
        self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, x_all, edge_index, mask=None):
        with torch.no_grad():
            out = self.model(x_all, edge_index)
            if mask is not None:
                out = out[mask]
            return out.cpu().numpy()
