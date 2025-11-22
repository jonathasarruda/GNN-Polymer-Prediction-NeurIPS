import torch
from gnn_polymer_predictor import SimpleGNN

class ModelService:
    def __init__(self):
        try:
            checkpoint = torch.load("model/simple_gnn.pt", map_location="cpu")

            # Caso 1: arquivo tem metadados + state_dict
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                in_channels = checkpoint.get("in_channels", 17)
                hidden_channels = checkpoint.get("hidden_channels", 64)
                out_channels = checkpoint.get("out_channels", 5)
                state_dict = checkpoint["state_dict"]
            else:
                # Caso 2: arquivo é só state_dict
                in_channels = 17   # valores usados no treino
                hidden_channels = 64
                out_channels = 5
                state_dict = checkpoint

            self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        except Exception as e:
            print("Erro ao carregar o modelo:", e)

    def predict(self, x_all, edge_index, mask=None):
        with torch.no_grad():
            out = self.model(x_all, edge_index)
            if mask is not None:
                out = out[mask]
            return out.cpu().numpy()
