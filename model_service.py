import torch
from gnn_polymer_predictor import SimpleGNN

class ModelService:
    def __init__(self):
        checkpoint = torch.load("model/simple_gnn.pt", map_location="cpu")

        # Se for dict com state_dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            in_channels = checkpoint.get("in_channels", 17)
            hidden_channels = checkpoint.get("hidden_channels", 64)
            out_channels = checkpoint.get("out_channels", 5)
            state_dict = checkpoint["state_dict"]
        else:
            # Se for apenas state_dict puro
            in_channels = 17   # igual ao treino
            hidden_channels = 64
            out_channels = 5   # igual ao treino
            state_dict = checkpoint

        self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
        try:
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print("Erro ao carregar pesos:", e)
        self.model.eval()

    def predict(self, x_all, edge_index, mask=None):
        with torch.no_grad():
            out = self.model(x_all, edge_index)
            if mask is not None:
                out = out[mask]
            return out.cpu().numpy()
