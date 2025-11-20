import torch
from gnn_polymer_predictor import SimpleGNN  # ajuste se a classe estiver em outro arquivo

class ModelService:
    def __init__(self):
        # Parâmetros do modelo — ajuste conforme seu projeto
        in_channels = 128
        hidden_channels = 256
        out_channels = 10

        # Reconstruir a arquitetura
        self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
        self.model.load_state_dict(torch.load("model/simple_gnn.pt", map_location="cpu"))
        self.model.eval()

    def predict(self, x_all, edge_index, mask=None):
        with torch.no_grad():
            out = self.model(x_all, edge_index)
            if mask is not None:
                out = out[mask]
            return out.cpu().numpy()
