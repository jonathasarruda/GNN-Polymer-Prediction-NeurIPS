import torch
from gnn_polymer_predictor import SimpleGNN

class ModelService:
    def __init__(self):
        in_channels = 128          # ou x_all.shape[1] se quiser calcular dinamicamente
        hidden_channels = 64       # confirmado no seu código
        out_channels = 5           # confirmado via target_cols

        self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
        self.model.load_state_dict(torch.load("model/simple_gnn.pt", map_location="cpu"))
        self.model.eval()

    def predict(self, x_all, edge_index, mask=None):
        with torch.no_grad():
            out = self.model(x_all, edge_index)
            if mask is not None:
                out = out[mask]
            return out.cpu().numpy()
