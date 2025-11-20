import torch
from gnn_polymer_predictor import SimpleGNN

class ModelService:
    def __init__(self):
        in_channels = 128          
        hidden_channels = 64       
        out_channels = 5          

        self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
        self.model.load_state_dict(torch.load("model/simple_gnn.pt", map_location="cpu"))
        self.model.eval()

    def predict(self, x_all, edge_index, mask=None):
        with torch.no_grad():
            out = self.model(x_all, edge_index)
            if mask is not None:
                out = out[mask]
            return out.cpu().numpy()
