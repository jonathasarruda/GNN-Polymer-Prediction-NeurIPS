import torch
from gnn_polymer_predictor import SimpleGNN

class ModelService:
    def __init__(self):
        checkpoint = torch.load("model/simple_gnn.pt", map_location="cpu")

        in_channels = checkpoint.get("in_channels", 17)
        hidden_channels = checkpoint.get("hidden_channels", 64)
        out_channels = checkpoint.get("out_channels", 5)

        self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
