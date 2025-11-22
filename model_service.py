import torch
from gnn_polymer_predictor import SimpleGNN

class ModelService:
    def __init__(self):
        in_channels = 17           # igual ao x_all.shape[1] no Kaggle
        hidden_channels = 64       
        out_channels = 6           # igual ao len(target_cols) no Kaggle

        self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
        
        try:
            self.model.load_state_dict(torch.load("model/simple_gnn.pt", map_location="cpu"))
        except Exception as e:
            print("Erro ao carregar os pesos do modelo:", e)
        
        self.model.eval()

    def predict(self, x_all, edge_index, mask=None):
        with torch.no_grad():
            out = self.model(x_all, edge_index)
            if mask is not None:
                out = out[mask]
            return out.cpu().numpy()
