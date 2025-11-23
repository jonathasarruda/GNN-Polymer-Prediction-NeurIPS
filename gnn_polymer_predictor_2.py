# gnn_polymer_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNN(nn.Module):
    def __init__(self, node_feature_dim=16, hidden_dim=32, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(node_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: tensor [batch, node_feature_dim]
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h)


class GNNEstimator:
    """
    Wrapper que carrega o modelo apenas 1 vez e faz inferência.
    Sem leitura de CSV do Kaggle.
    """

    def __init__(self, model_path=None):
        self.model = SimpleGNN()

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                print(f"✅ Modelo carregado de {model_path}")
            except Exception as e:
                print(f"⚠️ Falha ao carregar modelo ({e}). Usando pesos aleatórios.")

        self.model.eval()

    def predict(self, features: list[float]):
        """
        features: lista de floats enviados pela API
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(x).item()
        return float(pred)
