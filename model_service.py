import os
import torch
import traceback
from gnn_polymer_predictor import SimpleGNN

print("📁 Conteúdo da pasta model:", os.listdir("model"))

class ModelService:
    def __init__(self):
        print("🔍 Iniciando carregamento do modelo...")

        checkpoint_path = "model/simple_gnn.pt"

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print("✅ Checkpoint carregado com sucesso")

            # Detecta formato do checkpoint
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                print("📦 Checkpoint contém metadados")
                in_channels = checkpoint.get("in_channels", 17)
                hidden_channels = checkpoint.get("hidden_channels", 64)
                out_channels = checkpoint.get("out_channels", 5)
                state_dict = checkpoint["state_dict"]
            else:
                print("📦 Checkpoint é apenas state_dict")
                in_channels = 17
                hidden_channels = 64
                out_channels = 5
                state_dict = checkpoint

            print(f"📐 Parâmetros detectados: in={in_channels}, hidden={hidden_channels}, out={out_channels}")

            # Cria o modelo
            self.model = SimpleGNN(in_channels, hidden_channels, out_channels)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            print("✅ Modelo carregado e pronto")

        except Exception as e:
            print("❌ Erro ao carregar o modelo:")
            traceback.print_exc()
            raise e  # força o Render a mostrar o erro e parar aqui

    def predict(self, x_all, edge_index, mask=None):
        with torch.no_grad():
            out = self.model(x_all, edge_index)

            # Retorna só os nós selecionados
            if mask is not None:
                return out[mask].cpu().numpy()

            # Ou retorna todos os nós
            return out.cpu().numpy()
