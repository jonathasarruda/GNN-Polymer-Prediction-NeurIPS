import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path

print("🚀 Gerando arquivos x_all.pt e edge_index.pt ...")

# -----------------------------
# 1. Ler dados originais
# -----------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# -----------------------------------
# 2. Função de extração das features
# -----------------------------------
def smiles_robust_features(smiles):
    atom_counts = {
        'C': len(re.findall(r'C(?![a-z])', smiles)),
        'O': len(re.findall(r'O', smiles)),
        'N': len(re.findall(r'N(?![a-z])', smiles)),
        'S': len(re.findall(r'S(?![a-z])', smiles)),
        'F': len(re.findall(r'F', smiles)),
        'Cl': len(re.findall(r'Cl', smiles)),
        'Br': len(re.findall(r'Br', smiles)),
        'I': len(re.findall(r'I', smiles)),
        'P': len(re.findall(r'P', smiles)),
    }
    features = {
        'smiles_length': len(smiles),
        'num_branches': smiles.count('(') + smiles.count(')'),
        'num_double_bonds': smiles.count('='),
        'num_triple_bonds': smiles.count('#'),
        'num_ring_closures': len(re.findall(r'\d', smiles)),
        'num_aromatic_atoms': len(re.findall(r'[bcnops]', smiles)),
        'num_aliphatic_atoms': len(re.findall(r'[BCNOPSFHI]', smiles)),
        **atom_counts,
    }
    features['num_atoms_total'] = sum(atom_counts.values())
    features['num_atoms_unique'] = sum(v > 0 for v in atom_counts.values())
    return pd.Series(features)

# -----------------------------------
# 3. Extrair features
# -----------------------------------
train_feats = train['SMILES'].apply(smiles_robust_features)
test_feats = test['SMILES'].apply(smiles_robust_features)

numerical_cols = train_feats.columns.tolist()

# -----------------------------------
# 4. Normalizar os dados
# -----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(train_feats[numerical_cols])
X_test = scaler.transform(test_feats[numerical_cols])

# -----------------------------------
# 5. Combinar tudo
# -----------------------------------
X_all = np.vstack([X_train, X_test])
x_all_tensor = torch.tensor(X_all, dtype=torch.float32)

# -----------------------------------
# 6. Construir grafo
# -----------------------------------
sim_matrix = cosine_similarity(X_all)
threshold = 0.8

edges = np.array(np.nonzero(np.triu(sim_matrix, k=1) > threshold))
edges = np.hstack([edges, edges[::-1]])  # bidirecional

edge_index_tensor = torch.tensor(edges, dtype=torch.long)

# -----------------------------------
# 7. Salvar arquivos
# -----------------------------------
Path("model").mkdir(exist_ok=True)

torch.save(x_all_tensor, "model/x_all.pt")
torch.save(edge_index_tensor, "model/edge_index.pt")

print("✅ Arquivos prontos:")
print(" - model/x_all.pt")
print(" - model/edge_index.pt")
