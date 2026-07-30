import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from modelos import iTransformer, JanelaDataset
from scipy.stats import chi2
from pathlib import Path
import joblib

T, S = 96, 24

# ---------- FORA DO LOOP: monta a regua UMA vez ----------
scaler = joblib.load("scaler.pkl")            # o mesmo scaler do treino
model = iTransformer(seq_len=T, pred_len=S, d_model=128, n_heads=4, n_blocks=2)
model.load_state_dict(torch.load("itransformer_melhor.pth"))
model.eval()

def coletar_residuos(loader):
    res = []
    with torch.no_grad():
        for xb, yb in loader:
            res.append(yb - model(xb))
    return torch.cat(res)

def carrega_norm(caminho):
    df = pd.read_csv(caminho)
    v = df.drop(columns=["time_s", "cycleNumber"]).values
    return torch.tensor(scaler.transform(v), dtype=torch.float32)  # transform, nao fit

def non_maximum_supression(scores, T, S, k, largura=None):
    if largura is None:
        largura = T + S

    scores = scores.copy()
    selecionados = []
    for _ in range(k):
        i = int(np.argmax(scores))
        if scores[i] < 0:
            break
        selecionados.append(i)
        lo, hi = max(0, i - largura), min(len(scores), i + largura + 1)
        scores[lo:hi] = -1
    return selecionados

# referencia saudavel: os baselines de treino, INICIO de vida (saudavel)
ref_files = ["data/VAH01.csv", "data/VAH17.csv", "data/VAH27.csv"]
res_ref = []
for rf in ref_files:
    serie = carrega_norm(rf)
    corte = int(len(serie) * 0.8)                # so a parte de treino
    loader = DataLoader(JanelaDataset(serie[:corte], T, S), batch_size=64)
    res_ref.append(coletar_residuos(loader))
res_ref = torch.cat(res_ref)

mu_ref  = res_ref.mean(dim=0)                     # CONGELADO
sig_ref = res_ref.std(dim=0)                      # CONGELADO

def escores(residuos):
    z = (residuos - mu_ref) / sig_ref
    return (z ** 2).sum(dim=(1, 2)).numpy()

def payload_da_janela(serie, i, T, S):
    """Extrai a assinatura de anomalia da janela que comeca no indice i."""
    janela_in  = serie[i : i+T]                    # entrada [T, N]
    with torch.no_grad():
        prev = model(janela_in.unsqueeze(0))[0]    # previsto [S, N]
    real = serie[i+T : i+T+S]                       # observado [S, N]
    residuo = real - prev                           # [S, N]

    z = (residuo - mu_ref) / sig_ref               # z por sensor e horizonte [S, N]
    contrib = (z ** 2).sum(dim=0)                   # contribuicao de cada sensor [N]
    return {
        "indice": int(i),
        "escore": float((z ** 2).sum()),
        "contrib_por_sensor": contrib.numpy(),      # qual sensor puxou a anomalia
        "z_medio_por_sensor": z.mean(dim=0).numpy() # sinal do desvio (+ ou -) por sensor
    }

limiar = np.percentile(escores(res_ref), 99)     # CONGELADO (regua unica)
print(f"limiar unico (percentil 99 da referencia saudavel): {limiar:.1f}\n")

COLUNAS = list(pd.read_csv("data/VAH01.csv", nrows=0)
               .drop(columns=["time_s", "cycleNumber"]).columns)

files = [f.name for f in Path("data/").iterdir() if f.is_file() and f.name != "README.txt"]

for f in files:
    serie = carrega_norm("data/" + f)
    loader = DataLoader(JanelaDataset(serie, T, S), batch_size=64)
    sc = escores(coletar_residuos(loader))
    taxa = (sc > limiar).mean()

    # so seleciona picos que de fato passam o limiar
    sc_filtrado = np.where(sc > limiar, sc, -1.0)
    picos = non_maximum_supression(sc_filtrado, T, S, k=10)

    print(f"{f:14s} | anomalas: {100*taxa:5.2f}% | eventos distintos (NMS): {len(picos)}")
    for i in picos[:3]:                             # mostra os 3 principais
        p = payload_da_janela(serie, i, T, S)
        sensor_top = COLUNAS[int(p['contrib_por_sensor'].argmax())]
        print(f"    idx {p['indice']:6d} | escore {p['escore']:7.1f} | dominado por: {sensor_top}")

# # df = pd.concat((pd.read_csv(f) for f in files))
#
# for f in files:
#     print("Arquivo: ", f)
#     serie = carrega_norm("data/" + f)
#     loader = DataLoader(JanelaDataset(serie, T, S), batch_size=64)
#     sc = escores(coletar_residuos(loader))
#     taxa = (sc > limiar).mean()
#     non_maximum_supression(sc, T, S, k=)
#     print(f"{f:14s} | janelas anomalas: {100*taxa:.2f}%")
