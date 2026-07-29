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

limiar = np.percentile(escores(res_ref), 99)     # CONGELADO (regua unica)
print(f"limiar unico (percentil 99 da referencia saudavel): {limiar:.1f}\n")

files = [f.name for f in Path("data/").iterdir() if f.is_file() and f.name != "README.txt"]
# df = pd.concat((pd.read_csv(f) for f in files))

for f in files:
    print("Arquivo: ", f)
    serie = carrega_norm("data/" + f)
    loader = DataLoader(JanelaDataset(serie, T, S), batch_size=64)
    sc = escores(coletar_residuos(loader))
    taxa = (sc > limiar).mean()
    print(f"{f:14s} | janelas anomalas: {100*taxa:.2f}%")
    # df = pd.read_csv("data/"+ f)
    # valores = df.drop(columns=["time_s", "cycleNumber"]).values   # numpy cru
    # scaler = StandardScaler()
    # data_norm = scaler.fit_transform(valores)                     # normaliza (numpy)
    # data_norm = torch.tensor(data_norm, dtype=torch.float32)      # -> tensor
    # T = 96   # comprimento da janela de entrada
    # S = 24   # horizonte de previsão
    # # split cronologico na SERIE LONGA, antes de janelar
    # corte = int(len(data_norm) * 0.8)
    # serie_treino = data_norm[:corte]
    # serie_teste  = data_norm[corte:]
    # # janelamento sob demanda (nao materializa X e Y na RAM)
    # train_loader = DataLoader(JanelaDataset(serie_treino, T, S),
    #                           batch_size=64, shuffle=True)
    # test_loader  = DataLoader(JanelaDataset(serie_teste, T, S),
    #                           batch_size=64)
    # model = iTransformer(seq_len=T, pred_len=S, d_model=128, n_heads=4, n_blocks=2)
    # model.load_state_dict(torch.load("itransformer_melhor.pth"))
    # model.eval()
    # def coletar_residuos(loader):
    #     res = []
    #     with torch.no_grad():
    #         for xb, yb in loader:
    #             res.append(yb - model(xb))
    #     return torch.cat(res)
    # res_ref = coletar_residuos(train_loader)
    # mu_ref = res_ref.mean(dim=0)
    # sig_ref = res_ref.std(dim=0)
    # # calcula o escore de anomalia (soma dos z^2) de um conjunto de residuos
    # def escores(residuos):
    #     z = (residuos - mu_ref) / sig_ref
    #     return (z ** 2).sum(dim=(1, 2)).numpy()   # [janelas]
    # # limiar EMPIRICO: percentil dos escores saudaveis (treino)
    # scores_ref = escores(res_ref)
    # alpha = 0.01                                   # fracao de falsos positivos desejada
    # limiar = np.percentile(scores_ref, 100 * (1 - alpha))   # p/ alpha=0.01 -> percentil 99
    # # classifica o teste contra esse limiar
    # scores_test = escores(coletar_residuos(test_loader))
    # anomalias = scores_test > limiar
    # print(f"limiar empirico (percentil {100*(1-alpha):.0f}): {limiar:.1f}")
    # print(f"janelas sinalizadas: {anomalias.sum()} de {len(anomalias)} "
    #       f"({100*anomalias.mean():.2f}%)")
    # res_ref = coletar_residuos(train_loader)
    # mu_ref = res_ref.mean(dim=0)
    # sig_ref = res_ref.std(dim=0)
    #
    # def p_valores(residuos):
    #     z = (residuos - mu_ref)/sig_ref
    #     score = (z**2).sum(dim=(1,2)).numpy()
    #     gl = z.shape[1] * z.shape[2]
    #     return chi2.sf(score, df=gl)
    #
    # res_test = coletar_residuos(test_loader)
    # p_test = p_valores(res_test)
    #
    # alpha = 0.01
    # anomalias = p_test < alpha
    # print(f"janelas sinalizadas: {anomalias.sum()} de {len(anomalias)} "
    #       f"({100*anomalias.mean():.2f}%)")

