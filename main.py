import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from modelos import iTransformer, JanelaDataset
import joblib
from torch.utils.data import ConcatDataset

CHECKPOINT = "itransformer_checkpoint.pth"  # estado completo (modelo + otimizador + epoca)
MELHOR = "itransformer_melhor.pth"          # so os pesos do melhor modelo

if __name__ == "__main__":

    files = ["VAH01.csv", "VAH17.csv", "VAH27.csv"]
    T, S = 96, 24

    # --- 1. carrega e corta CADA arquivo cronologicamente (ainda em numpy cru) ---
    treino_cru, teste_cru = [], []
    for f in files:
        df = pd.read_csv(f)
        v = df.drop(columns=["time_s", "cycleNumber"]).values
        corte = int(len(v) * 0.8)
        treino_cru.append(v[:corte])       # 80% inicial de CADA bateria -> treino
        teste_cru.append(v[corte:])        # 20% final de CADA bateria -> teste

    # --- 2. ajusta o scaler SO no treino combinado (sem ver o teste) ---
    scaler = StandardScaler()
    scaler.fit(np.concatenate(treino_cru, axis=0))     # fit so no treino
    joblib.dump(scaler, "scaler.pkl")                  # salva p/ o detector usar depois

    # --- 3. aplica (transform) e janela CADA pedaco separadamente ---
    def to_ds(arr):
        t = torch.tensor(scaler.transform(arr), dtype=torch.float32)
        return JanelaDataset(t, T, S)

    train_ds = ConcatDataset([to_ds(a) for a in treino_cru])
    test_ds  = ConcatDataset([to_ds(a) for a in teste_cru])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    model = iTransformer(seq_len=T, pred_len=S, d_model=128, n_heads=4, n_blocks=2)

    #TREINO
    # otimizador = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=20)
    # perda_fn = nn.MSELoss()
    #
    # # retoma o treino se ja existir checkpoint (precisa vir depois de model/otimizador)
    # epoca_inicial = 0
    # melhor = float("inf")
    # if os.path.exists(CHECKPOINT):
    #     ckpt = torch.load(CHECKPOINT, map_location="cpu")
    #     model.load_state_dict(ckpt["model"])
    #     otimizador.load_state_dict(ckpt["otimizador"])
    #     scheduler.load_state_dict(ckpt["scheduler"])
    #     epoca_inicial = ckpt["epoca"]
    #     melhor = ckpt["melhor"]
    #     print(f"checkpoint encontrado — retomando da epoca {epoca_inicial}")
    # elif os.path.exists(MELHOR):
    #     # arquivo antigo: contem apenas os pesos, sem estado do otimizador
    #     model.load_state_dict(torch.load(MELHOR, map_location="cpu"))
    #     print("sem checkpoint — carregando pesos de", MELHOR)
    # else:
    #     print("nenhum checkpoint — treino do zero")
    #
    # EPOCAS = 15
    # for epoca in range(epoca_inicial, epoca_inicial + EPOCAS):
    #     model.train()
    #     total = 0.0; n = 0
    #     for xb, yb in train_loader:
    #         otimizador.zero_grad()
    #         loss = perda_fn(model(xb), yb)
    #         loss.backward()
    #         otimizador.step()
    #         total += loss.item() * len(xb); n += len(xb)
    #
    #     # avaliacao em lotes (nao passa o teste inteiro de uma vez)
    #     model.eval()
    #     val = 0.0; nv = 0
    #     with torch.no_grad():
    #         for xb, yb in test_loader:
    #             val += perda_fn(model(xb), yb).item() * len(xb); nv += len(xb)
    #
    #     scheduler.step()
    #     print(f"epoca {epoca:2d} | treino {total/n:.4f} | teste {val/nv:.4f}")
    #     val_epoca = val/nv
    #     if val_epoca < melhor:                       # <- salva o MELHOR
    #         melhor = val_epoca
    #         torch.save(model.state_dict(), MELHOR)
    #         print(f"   (novo melhor: {melhor:.4f} salvo)")
    #
    # torch.save({
    #     "model": model.state_dict(),
    #     "otimizador": otimizador.state_dict(),
    #     "scheduler": scheduler.state_dict(),
    #     "epoca": epoca + 1,               # proxima epoca a rodar
    #     "melhor": melhor,                 # senao o proximo run sobrescreve o melhor
    # }, CHECKPOINT)

    #Benchmark
    model.load_state_dict(torch.load("itransformer_melhor.pth"))
    model.eval()

    preds, reais = [],[]
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb))
            reais.append(yb)

    preds = torch.cat(preds)
    reais = torch.cat(reais)

    # MSE = media((pred - real)²)
    # MAE = media( |pred - real| )
    # RMSE = sqrt(MSE)

    difs = []
    for i in range(len(preds)):
        difs.append(preds[i] - reais[i]) 

    MSE = np.mean([x ** 2 for x in difs])
    print("MSE: ", MSE)

    MAE = np.mean(np.abs(difs))
    print("MAE: ", MAE)

    print("RMSE: ", np.sqrt(MSE))
