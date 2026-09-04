"""
anomaly_plot.py
Generates per-window anomaly-score plots for the selected files, highlighting
the peaks that exceed the threshold (the anomalies selected by the iTransformer
filter). Saves one PNG per file.
Reuses the pipeline components (modelos.py, scaler, trained model).
Adjust the names in the CONFIGURATION section to match your project.
"""
import matplotlib
matplotlib.use("Agg")          # headless backend: writes PNG directly
                               # (essential in an environment without a display)
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from modelos import iTransformer, JanelaDataset

# ============================ CONFIGURATION ============================
T, S        = 96, 24
DATA_DIR    = "data"
MODELO_PTH  = "itransformer_melhor.pth"
SCALER_PKL  = "scaler.pkl"
# healthy baselines that define the reference (same as training)
BASELINES   = ["VAH01.csv", "VAH17.csv", "VAH27.csv"]
# files to plot (illustrative contrast: one baseline vs. one that fires a lot)
ALVOS       = ["VAH01.csv", "VAH09.csv"]
ALPHA       = 0.01             # false-positive fraction (percentile = 1 - ALPHA)
USAR_LOG    = True             # log scale on Y — recommended for heavy tails
# ======================================================================

# ---------------- setup: model and scaler ----------------
scaler = joblib.load(SCALER_PKL)
model = iTransformer(seq_len=T, pred_len=S, d_model=128, n_heads=4, n_blocks=2)
model.load_state_dict(torch.load(MODELO_PTH, map_location="cpu"))
model.eval()

def carrega_norm(caminho):
    df = pd.read_csv(caminho)
    v = df.drop(columns=["time_s", "cycleNumber"]).values
    return torch.tensor(scaler.transform(v), dtype=torch.float32)

def coletar_residuos(loader):
    res = []
    with torch.no_grad():
        for xb, yb in loader:
            res.append(yb - model(xb))
    return torch.cat(res)

# ---------------- healthy reference (frozen, single ruler) ----------------
res_ref = []
for bf in BASELINES:
    serie = carrega_norm(f"{DATA_DIR}/{bf}")
    corte = int(len(serie) * 0.8)                       # training portion only
    loader = DataLoader(JanelaDataset(serie[:corte], T, S), batch_size=64)
    res_ref.append(coletar_residuos(loader))
res_ref = torch.cat(res_ref)

mu_ref  = res_ref.mean(dim=0)
sig_ref = res_ref.std(dim=0)

def escores(residuos):
    z = (residuos - mu_ref) / sig_ref
    return (z ** 2).sum(dim=(1, 2)).numpy()

limiar = np.percentile(escores(res_ref), 100 * (1 - ALPHA))

# ---------------- plot ----------------
def plotar(nome_arquivo):
    serie = carrega_norm(f"{DATA_DIR}/{nome_arquivo}")
    loader = DataLoader(JanelaDataset(serie, T, S), batch_size=64)
    sc = escores(coletar_residuos(loader))
    anomalas = sc > limiar

    plt.figure(figsize=(14, 4))
    plt.plot(sc, linewidth=0.5, color="steelblue", label="per-window score")
    plt.axhline(limiar, color="red", linestyle="--", linewidth=1,
                label=f"threshold (p{100*(1-ALPHA):.0f})")
    if anomalas.any():
        plt.scatter(np.where(anomalas)[0], sc[anomalas],
                    color="red", s=8, zorder=3, label="flagged anomalies")
    if USAR_LOG:
        plt.yscale("log")
    plt.xlabel("window index (time →)")
    plt.ylabel("anomaly score" + (" (log scale)" if USAR_LOG else ""))
    plt.title(f"Per-window anomaly score — {nome_arquivo} "
              f"({100*anomalas.mean():.2f}% flagged)")
    plt.legend()
    plt.tight_layout()
    saida = f"anomalias_{nome_arquivo.replace('.csv', '')}.png"
    plt.savefig(saida, dpi=150)
    plt.close()
    print(f"{nome_arquivo}: {anomalas.sum()}/{len(sc)} flagged "
          f"({100*anomalas.mean():.2f}%) -> {saida}")

# ---------------- run ----------------
if __name__ == "__main__":
    print(f"threshold (p{100*(1-ALPHA):.0f} of healthy reference): {limiar:.1f}\n")
    for alvo in ALVOS:
        plotar(alvo)

