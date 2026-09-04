"""
metricas_baseline.py

Calcula MSE, MAE e RMSE do modelo e dos baselines (persistencia e media)
sobre o MESMO conjunto de teste, para uma comparacao justa. Imprime tambem
a melhoria do modelo sobre a persistencia e o R².

O baseline de persistencia (repetir o ultimo valor observado) e a referencia
que valida o modelo: superar a media e facil, superar a persistencia mostra
que o modelo aprendeu dinamica temporal real.
"""

import torch
import pandas as pd
import joblib
from torch.utils.data import DataLoader, ConcatDataset
from modelos import iTransformer, JanelaDataset

# ============================ CONFIGURACAO ============================
T, S       = 96, 24
DATA_DIR   = "data"
MODELO_PTH = "itransformer_melhor.pth"
SCALER_PKL = "scaler.pkl"
FILES      = ["VAH01.csv", "VAH17.csv", "VAH27.csv"]   # datasets de treino/teste
SPLIT      = 0.8                                        # fracao de treino (resto = teste)
# =====================================================================


# ---------------- setup ----------------
scaler = joblib.load(SCALER_PKL)
model = iTransformer(seq_len=T, pred_len=S, d_model=128, n_heads=4, n_blocks=2)
model.load_state_dict(torch.load(MODELO_PTH, map_location="cpu"))
model.eval()


def carrega_norm(caminho):
    df = pd.read_csv(caminho)
    v = df.drop(columns=["time_s", "cycleNumber"]).values
    return torch.tensor(scaler.transform(v), dtype=torch.float32)


# ---------------- conjunto de teste (ultimos 20% de cada arquivo) ----------------
# split e janelamento POR arquivo, para nao cruzar fronteiras (mesma logica do treino)
test_sets = []
for f in FILES:
    serie = carrega_norm(f"{DATA_DIR}/{f}")
    corte = int(len(serie) * SPLIT)
    test_sets.append(JanelaDataset(serie[corte:], T, S))
test_loader = DataLoader(ConcatDataset(test_sets), batch_size=64)


# ---------------- coleta erros do modelo e dos baselines ----------------
def coletar_erros():
    err_modelo, err_persist, err_media = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            # modelo
            err_modelo.append(model(xb) - yb)
            # persistencia: repete o ultimo instante da entrada por todo o horizonte
            ultimo = xb[:, -1:, :]                              # [B, 1, N]
            pred_p = ultimo.expand(-1, yb.shape[1], -1)         # [B, S, N]
            err_persist.append(pred_p - yb)
            # media: no espaco normalizado a media de cada sensor e ~0
            err_media.append(torch.zeros_like(yb) - yb)
    return (torch.cat(err_modelo), torch.cat(err_persist), torch.cat(err_media))


def metricas(err):
    mse  = (err ** 2).mean().item()
    mae  = err.abs().mean().item()
    rmse = mse ** 0.5
    return mse, mae, rmse


# ---------------- executa ----------------
if __name__ == "__main__":
    err_modelo, err_persist, err_media = coletar_erros()

    resultados = {
        "Modelo":       metricas(err_modelo),
        "Persistencia": metricas(err_persist),
        "Media":        metricas(err_media),
    }

    # tabela
    print(f"\n{'':16s} {'MSE':>10s} {'MAE':>10s} {'RMSE':>10s}")
    print("-" * 50)
    for nome, (mse, mae, rmse) in resultados.items():
        print(f"{nome:16s} {mse:10.4f} {mae:10.4f} {rmse:10.4f}")

    # melhoria do modelo sobre a persistencia (o baseline que importa)
    mse_m, mae_m, rmse_m = resultados["Modelo"]
    mse_p, mae_p, rmse_p = resultados["Persistencia"]
    mse_med = resultados["Media"][0]

    print(f"\nMelhoria do modelo vs. persistencia:")
    print(f"  MSE:  {100 * (1 - mse_m / mse_p):5.1f}%")
    print(f"  MAE:  {100 * (1 - mae_m / mae_p):5.1f}%")
    print(f"  RMSE: {100 * (1 - rmse_m / rmse_p):5.1f}%")

    # R² = 1 - MSE_modelo / MSE_media  (variancia explicada)
    print(f"\nR² (variancia explicada): {1 - mse_m / mse_med:.4f}")
