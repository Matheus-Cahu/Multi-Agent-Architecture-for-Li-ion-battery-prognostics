#!/usr/bin/env python3
"""
Treino e avaliação do iTransformer com separação por fase de vida (um fold de
leave-one-cell-out).

Desenho
  Células de treino: usa só o INÍCIO DE VIDA, isto é, as primeiras --train-frac
  missões da vida útil. Vida útil = missão do EOL se a célula cruza o limiar de
  SOH; senão, fim do teste. Esse trecho é dividido cronologicamente (por missão)
  em treino / validação / referência:
      treino -> ajuste dos pesos e do scaler
      val    -> escolha do checkpoint (nunca a célula de teste)
      ref    -> mu_ref, sigma_ref e tau (disjunto do treino)

  Célula de teste: vida inteira, com cada janela rotulada pela missão do
  primeiro passo previsto:
      inicio    -> primeiras --train-frac missões da vida útil dela
                   (fidelidade de previsão e taxa de falso alarme)
      degradado -> missões após o EOL (só existe se a célula cruza o limiar)
      meio      -> restante (entra só na correlação escore x SOH)

  Janelas nunca atravessam: fronteira de missão, RPT, lacuna temporal ou linha
  inválida. RPTs são excluídos dos dados.

Exemplo (fold baseline com VAH17 como teste):
    python treino_eol.py --train data/VAH01.csv data/VAH27.csv \
                         --test data/VAH17.csv --seed 0 --backbone lstm

Requer soh_eol.py no mesmo diretório e iTransformer em modelos.py.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from soh_eol import discharge_segments, eol_crossing, find_missions, find_rpts

FEATURES_PADRAO = ["Ecell_V", "I_mA", "Temperature__C", "EnergyCharge_W_h",
                   "QCharge_mA_h", "EnergyDischarge_W_h", "QDischarge_mA_h"]
FASES = ["inicio", "meio", "degradado"]


# ============================================================ preparação
def prepara_celula(path: Path, features, a) -> dict:
    usecols = list(dict.fromkeys(["time_s", "Ecell_V", "I_mA", *features]))
    df = pd.read_csv(path, usecols=usecols)
    df = df.dropna(subset=["time_s", "Ecell_V", "I_mA"]).reset_index(drop=True)
    t = df["time_s"].to_numpy(dtype=np.float64)
    if np.any(np.diff(t) < 0):
        raise SystemExit(f"{path.name}: time_s não é monótono.")

    seg = discharge_segments(df, a.min_current, a.max_gap)
    rpt, rej = find_rpts(seg, a.rpt_i_min, a.rpt_i_max, a.rpt_min_hours, a.rpt_v_end)
    inicios_m = find_missions(seg, a.mission_i_min, a.mission_min_minutes,
                              a.mission_max_hours, a.mission_merge)
    if rpt.empty or len(inicios_m) == 0:
        raise SystemExit(f"{path.name}: RPTs ou missões não identificados.")

    # índice da missão a que cada linha pertence (-1 = antes da 1ª missão)
    missao = np.searchsorted(inicios_m, t, side="right") - 1

    # exclui RPTs (válidos e interrompidos)
    fora = np.zeros(len(t), dtype=bool)
    for t0, t1 in pd.concat([rpt, rej])[["t_start", "t_end"]].to_numpy():
        i0 = np.searchsorted(t, t0, side="left")
        i1 = np.searchsorted(t, t1, side="right")
        fora[i0:i1] = True

    dados = df[features].to_numpy(dtype=np.float32)
    valido = (missao >= 0) & ~fora & np.isfinite(dados).all(axis=1)

    # blocos contíguos: quebra em mudança de missão, de validade ou lacuna temporal
    quebra = np.ones(len(t), dtype=bool)
    quebra[1:] = ((missao[1:] != missao[:-1]) | (valido[1:] != valido[:-1])
                  | (np.diff(t) > a.max_gap))
    bloco = np.cumsum(quebra).astype(np.int64)
    bloco[~valido] = -1

    rpt_m = np.searchsorted(inicios_m, rpt.t_start.to_numpy()).astype(float)
    cap = rpt.capacity_mah.to_numpy()
    soh = cap / cap[0]
    eol = eol_crossing(rpt_m, soh, a.eol)
    n_m = len(inicios_m)
    return dict(nome=path.stem, dados=dados, missao=missao, bloco=bloco,
                n_missoes=n_m, eol=eol, vida=float(eol if eol is not None else n_m),
                rpt_missao=rpt_m, rpt_soh=soh)


def resumo(c: dict, a) -> dict:
    if c["eol"] is not None:
        status = f"cruza EOL {a.eol:.0%} na missão {c['eol']:.0f} (p)"
    else:
        status = f"NÃO cruza EOL {a.eol:.0%}; SOH no último RPT {c['rpt_soh'][-1]:.3f} (i)"
    tam = np.bincount(c["bloco"][c["bloco"] >= 0])
    tam = tam[tam > 0]
    print(f"  {c['nome']}: {c['n_missoes']} missões, {len(c['rpt_soh'])} RPTs, {status}; "
          f"corte {a.train_frac:.0%} = missão {a.train_frac * c['vida']:.0f}; "
          f"linhas por bloco (mediana) = {np.median(tam):.0f}")
    return dict(nome=c["nome"], n_missoes=c["n_missoes"], eol=c["eol"], vida=c["vida"],
                corte=a.train_frac * c["vida"], soh_final=float(c["rpt_soh"][-1]))


def janelas(c: dict, m_lo: float, m_hi: float, T: int, S: int, passo: int) -> np.ndarray:
    """Inícios de janela contidos num único bloco, com missão prevista em [m_lo, m_hi)."""
    L = T + S
    n = len(c["bloco"])
    if n < L:
        return np.empty(0, dtype=np.int64)
    s = np.arange(0, n - L + 1, passo, dtype=np.int64)
    b0, b1 = c["bloco"][s], c["bloco"][s + L - 1]
    m = c["missao"][s + T]
    ok = (b0 >= 0) & (b0 == b1) & (m >= m_lo) & (m < m_hi)
    return s[ok]


def empilha(cels, listas, scaler, T, S):
    """Concatena as linhas necessárias de cada célula e desloca os inícios."""
    blocos, inicios, off = [], [[] for _ in listas], 0
    for j, c in enumerate(cels):
        fim = max((int(l[j].max()) + T + S if len(l[j]) else 0) for l in listas)
        blocos.append(scaler.transform(c["dados"][:fim]).astype(np.float32))
        for k, l in enumerate(listas):
            inicios[k].append(l[j] + off)
        off += fim
    X = torch.from_numpy(np.concatenate(blocos))
    return X, [np.concatenate(i) for i in inicios]


# ============================================================ modelo e treino
def constroi_modelo(a, n_vars: int) -> nn.Module:
    # Ponto único para trocar o backbone mantendo o resto do pipeline idêntico.
    if a.backbone == "lstm":
        from modelo_lstm import LSTMForecaster
        return LSTMForecaster(n_vars, a.T, a.S, hidden=a.lstm_hidden,
                              layers=a.lstm_layers, dropout=a.lstm_dropout)
    from modelos import iTransformer
    return iTransformer(seq_len=a.T, pred_len=a.S, d_model=a.d_model,
                        n_heads=a.n_heads, n_blocks=a.n_blocks)


def lote(X: torch.Tensor, s: np.ndarray, T: int, S: int):
    idx = torch.from_numpy(s)[:, None] + torch.arange(T + S)
    w = X[idx]
    return w[:, :T], w[:, T:]


def itera_residuos(model, X, s, a, dev):
    """Gera (x, y, R = y - y_hat) em numpy, lote a lote, na ordem de s."""
    model.eval()
    with torch.no_grad():
        for i in range(0, len(s), a.eval_bs):
            xb, yb = lote(X, s[i:i + a.eval_bs], a.T, a.S)
            pred = model(xb.to(dev)).float().cpu()
            yield xb.numpy(), yb.numpy(), (yb - pred).numpy()


def treina(model, X, s_tr, s_va, a, dev, gen, caminho_pesos: Path):
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    perda = nn.MSELoss()
    melhor, historico = float("inf"), []
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(s_tr), generator=gen).numpy()
        tot, n = 0.0, 0
        for i in range(0, len(perm), a.bs):
            xb, yb = lote(X, s_tr[perm[i:i + a.bs]], a.T, a.S)
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = perda(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sq, cnt = 0.0, 0
        for _, _, R in itera_residuos(model, X, s_va, a, dev):
            sq += float((R.astype(np.float64) ** 2).sum())
            cnt += R.size
        val = sq / cnt
        sched.step()
        novo = val < melhor
        if novo:
            melhor = val
            torch.save(model.state_dict(), caminho_pesos)
        historico.append(dict(epoca=ep, treino=tot / n, val=val))
        print(f"  época {ep:2d} | treino {tot / n:.4f} | val {val:.4f} | "
              f"{time.time() - t0:.0f}s{'  *' if novo else ''}")
    model.load_state_dict(torch.load(caminho_pesos, map_location=dev))
    return melhor, historico


# ============================================================ avaliação (numpy)
def estatisticas_ref(iterador):
    Rs = [R for _, _, R in iterador]
    R = np.concatenate(Rs).astype(np.float64)
    mu = R.mean(axis=0)
    sd = R.std(axis=0)
    sd = np.where(sd < 1e-8, 1e-8, sd)
    escores = (((R - mu) / sd) ** 2).sum(axis=(1, 2))
    return mu, sd, escores


def avalia_teste(iterador, fase: np.ndarray, mu, sd):
    """Acumula métricas de previsão por fase (modelo, persistência, média) e escores."""
    metodos = ["modelo", "persistencia", "media"]
    acc = {m: {f: [0.0, 0.0, 0] for f in range(3)} for m in metodos}
    escores, i = [], 0
    for xb, yb, R in iterador:
        fb = fase[i:i + len(R)]
        i += len(R)
        res = {"modelo": R,
               "persistencia": yb - xb[:, -1:, :],
               "media": yb}  # espaço padronizado: média do treino = 0
        for m, E in res.items():
            E = E.astype(np.float64)
            for f in range(3):
                sel = fb == f
                if sel.any():
                    acc[m][f][0] += float((E[sel] ** 2).sum())
                    acc[m][f][1] += float(np.abs(E[sel]).sum())
                    acc[m][f][2] += E[sel].size
        escores.append((((R.astype(np.float64) - mu) / sd) ** 2).sum(axis=(1, 2)))
    metricas = {}
    for m in metodos:
        for f in range(3):
            se, ae, n = acc[m][f]
            if n:
                metricas.setdefault(FASES[f], {})[m] = dict(
                    MSE=se / n, MAE=ae / n, RMSE=float(np.sqrt(se / n)))
    return metricas, np.concatenate(escores)


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    r = pd.Series(np.concatenate([neg, pos])).rank().to_numpy()
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman_soh(c: dict, m_janela: np.ndarray, escores: np.ndarray, min_janelas=10):
    rm, rs = c["rpt_missao"], c["rpt_soh"]
    sup = np.append(rm[1:], rm[-1] + 50)
    med, sohs = [], []
    for m0, m1, sv in zip(rm, sup, rs):
        sel = (m_janela >= m0) & (m_janela < m1)
        if sel.sum() >= min_janelas:
            med.append(np.median(escores[sel]))
            sohs.append(sv)
    if len(med) < 3:
        return float("nan"), len(med)
    ra = pd.Series(med).rank().to_numpy()
    rb = pd.Series(sohs).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]), len(med)


def conformidade(x: float, k: float, c: float) -> float:
    return float(1.0 / (1.0 + np.exp(k * (x - c))))


# ============================================================ main
def args_cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("resultados"))
    ap.add_argument("--features", nargs="+", default=FEATURES_PADRAO)
    # desenho
    ap.add_argument("--eol", type=float, default=0.80)
    ap.add_argument("--train-frac", type=float, default=0.20,
                    help="fração inicial da vida útil tratada como regime nominal")
    ap.add_argument("--split", type=float, nargs=2, default=[0.70, 0.15],
                    help="frações treino e val dentro do início de vida (ref = resto)")
    ap.add_argument("--q", type=float, default=99.0, help="percentil do limiar tau")
    ap.add_argument("--k", type=float, default=219.7)
    ap.add_argument("--c", type=float, default=0.03)
    # modelo e treino
    ap.add_argument("--backbone", choices=["itransformer", "lstm"], default="itransformer")
    ap.add_argument("--lstm-hidden", type=int, default=176)
    ap.add_argument("--lstm-layers", type=int, default=2)
    ap.add_argument("--lstm-dropout", type=float, default=0.0)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--S", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--eval-bs", type=int, default=1024)
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--eval-stride", type=int, default=4)
    # detecção de RPT e missões (mesmos padrões de soh_eol.py)
    ap.add_argument("--rpt-i-min", type=float, default=450.0)
    ap.add_argument("--rpt-i-max", type=float, default=750.0)
    ap.add_argument("--rpt-min-hours", type=float, default=2.0)
    ap.add_argument("--rpt-v-end", type=float, default=2.6)
    ap.add_argument("--mission-i-min", type=float, default=1000.0)
    ap.add_argument("--mission-min-minutes", type=float, default=5.0)
    ap.add_argument("--mission-max-hours", type=float, default=1.5)
    ap.add_argument("--mission-merge", type=float, default=1800.0)
    ap.add_argument("--min-current", type=float, default=20.0)
    ap.add_argument("--max-gap", type=float, default=300.0)
    return ap.parse_args()


def main():
    a = args_cli()
    if a.test.stem in {p.stem for p in a.train}:
        raise SystemExit("A célula de teste não pode estar no treino.")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.backbone}_{a.test.stem}_train-{'+'.join(p.stem for p in a.train)}_seed{a.seed}"

    print("Preparando células...")
    tr = [prepara_celula(p, a.features, a) for p in a.train]
    te = prepara_celula(a.test, a.features, a)
    info = {"treino": [resumo(c, a) for c in tr]}
    print("  --- teste ---")
    info["teste"] = resumo(te, a)

    # ---- splits das células de treino (por missão, dentro do início de vida)
    f_tr, f_va = a.split
    s_tr, s_va, s_ref, lim_tr = [], [], [], []
    for c in tr:
        corte = a.train_frac * c["vida"]
        m1, m2 = f_tr * corte, (f_tr + f_va) * corte
        s_tr.append(janelas(c, 0, m1, a.T, a.S, a.train_stride))
        s_va.append(janelas(c, m1, m2, a.T, a.S, a.eval_stride))
        s_ref.append(janelas(c, m2, corte, a.T, a.S, a.eval_stride))
        lim_tr.append(m1)
    for nome, l in [("treino", s_tr), ("val", s_va), ("ref", s_ref)]:
        tot = sum(len(x) for x in l)
        print(f"  janelas {nome}: {tot}")
        if tot == 0:
            raise SystemExit(f"Nenhuma janela de {nome}: confira --train-frac, --T/--S e o tamanho dos blocos.")

    scaler = StandardScaler().fit(np.concatenate(
        [c["dados"][(c["bloco"] >= 0) & (c["missao"] < m)] for c, m in zip(tr, lim_tr)]))
    joblib.dump(scaler, a.out_dir / f"{tag}_scaler.pkl")

    X, (s_tr, s_va, s_ref) = empilha(tr, [s_tr, s_va, s_ref], scaler, a.T, a.S)

    # ---- modelo
    model = constroi_modelo(a, len(a.features)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\nTreinando {a.backbone} em {dev} ({n_par} parâmetros)...")
    pesos = a.out_dir / f"{tag}_melhor.pth"
    t0 = time.time()
    melhor_val, historico = treina(model, X, s_tr, s_va, a, dev, gen, pesos)
    t_treino = time.time() - t0

    # ---- referência: mu, sigma e tau
    mu, sd, esc_ref = estatisticas_ref(itera_residuos(model, X, s_ref, a, dev))
    tau = float(np.percentile(esc_ref, a.q))
    del X

    # ---- teste: vida inteira
    s_te = janelas(te, 0, np.inf, a.T, a.S, a.eval_stride)
    m_te = te["missao"][s_te + a.T]
    corte_te = a.train_frac * te["vida"]
    fase = np.ones(len(s_te), dtype=np.int8)
    fase[m_te < corte_te] = 0
    if te["eol"] is not None:
        fase[m_te >= te["eol"]] = 2
    Xte = torch.from_numpy(scaler.transform(te["dados"]).astype(np.float32))
    metricas, escores = avalia_teste(itera_residuos(model, Xte, s_te, a, dev), fase, mu, sd)

    deteccao = {}
    for f, nome in enumerate(FASES):
        sel = fase == f
        if sel.any():
            x = float((escores[sel] > tau).mean())
            deteccao[nome] = dict(janelas=int(sel.sum()), frac_anomala=x,
                                  conformidade=conformidade(x, a.k, a.c))
    if (fase == 2).any() and (fase == 0).any():
        deteccao["auroc_inicio_vs_degradado"] = auroc(escores[fase == 0], escores[fase == 2])
    rho, n_blocos = spearman_soh(te, m_te, escores)
    deteccao["spearman_escore_soh"] = rho
    deteccao["spearman_n_blocos"] = n_blocos

    # ---- relatório
    print(f"\nTempo de treino: {t_treino / 60:.1f} min | melhor val MSE: {melhor_val:.4f} | tau (P{a.q:g}): {tau:.2f}")
    print("\nPrevisão (espaço padronizado)")
    for nome, d in metricas.items():
        print(f"  [{nome}]")
        for m, v in d.items():
            print(f"    {m:13s} MSE {v['MSE']:.4f}  MAE {v['MAE']:.4f}  RMSE {v['RMSE']:.4f}")
    print("\nDetecção")
    for nome in FASES:
        if nome in deteccao:
            d = deteccao[nome]
            print(f"  {nome:10s} janelas {d['janelas']:8d} | fração anômala {d['frac_anomala']:.4f} | "
                  f"conformidade {d['conformidade']:.3f}")
    if "auroc_inicio_vs_degradado" in deteccao:
        print(f"  AUROC início vs degradado: {deteccao['auroc_inicio_vs_degradado']:.3f}")
    print(f"  Spearman (mediana do escore por bloco de RPT x SOH): {rho:.3f} (n = {n_blocos})")

    resultado = dict(tag=tag, backbone=a.backbone, args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                     celulas=info, parametros=n_par, tempo_treino_s=t_treino, melhor_val=melhor_val,
                     historico=historico, tau=tau, previsao=metricas, deteccao=deteccao)
    resultado["args"]["train"] = [str(p) for p in a.train]
    (a.out_dir / f"{tag}.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    np.savez_compressed(a.out_dir / f"{tag}_janelas.npz", missao=m_te, fase=fase, escore=escores,
                        tau=tau, rpt_missao=te["rpt_missao"], rpt_soh=te["rpt_soh"])
    print(f"\nResultados em {a.out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
