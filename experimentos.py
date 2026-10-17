#!/usr/bin/env python3
"""
Roda o experimento completo: para cada fold configurado abaixo, treina os dois
motores (iTransformer e LSTM), com várias seeds, e avalia cada um nas células de
teste do fold. Ao final, agrega tudo em média, desvio-padrão e IC 95%.

Economia: as células são lidas e preparadas uma única vez (cache), e cada modelo
treinado é avaliado em todas as células de teste do fold.

ATENÇÃO ao desenho: cada entrada de FOLDS define um conjunto de treino e um de
teste. Leave-one-cell-out é obtido listando um fold por rotação, como no padrão
abaixo. Se você puser células de perfis diferentes num mesmo treino, o tau passa
a ser calibrado sobre uma mistura de perfis e os resultados ficam difíceis de
interpretar.

Uso:
    python experimentos.py            # roda tudo que está configurado
    python experimentos.py --so-agregar   # só reagrega os JSONs já existentes

Requer treino_eol.py, soh_eol.py, modelo_lstm.py e modelos.py no mesmo diretório.
"""
import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import treino_eol as te

# ============================================================ CONFIGURAÇÃO
DATA = Path("data")

TODAS = ["VAH01", "VAH05", "VAH09", "VAH10", "VAH11", "VAH12",
         "VAH13", "VAH17", "VAH25", "VAH26", "VAH27", "VAH28"]
TESTE = ["VAH17", "VAH09", "VAH13", "VAH10", "VAH11", "VAH12"]

FOLDS = [{"nome": f"multi-{c}", "treino": [x for x in TODAS if x != c], "teste": [c]}
         for c in TESTE]

BACKBONES = ["itransformer", "lstm"]
SEEDS = [0, 1, 2, 3, 4]
SAIDA = Path("resultados")

# Parâmetros comuns a todas as runs (mesmos padrões de treino_eol.py).
PARAMS = dict(
    features=te.FEATURES_PADRAO,
    eol=0.80, train_frac=0.20, split=[0.70, 0.15], q=99.0, k=219.7, c=0.03,
    T=96, S=24, d_model=128, n_heads=4, n_blocks=2,
    lstm_hidden=176, lstm_layers=2, lstm_dropout=0.0,
    epochs=15, lr=1e-3, bs=64, eval_bs=1024, train_stride=1, eval_stride=4,
    rpt_i_min=450.0, rpt_i_max=750.0, rpt_min_hours=2.0, rpt_v_end=2.6,
    mission_i_min=1000.0, mission_min_minutes=5.0, mission_max_hours=1.5,
    mission_merge=1800.0, min_current=20.0, max_gap=300.0,
)

# Métricas que entram na agregação final.
METRICAS = ["mse_inicio", "mae_inicio", "rmse_inicio", "mse_degradado",
            "far_inicio", "frac_meio", "frac_degradado", "conf_inicio",
            "auroc", "spearman", "tau", "tempo_treino_min"]

# t de Student bicaudal, 95%, por graus de liberdade (n-1).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       25: 2.060, 30: 2.042}


def t_critico(n: int) -> float:
    if n < 2:
        return float("nan")
    gl = n - 1
    if gl in T95:
        return T95[gl]
    return T95[min(T95, key=lambda g: abs(g - gl))] if gl < 30 else 1.96


# ============================================================ execução
CACHE = {}


def celula(nome: str, a) -> dict:
    if nome not in CACHE:
        print(f"  lendo {nome}...", flush=True)
        CACHE[nome] = te.prepara_celula(DATA / f"{nome}.csv", a.features, a)
    return CACHE[nome]


def prepara_fold(fold, a):
    """Splits, scaler e tensor de treino: dependem do fold, não do backbone/seed."""
    tr = [celula(n, a) for n in fold["treino"]]
    print("  --- treino ---")
    info_tr = [te.resumo(c, a) for c in tr]

    f_tr, f_va = a.split
    s_tr, s_va, s_ref, lim_tr = [], [], [], []
    for c in tr:
        corte = a.train_frac * c["vida"]
        m1, m2 = f_tr * corte, (f_tr + f_va) * corte
        s_tr.append(te.janelas(c, 0, m1, a.T, a.S, a.train_stride))
        s_va.append(te.janelas(c, m1, m2, a.T, a.S, a.eval_stride))
        s_ref.append(te.janelas(c, m2, corte, a.T, a.S, a.eval_stride))
        lim_tr.append(m1)
    for nome, l in [("treino", s_tr), ("val", s_va), ("ref", s_ref)]:
        tot = sum(len(x) for x in l)
        print(f"  janelas {nome}: {tot}")
        if tot == 0:
            raise SystemExit(f"Fold {fold['nome']}: nenhuma janela de {nome}.")

    scaler = StandardScaler().fit(np.concatenate(
        [c["dados"][(c["bloco"] >= 0) & (c["missao"] < m)] for c, m in zip(tr, lim_tr)]))
    X, inicios = te.empilha(tr, [s_tr, s_va, s_ref], scaler, a.T, a.S)
    return dict(info=info_tr, scaler=scaler, X=X, s_tr=inicios[0],
                s_va=inicios[1], s_ref=inicios[2])


def avalia_celula(model, nome_teste, mu, sd, tau, scaler, a, dev):
    c = celula(nome_teste, a)
    s = te.janelas(c, 0, np.inf, a.T, a.S, a.eval_stride)
    m = c["missao"][s + a.T]
    corte = a.train_frac * c["vida"]
    fase = np.ones(len(s), dtype=np.int8)
    fase[m < corte] = 0
    if c["eol"] is not None:
        fase[m >= c["eol"]] = 2
    X = torch.from_numpy(scaler.transform(c["dados"]).astype(np.float32))
    metricas, escores = te.avalia_teste(te.itera_residuos(model, X, s, a, dev), fase, mu, sd)

    det = {}
    for f, nome in enumerate(te.FASES):
        sel = fase == f
        if sel.any():
            x = float((escores[sel] > tau).mean())
            det[nome] = dict(janelas=int(sel.sum()), frac_anomala=x,
                             conformidade=te.conformidade(x, a.k, a.c))
    if (fase == 0).any() and (fase == 2).any():
        det["auroc_inicio_vs_degradado"] = te.auroc(escores[fase == 0], escores[fase == 2])
    rho, n_blocos = te.spearman_soh(c, m, escores)
    det["spearman_escore_soh"] = rho
    det["spearman_n_blocos"] = n_blocos
    return c, metricas, det, m, fase, escores


def linha_resumo(previsao, deteccao, extras) -> dict:
    def g(d, *ks):
        for k in ks:
            if d is None or k not in d:
                return float("nan")
            d = d[k]
        return d
    return dict(
        mse_inicio=g(previsao, "inicio", "modelo", "MSE"),
        mae_inicio=g(previsao, "inicio", "modelo", "MAE"),
        rmse_inicio=g(previsao, "inicio", "modelo", "RMSE"),
        mse_degradado=g(previsao, "degradado", "modelo", "MSE"),
        mse_persistencia_inicio=g(previsao, "inicio", "persistencia", "MSE"),
        far_inicio=g(deteccao, "inicio", "frac_anomala"),
        frac_meio=g(deteccao, "meio", "frac_anomala"),
        frac_degradado=g(deteccao, "degradado", "frac_anomala"),
        conf_inicio=g(deteccao, "inicio", "conformidade"),
        auroc=deteccao.get("auroc_inicio_vs_degradado", float("nan")),
        spearman=deteccao.get("spearman_escore_soh", float("nan")),
        **extras)


def roda_tudo(a) -> pd.DataFrame:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas, t_inicio = [], time.time()
    total = sum(len(BACKBONES) * len(SEEDS) for _ in FOLDS)
    feito = 0

    for fold in FOLDS:
        conflito = set(fold["treino"]) & set(fold["teste"])
        if conflito:
            raise SystemExit(f"Fold {fold['nome']}: {sorted(conflito)} está no treino e no teste.")
        print(f"\n{'=' * 70}\nFOLD {fold['nome']}: treino {fold['treino']} | teste {fold['teste']}")
        F = prepara_fold(fold, a)
        joblib.dump(F["scaler"], SAIDA / f"{fold['nome']}_scaler.pkl")

        for bb in BACKBONES:
            for seed in SEEDS:
                a.backbone, a.seed = bb, seed
                tag = f"{bb}_{fold['nome']}_seed{seed}"
                print(f"\n--- {tag} ({feito + 1}/{total}) ---")
                torch.manual_seed(seed)
                np.random.seed(seed)
                gen = torch.Generator().manual_seed(seed)

                model = te.constroi_modelo(a, len(a.features)).to(dev)
                n_par = sum(p.numel() for p in model.parameters())
                print(f"Treinando {bb} em {dev} ({n_par} parâmetros)...")
                t0 = time.time()
                melhor_val, historico = te.treina(model, F["X"], F["s_tr"], F["s_va"],
                                                  a, dev, gen, SAIDA / f"{tag}_melhor.pth")
                t_treino = time.time() - t0

                mu, sd, esc_ref = te.estatisticas_ref(
                    te.itera_residuos(model, F["X"], F["s_ref"], a, dev))
                tau = float(np.percentile(esc_ref, a.q))

                for nome_te in fold["teste"]:
                    c, previsao, det, m, fase, escores = avalia_celula(
                        model, nome_te, mu, sd, tau, F["scaler"], a, dev)
                    linha = linha_resumo(previsao, det, dict(
                        fold=fold["nome"], backbone=bb, seed=seed, teste=nome_te,
                        treino="+".join(fold["treino"]), parametros=n_par,
                        tempo_treino_min=t_treino / 60, melhor_val=melhor_val, tau=tau,
                        eol=c["eol"], spearman_n=det["spearman_n_blocos"]))
                    linhas.append(linha)
                    print(f"  [{nome_te}] MSE(início) {linha['mse_inicio']:.4f} | "
                          f"FAR {linha['far_inicio']:.4f} | AUROC {linha['auroc']:.3f} | "
                          f"Spearman {linha['spearman']:.3f} | {t_treino / 60:.1f} min")

                    (SAIDA / f"{tag}_{nome_te}.json").write_text(json.dumps(dict(
                        tag=tag, backbone=bb, fold=fold["nome"], seed=seed, teste=nome_te,
                        treino=fold["treino"], parametros=n_par, tempo_treino_s=t_treino,
                        melhor_val=melhor_val, historico=historico, tau=tau,
                        celulas=dict(treino=F["info"], teste=te.resumo(c, a)),
                        previsao=previsao, deteccao=det,
                        params={k: v for k, v in vars(a).items()},
                    ), indent=2, ensure_ascii=False, default=str))
                    np.savez_compressed(SAIDA / f"{tag}_{nome_te}_janelas.npz",
                                        missao=m, fase=fase, escore=escores, tau=tau,
                                        rpt_missao=c["rpt_missao"], rpt_soh=c["rpt_soh"])
                feito += 1
                resta = (time.time() - t_inicio) / feito * (total - feito) / 60
                print(f"  ({feito}/{total} concluídas, ~{resta:.0f} min restantes)")
        del F
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "runs.csv", index=False)
    return df


# ============================================================ agregação
def carrega_runs() -> pd.DataFrame:
    linhas = []
    for p in sorted(SAIDA.glob("*.json")):
        d = json.loads(p.read_text())
        if "deteccao" not in d or "fold" not in d:
            continue  # ignora JSONs de outro formato
        linhas.append(linha_resumo(d["previsao"], d["deteccao"], dict(
            fold=d["fold"], backbone=d["backbone"], seed=d["seed"], teste=d["teste"],
            treino="+".join(d["treino"]), parametros=d["parametros"],
            tempo_treino_min=d["tempo_treino_s"] / 60, melhor_val=d["melhor_val"],
            tau=d["tau"], eol=None, spearman_n=d["deteccao"].get("spearman_n_blocos"))))
    return pd.DataFrame(linhas)


def agrega(df: pd.DataFrame, por) -> pd.DataFrame:
    """Média, desvio (ddof=1), n e IC 95% de cada métrica, agrupando por 'por'."""
    saida = []
    for chave, g in df.groupby(por, dropna=False):
        chave = chave if isinstance(chave, tuple) else (chave,)
        linha = dict(zip(por, chave))
        linha["n"] = len(g)
        for m in METRICAS:
            if m not in g:
                continue
            v = g[m].dropna().to_numpy(dtype=float)
            if len(v) == 0:
                continue
            media, dp = v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))
            ic = t_critico(len(v)) * dp / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            linha[f"{m}_media"], linha[f"{m}_dp"], linha[f"{m}_ic95"] = media, dp, ic
        saida.append(linha)
    return pd.DataFrame(saida)


def compara_backbones(df: pd.DataFrame) -> pd.DataFrame:
    """Diferença pareada (itransformer - lstm) por fold/teste/seed."""
    if df.backbone.nunique() < 2:
        return pd.DataFrame()
    chave = ["fold", "teste", "seed"]
    a_ = df[df.backbone == "itransformer"].set_index(chave)
    b_ = df[df.backbone == "lstm"].set_index(chave)
    comum = a_.index.intersection(b_.index)
    if len(comum) == 0:
        return pd.DataFrame()
    saida = []
    for m in METRICAS:
        if m not in df:
            continue
        d = (a_.loc[comum, m] - b_.loc[comum, m]).dropna().to_numpy(dtype=float)
        if len(d) == 0:
            continue
        dp = d.std(ddof=1) if len(d) > 1 else float("nan")
        ic = t_critico(len(d)) * dp / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        saida.append(dict(metrica=m, n_pares=len(d), dif_media=d.mean(), dp=dp, ic95=ic,
                          zero_no_ic=(abs(d.mean()) < ic) if np.isfinite(ic) else None))
    return pd.DataFrame(saida)


def formata(df: pd.DataFrame, chaves, metricas) -> str:
    cols = list(chaves) + ["n"]
    out = df[cols].copy()
    for m in metricas:
        if f"{m}_media" in df:
            out[m] = [f"{mu:.4f} ± {dp:.4f} (IC {ic:.4f})" if np.isfinite(dp) else f"{mu:.4f}"
                      for mu, dp, ic in zip(df[f"{m}_media"], df[f"{m}_dp"], df[f"{m}_ic95"])]
    return out.to_string(index=False)


def relatorio(df: pd.DataFrame):
    principais = ["mse_inicio", "far_inicio", "auroc", "spearman", "tempo_treino_min"]
    print(f"\n{'=' * 70}\nAGREGADO POR BACKBONE, FOLD E CÉLULA DE TESTE (sobre as seeds)")
    por_cel = agrega(df, ["backbone", "fold", "teste"])
    print(formata(por_cel, ["backbone", "fold", "teste"], principais))

    print(f"\n{'=' * 70}\nAGREGADO POR BACKBONE (todas as runs)")
    print("Aviso: seeds e folds não são unidades independentes; use como resumo.")
    por_bb = agrega(df, ["backbone"])
    print(formata(por_bb, ["backbone"], METRICAS))

    cmp = compara_backbones(df)
    if not cmp.empty:
        print(f"\n{'=' * 70}\nDIFERENÇA PAREADA (itransformer - lstm), mesma seed/fold/célula")
        print("zero_no_ic = True significa que o IC 95% cruza zero (diferença não significativa).")
        print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    por_cel.to_csv(SAIDA / "agregado_por_celula.csv", index=False)
    por_bb.to_csv(SAIDA / "agregado_por_backbone.csv", index=False)
    if not cmp.empty:
        cmp.to_csv(SAIDA / "comparacao_pareada.csv", index=False)
    print(f"\nCSVs em {SAIDA}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so-agregar", action="store_true",
                    help="não treina; só reagrega os JSONs já em resultados/")
    opc = ap.parse_args()

    if opc.so_agregar:
        df = carrega_runs()
        if df.empty:
            raise SystemExit(f"Nenhum JSON deste formato em {SAIDA}/.")
    else:
        a = Namespace(**PARAMS, backbone=BACKBONES[0], seed=SEEDS[0])
        df = roda_tudo(a)
    relatorio(df)


if __name__ == "__main__":
    main()
