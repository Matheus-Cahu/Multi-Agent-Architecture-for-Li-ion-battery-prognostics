#!/usr/bin/env python3
"""
Gera a figura de escore ao longo da vida da célula, a partir do .npz salvo por
treino_eol.py / experimentos.py.

Painel (a): escore por janela (escala log) contra o número da missão, com o
limiar tau, a fase de início de vida sombreada e o EOL marcado.
Painel (b): fração de janelas sinalizadas por bloco de RPT (eixo esquerdo) e
SOH medido no RPT correspondente (eixo direito), com a linha c = 3%.

Imprime também o SOH em que a fração sinalizada cruza c, que é o número
citável no texto ("the method flags departure from nominal behavior at
SOH ~ X%").

Uso:
    python figura_escore.py resultados/itransformer_VAH17_..._janelas.npz
    python figura_escore.py ... --out fig2.pdf --c 0.03
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np


def blocos_rpt(missao, escore, tau, rpt_missao, rpt_soh, passo=50, min_janelas=10):
    """Fração sinalizada e mediana do escore em cada intervalo entre RPTs."""
    sup = np.append(rpt_missao[1:], rpt_missao[-1] + passo)
    m, frac, soh, med = [], [], [], []
    for m0, m1, sv in zip(rpt_missao, sup, rpt_soh):
        sel = (missao >= m0) & (missao < m1)
        if sel.sum() >= min_janelas:
            m.append(m0)
            frac.append(float((escore[sel] > tau).mean()))
            med.append(float(np.median(escore[sel])))
            soh.append(float(sv))
    return (np.array(m), np.array(frac), np.array(soh), np.array(med))


def cruzamento(x, y, alvo):
    """Primeiro ponto em que y cruza alvo (subindo), interpolado; devolve (x, índice)."""
    acima = np.flatnonzero(y >= alvo)
    if acima.size == 0:
        return None, None
    k = acima[0]
    if k == 0:
        return float(x[0]), 0
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    if y1 == y0:
        return float(x1), k
    return float(x0 + (alvo - y0) / (y1 - y0) * (x1 - x0)), k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--c", type=float, default=0.03, help="limiar da sigmoide")
    ap.add_argument("--passo", type=int, default=50, help="missões entre RPTs")
    ap.add_argument("--max-pontos", type=int, default=20000,
                    help="janelas desenhadas no painel (a)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--largura", type=float, default=3.5, help="polegadas (uma coluna IEEE)")
    ap.add_argument("--altura", type=float, default=4.3)
    ap.add_argument("--sub-a", default="(a) Per-window anomaly score",
                    help="subtítulo abaixo do painel (a)")
    ap.add_argument("--sub-b", default="(b) Flagged windows and SOH per RPT block",
                    help="subtítulo abaixo do painel (b)")
    a = ap.parse_args()

    if a.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(a.npz)
    missao, fase, escore, tau = d["missao"], d["fase"], d["escore"], float(d["tau"])
    rpt_m, rpt_soh = d["rpt_missao"], d["rpt_soh"]

    eol = float(missao[fase == 2].min()) if (fase == 2).any() else None
    fim_inicio = float(missao[fase == 0].max()) if (fase == 0).any() else None
    m_b, frac, soh_b, _ = blocos_rpt(missao, escore, tau, rpt_m, rpt_soh, a.passo)

    m_cruz, k = cruzamento(m_b, frac, a.c)
    soh_cruz = float(np.interp(m_cruz, m_b, soh_b)) if m_cruz is not None else None

    print(f"Arquivo: {a.npz.name}")
    print(f"  janelas: {len(escore)} | tau: {tau:.1f} | blocos de RPT: {len(m_b)}")
    print(f"  fim do início de vida: missão {fim_inicio:.0f}" if fim_inicio else "  sem fase inicial")
    print(f"  EOL: missão {eol:.0f}" if eol else "  célula não cruza o EOL")
    for f, nome in enumerate(["inicio", "meio", "degradado"]):
        sel = fase == f
        if sel.any():
            print(f"  {nome:10s} {sel.sum():7d} janelas | fração sinalizada {(escore[sel] > tau).mean():.4f}")
    if m_cruz is not None:
        print(f"  fração sinalizada cruza c={a.c:.0%} na missão {m_cruz:.0f}, SOH ~ {soh_cruz:.3f}")
    else:
        print(f"  fração sinalizada nunca cruza c={a.c:.0%}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(a.largura, a.altura), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.15, 1]))

    # (a) escore por janela
    passo = max(1, len(escore) // a.max_pontos)
    ms, es = missao[::passo], escore[::passo]
    ax1.scatter(ms, np.maximum(es, 1e-3), s=1, alpha=0.25, color="tab:blue",
                linewidths=0, rasterized=True)
    ax1.axhline(tau, color="tab:red", ls="--", lw=0.9, label=r"$\tau$")
    if fim_inicio:
        ax1.axvspan(0, fim_inicio, color="0.85", zorder=0, label="early life (train phase)")
    if eol:
        ax1.axvline(eol, color="k", ls=":", lw=0.9, label="EOL (SOH 80%)")
    ax1.set_yscale("log")
    ax1.set_ylabel("anomaly score")
    ax1.legend(fontsize=6, loc="lower right", framealpha=0.9)

    # (b) fração sinalizada por bloco + SOH
    ax2.plot(m_b, frac * 100, "o-", ms=3, lw=1, color="tab:blue", label="flagged windows")
    ax2.axhline(a.c * 100, color="tab:red", ls="--", lw=0.9, label=f"c = {a.c:.0%}")
    if eol:
        ax2.axvline(eol, color="k", ls=":", lw=0.9)
    ax2.set_ylabel("flagged (%)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_xlabel("mission")

    ax3 = ax2.twinx()
    ax3.plot(m_b, soh_b * 100, "s-", ms=3, lw=1, color="tab:green")
    ax3.set_ylabel("SOH (%)", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")
    if m_cruz is not None:
        ax2.plot([m_cruz], [a.c * 100], marker="v", color="tab:red", ms=5, zorder=5)
        ax2.annotate(f"SOH {soh_cruz:.0%}", (m_cruz, a.c * 100), fontsize=6,
                     textcoords="offset points", xytext=(4, 6), color="tab:red")
    ax2.legend(fontsize=6, loc="upper left", framealpha=0.9)

    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    # subtítulos abaixo de cada painel
    ax1.text(0.5, -0.07, a.sub_a, transform=ax1.transAxes, ha="center", va="top", fontsize=8)
    ax2.text(0.5, -0.30, a.sub_b, transform=ax2.transAxes, ha="center", va="top", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(hspace=0.28)

    out = a.out or a.npz.with_name(a.npz.stem.replace("_janelas", "") + "_fig_escore.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Figura salva em {out}")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
