#!/usr/bin/env python3
"""
Extrai o SOH de uma célula do eVTOL Battery Dataset (Bills et al., 2023)
a partir dos testes de referência (RPT), plota a curva e reporta o ciclo
em que o SOH cruza o limiar de EOL.

Identificação do RPT: descarga contínua lenta (~C/5 = 600 mA) por várias horas,
terminando próxima de 2,5 V. Os pulsos de C/50 da medição de resistência
(20% e 60% DOD) mantêm a corrente negativa, então não quebram o segmento.
A capacidade é obtida integrando a corrente no tempo, sem depender de como
QDischarge_mA_h é zerado entre ciclos.

A coluna cycleNumber NÃO é um contador global (reinicia ao longo do arquivo
concatenado). Por isso o eixo de ciclos é reconstruído contando as descargas
de missão: segmentos de alta corrente com duração de minutos. Segmentos de
missão separados por menos de --mission-merge segundos (ex.: interrupção do
testador no meio da missão) são contados como uma única missão.

Uso:
    python soh_eol.py VAH01.csv
    python soh_eol.py VAH01.csv --eol 0.8 --out vah01_soh.png --no-show
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


def load(path: Path) -> pd.DataFrame:
    cols = ["time_s", "Ecell_V", "I_mA"]
    df = pd.read_csv(path, usecols=cols,
                     dtype={"time_s": "float64", "Ecell_V": "float32",
                            "I_mA": "float32"})
    return df.dropna(subset=["time_s", "I_mA"]).reset_index(drop=True)


def discharge_segments(df: pd.DataFrame, min_current_ma: float,
                       max_gap_s: float) -> pd.DataFrame:
    """Agrupa amostras contíguas de descarga (I < -min_current) em segmentos."""
    t = df["time_s"].to_numpy()
    i = df["I_mA"].to_numpy()

    neg = i < -min_current_ma
    dt_prev = np.diff(t, prepend=t[0])
    starts = neg & (~np.roll(neg, 1) | (dt_prev > max_gap_s))
    starts[0] = neg[0]
    seg = np.cumsum(starts)
    seg[~neg] = 0  # 0 = fora de descarga

    # Integração (Riemann à esquerda) só entre amostras do mesmo segmento
    dt_next = np.append(np.diff(t), 0.0)
    same_next = np.append(seg[1:] == seg[:-1], False)
    valid = same_next & (dt_next <= max_gap_s) & (seg > 0)
    charge_mas = np.where(valid, np.abs(i) * dt_next, 0.0)

    d = pd.DataFrame({"seg": seg, "t": t, "absI": np.abs(i),
                      "V": df["Ecell_V"].to_numpy(),
                      "q_mas": charge_mas})
    d = d[d.seg > 0]
    g = d.groupby("seg").agg(
        t_start=("t", "min"),
        t_end=("t", "max"),
        i_median_ma=("absI", "median"),
        v_min=("V", "min"),
        q_mas=("q_mas", "sum"),
    )
    g["duration_h"] = (g.t_end - g.t_start) / 3600.0
    g["capacity_mah"] = g.q_mas / 3600.0
    return g.drop(columns=["q_mas"])


def find_rpts(seg: pd.DataFrame, i_min: float, i_max: float,
              min_hours: float, v_end: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = seg[seg.i_median_ma.between(i_min, i_max) & (seg.duration_h >= min_hours)]
    ok = cand[cand.v_min <= v_end].copy()
    rejected = cand[cand.v_min > v_end].copy()  # ex.: RPT interrompido antes de 2,5 V
    return ok.sort_values("t_start"), rejected.sort_values("t_start")


def find_missions(seg: pd.DataFrame, i_min: float, min_min: float,
                  max_hours: float, merge_s: float) -> np.ndarray:
    """Retorna os instantes de início de cada missão (após fundir fragmentos)."""
    m = seg[(seg.i_median_ma > i_min) & (seg.duration_h * 60 >= min_min)
            & (seg.duration_h <= max_hours)].sort_values("t_start")
    if m.empty:
        return np.array([])
    starts, ends = [m.t_start.iloc[0]], [m.t_end.iloc[0]]
    for ts, te in zip(m.t_start.iloc[1:], m.t_end.iloc[1:]):
        if ts - ends[-1] < merge_s:
            ends[-1] = te  # fragmento da mesma missão
        else:
            starts.append(ts)
            ends.append(te)
    return np.asarray(starts)


def eol_crossing(cycles: np.ndarray, soh: np.ndarray, eol: float):
    """Primeiro cruzamento de SOH abaixo de eol, com interpolação linear."""
    below = np.flatnonzero(soh < eol)
    if below.size == 0:
        return None
    k = below[0]
    if k == 0:
        return float(cycles[0])
    c0, c1, s0, s1 = cycles[k - 1], cycles[k], soh[k - 1], soh[k]
    return float(c0 + (s0 - eol) / (s0 - s1) * (c1 - c0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="arquivo VAHxx.csv")
    ap.add_argument("--eol", type=float, default=0.80, help="limiar de SOH (padrão 0.80)")
    ap.add_argument("--ref", choices=["first", "nominal"], default="first",
                    help="normalizar pelo 1º RPT ou pela capacidade nominal")
    ap.add_argument("--nominal-mah", type=float, default=3000.0)
    ap.add_argument("--rpt-i-min", type=float, default=450.0, help="mA")
    ap.add_argument("--rpt-i-max", type=float, default=750.0, help="mA")
    ap.add_argument("--rpt-min-hours", type=float, default=2.0)
    ap.add_argument("--rpt-v-end", type=float, default=2.6,
                    help="tensão mínima que um RPT válido deve atingir (V)")
    ap.add_argument("--mission-i-min", type=float, default=1000.0,
                    help="corrente mediana mínima de uma descarga de missão (mA)")
    ap.add_argument("--mission-min-minutes", type=float, default=5.0)
    ap.add_argument("--mission-max-hours", type=float, default=1.5)
    ap.add_argument("--mission-merge", type=float, default=1800.0,
                    help="fragmentos de missão mais próximos que isso são fundidos (s)")
    ap.add_argument("--cut", type=float, default=0.20,
                    help="fração inicial de missões para o regime nominal (padrão 0.20)")
    ap.add_argument("--min-current", type=float, default=20.0,
                    help="|I| mínimo para considerar descarga (mA)")
    ap.add_argument("--max-gap", type=float, default=300.0,
                    help="lacuna máxima entre amostras dentro de um segmento (s)")
    ap.add_argument("--out", type=Path, default=None, help="PNG de saída")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell = args.csv.stem
    df = load(args.csv)
    seg = discharge_segments(df, args.min_current, args.max_gap)
    rpt, rejected = find_rpts(seg, args.rpt_i_min, args.rpt_i_max,
                              args.rpt_min_hours, args.rpt_v_end)

    if rpt.empty:
        raise SystemExit("Nenhum RPT identificado. Ajuste --rpt-i-min/--rpt-i-max/--rpt-min-hours.")

    mission_starts = find_missions(seg, args.mission_i_min, args.mission_min_minutes,
                                   args.mission_max_hours, args.mission_merge)
    n_missions = len(mission_starts)
    if n_missions == 0:
        raise SystemExit("Nenhuma missão identificada. Ajuste --mission-i-min/--mission-max-hours.")

    # Ciclo do RPT = número de missões concluídas antes dele
    rpt["cycle"] = np.searchsorted(mission_starts, rpt.t_start.to_numpy())
    rejected["cycle"] = np.searchsorted(mission_starts, rejected.t_start.to_numpy())

    ref = rpt.capacity_mah.iloc[0] if args.ref == "first" else args.nominal_mah
    rpt["soh"] = rpt.capacity_mah / ref

    print(f"\nCélula: {cell}")
    print(f"Missões identificadas: {n_missions}")
    print(f"Capacidade de referência: {ref:.1f} mAh ({args.ref})")
    print(f"\nRPTs identificados: {len(rpt)}")
    print(rpt[["cycle", "duration_h", "i_median_ma", "v_min", "capacity_mah", "soh"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not rejected.empty:
        print(f"\nRPTs descartados (não atingiram {args.rpt_v_end} V): {len(rejected)}")
        print(rejected[["cycle", "duration_h", "v_min", "capacity_mah"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    diffs = np.diff(rpt.cycle.to_numpy())
    if diffs.size and (np.abs(diffs - 50) > 10).any():
        print("\nAviso: espaçamento entre RPTs difere de ~50 missões; confira a detecção.")
    if (np.diff(rpt.soh.to_numpy()) > 0.01).any():
        print("Aviso: SOH não monótono (aumento > 1 p.p. entre RPTs); confira o gráfico.")

    cycles, soh = rpt.cycle.to_numpy(dtype=float), rpt.soh.to_numpy()
    eol_cycle = eol_crossing(cycles, soh, args.eol)
    if eol_cycle is None:
        print(f"\nEOL ({args.eol:.0%}) NÃO atingido. SOH no último RPT: {soh[-1]:.3f} "
              f"(missão {int(cycles[-1])} de {n_missions}).")
    else:
        print(f"\nCiclo de EOL (SOH < {args.eol:.0%}): missão {eol_cycle:.0f} (interpolado)")

    def soh_at(c):
        return float(np.interp(c, cycles, soh))
    cut_test = args.cut * n_missions
    print(f"Corte de {args.cut:.0%} (fim do teste): missão {cut_test:.0f}, SOH ≈ {soh_at(cut_test):.3f}")
    if eol_cycle is not None:
        cut_eol = args.cut * eol_cycle
        print(f"Corte de {args.cut:.0%} (até EOL):     missão {cut_eol:.0f}, SOH ≈ {soh_at(cut_eol):.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cycles, soh * 100, "o-", ms=4, label="SOH (RPT, C/5)")
    ax.axhline(args.eol * 100, ls="--", c="tab:red", lw=1, label=f"EOL = {args.eol:.0%}")
    if eol_cycle is not None:
        ax.axvline(eol_cycle, ls=":", c="tab:red", lw=1)
        ax.annotate(f"missão {eol_cycle:.0f}", (eol_cycle, args.eol * 100),
                    textcoords="offset points", xytext=(6, 6), color="tab:red")
    ax.set_xlabel("Missões concluídas")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"{cell}: SOH por RPT")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = args.out or args.csv.with_name(f"{cell}_soh.png")
    fig.savefig(out, dpi=200)
    print(f"Gráfico salvo em {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
