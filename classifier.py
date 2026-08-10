import pandas as pd
import requests
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
import math
import ollama
import xml.etree.ElementTree as ET

T, S = 96, 24

scaler = joblib.load("scaler.pkl")            
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
    return torch.tensor(scaler.transform(v), dtype=torch.float32)  

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

ref_files = ["data/VAH01.csv", "data/VAH17.csv", "data/VAH27.csv"]
res_ref = []
for rf in ref_files:
    serie = carrega_norm(rf)
    corte = int(len(serie) * 0.8)                
    loader = DataLoader(JanelaDataset(serie[:corte], T, S), batch_size=64)
    res_ref.append(coletar_residuos(loader))
res_ref = torch.cat(res_ref)

mu_ref  = res_ref.mean(dim=0)                     # CONGELADO
sig_ref = res_ref.std(dim=0)                      # CONGELADO

def escores(residuos):
    z = (residuos - mu_ref) / sig_ref
    return (z ** 2).sum(dim=(1, 2)).numpy()

def confianca(anomalia, k, c):
    return 1/ (1 + math.exp(k * (anomalia - c)))

def payload_da_janela(serie, i, T, S):
    """Extrai a assinatura de anomalia da janela que comeca no indice i."""
    janela_in  = serie[i : i+T]                    # entrada [T, N]
    with torch.no_grad():
        prev = model(janela_in.unsqueeze(0))[0]    # previsto [S, N]
    real = serie[i+T : i+T+S]                       # observado [S, N]
    residuo = real - prev                           # [S, N]

    z = (residuo - mu_ref) / sig_ref               # z por sensor e horizonte [S, N]
    contrib = (z ** 2).sum(dim=0)                   # contribuicao de cada sensor [N]

    media_dif  = residuo.mean(dim=0)                # media (real - prev) por sensor, COM sinal [N]
    desvio_dif = residuo.std(dim=0)                 # desvio da diferenca por sensor [N]

    return {
        "indice": int(i),
        "escore": float((z ** 2).sum()),
        "contrib_por_sensor": contrib.numpy(),      # qual sensor puxou a anomalia
        "z_medio_por_sensor": z.mean(dim=0).numpy(),# sinal do desvio em sigma (+ ou -) por sensor
        "media_dif_por_sensor": media_dif.numpy(),  # media da diferenca real-prev por sensor
        "desvio_dif_por_sensor": desvio_dif.numpy() # desvio padrao da diferenca por sensor
    }

def payload_para_texto(p, colunas):
    linhas = []
    for n, nome in enumerate(colunas):
        media = p['media_dif_por_sensor'][n]
        desvio = p['desvio_dif_por_sensor'][n]
        linhas.append(f"- {nome}: desvio médio de {media:+.2f} (±{desvio:.2f}) em relação ao previsto")
    return "\n".join(linhas)

def payloads_para_texto(payloads, colunas):
    """Concatena a assinatura de todos os picos, do mais severo ao menos severo."""
    blocos = []
    for ordem, p in enumerate(payloads, start=1):
        sensor_top = colunas[int(p['contrib_por_sensor'].argmax())]
        blocos.append(
            f"Evento {ordem} (índice {p['indice']}, escore {p['escore']:.1f}, "
            f"sensor dominante: {sensor_top}):\n"
            f"{payload_para_texto(p, colunas)}"
        )
    return "\n\n".join(blocos)

def extrair_secoes(caminho_xml, titulos_alvo):
    """Extrai o texto das secoes cujo titulo contem um dos termos-alvo."""
    tree = ET.parse(caminho_xml)
    root = tree.getroot()
    partes = []
    for sec in root.iter("sec"):
        t_el = sec.find("title")
        titulo = "".join(t_el.itertext()) if t_el is not None else ""
        if any(alvo.lower() in titulo.lower() for alvo in titulos_alvo):
            partes.append("".join(sec.itertext()).strip())
    return "\n\n".join(partes)

def montar_prompt(nome_arquivo, taxa_anomalia, indice_conf, assinatura_texto, n_eventos):
    corpus = extrair_secoes("corpus/energies-18-00342.xml",
    ["3.1", "3.2"]        # testing/EOL + failure mechanisms
)
    print(len(corpus)//4, "tokens aprox")
    return f"""Você é um assistente de análise de baterias de eVTOL. Os valores abaixo estão em unidades normalizadas (desvios-padrão), não em unidades físicas. Um desvio positivo significa que o sensor mediu acima do previsto por um modelo de bateria saudável; negativo, abaixo.

Célula analisada: {nome_arquivo}
Percentual de janelas anômalas: {100*taxa_anomalia:.2f}%
Índice de conformidade: {100*indice_conf:.2f}%

Assinaturas dos {n_eventos} eventos anômalos distintos detectados (supressão de não máximos), ordenados do mais severo para o menos severo:
{assinatura_texto}

Base de conhecimento específico de campo: {corpus}
Formate os dados fornecidos para json, com um objeto por evento anômalo. Caso o coeficiente de confiança esteja baixo (abaixo de 80%), analise o problema fazendo uso do conhecimento específico de campo fornecido acima, considerando a evolução dos eventos ao longo dos índices."""

def analisar(nome_arquivo, taxa, indice_conf, payloads, colunas):
    assinatura = payloads_para_texto(payloads, colunas)
    prompt = montar_prompt(nome_arquivo, taxa, indice_conf, assinatura, len(payloads))
    resposta = ollama.chat(
        model="gemma2:9b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )
    return resposta["message"]["content"]

limiar = np.percentile(escores(res_ref), 99)     # CONGELADO (regua unica)
print(f"limiar unico (percentil 99 da referencia saudavel): {limiar:.1f}\n")

COLUNAS = list(pd.read_csv("data/VAH01.csv", nrows=0)
               .drop(columns=["time_s", "cycleNumber"]).columns)

files = [f.name for f in Path("data/").iterdir() if f.is_file() and f.name != "README.txt"]


for f in files:
    relatorio = ""
    serie = carrega_norm("data/" + f)
    loader = DataLoader(JanelaDataset(serie, T, S), batch_size=64)
    sc = escores(coletar_residuos(loader))
    taxa = (sc > limiar).mean()
    trust = confianca(taxa, 219.7, 0.03)

    sc_filtrado = np.where(sc > limiar, sc, -1.0)
    picos = non_maximum_supression(sc_filtrado, T, S, k=10)

    if len(picos) == 0:                      # celula sem anomalias -> nada a analisar
        print(f"{f}: sem eventos anômalos (conformidade {100*trust:.1f}%)")
        continue

    payloads = [payload_da_janela(serie, i, T, S) for i in picos]   # todos os eventos distintos
    laudo = analisar(f, taxa, trust, payloads, COLUNAS)
    relatorio_celula = {
        "celula": f,
        "taxa_anomalia": float(taxa),
        "indice_conformidade": float(trust),
        "n_eventos": len(picos),
        "laudo_slm": laudo,                    # o texto que o Gemma retornou
    }

    try:
        r = requests.post("http://localhost:3000/reports",
                          json=relatorio_celula, timeout=30)
        r.raise_for_status()
        resp = r.json()
        print(f"{f}: ancorado — reportId={resp['reportId']} tx={resp['txHash']}")
    except Exception as e:
        print(f"{f}: FALHA ao ancorar — {e}")


# for f in files:
#     serie = carrega_norm("data/" + f)
#     loader = DataLoader(JanelaDataset(serie, T, S), batch_size=64)
#     sc = escores(coletar_residuos(loader))
#     taxa = (sc > limiar).mean()
#     trust = confianca(taxa, 219.7, 0.03)
#
#     # so seleciona picos que de fato passam o limiar
#     sc_filtrado = np.where(sc > limiar, sc, -1.0)
#     picos = non_maximum_supression(sc_filtrado, T, S, k=10)
#
#     print(f"{f:14s} | anomalas: {100*taxa:5.2f}% | eventos distintos (NMS): {len(picos)}| confiança: {100*trust:5.2f}%")
#     for i in picos[:1]:                             # mostra os 3 principais
#         p = payload_da_janela(serie, i, T, S)
#         sensor_top = COLUNAS[int(p['contrib_por_sensor'].argmax())]
#         print(f"    idx {p['indice']:6d} | escore {p['escore']:7.1f} | dominado por: {sensor_top}")
#         for n, nome in enumerate(COLUNAS):
#             print(f"        {nome:20s} | dif media: {p['media_dif_por_sensor'][n]:+7.3f} | desvio: {p['desvio_dif_por_sensor'][n]:6.3f}")
