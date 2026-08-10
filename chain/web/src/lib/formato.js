// Vocabulario de status compartilhado entre o resumo e os cards.
// Os quatro valores espelham o que GET /reports devolve em `status`.
export const ROTULOS = {
  integro: "íntegro",
  violado: "VIOLADO",
  "sem-ancora": "sem âncora",
  "chain-indisponivel": "chain indisponível"
};

export const rotulo = (status) => ROTULOS[status] ?? status;

export function dataLocal(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "medium" });
}

// Rotulos dos campos do relatorio: as chaves chegam em snake_case do
// classifier.py. Chave desconhecida cai no fallback (payload e livre).
const CAMPOS = {
  celula: "célula",
  taxa_anomalia: "taxa de anomalia",
  indice_conformidade: "índice de conformidade",
  n_eventos: "eventos detectados",
  laudo_slm: "laudo (SLM)"
};

export const rotuloCampo = (chave) => CAMPOS[chave] ?? chave.replace(/_/g, " ");

// taxa_* e indice_* saem do classifier como fracao 0..1 e leem melhor em %
const FRACAO = /^(taxa|indice|percentual|conformidade)/;

export function valorCurto(chave, valor) {
  if (valor === null || valor === undefined) return "—";
  if (typeof valor === "boolean") return valor ? "sim" : "não";
  if (typeof valor === "number") {
    if (FRACAO.test(chave) && valor >= 0 && valor <= 1) {
      // sempre em % com duas casas: valores muito perto de zero arredondam para 0,00%
      const pct = (valor * 100).toLocaleString("pt-BR", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      });
      return `${pct}%`;
    }
    return valor.toLocaleString("pt-BR", { maximumFractionDigits: 4 });
  }
  return String(valor);
}

export function contarPorStatus(entradas) {
  const contagem = new Map();
  for (const e of entradas) contagem.set(e.status, (contagem.get(e.status) ?? 0) + 1);
  return [...contagem.entries()];
}
