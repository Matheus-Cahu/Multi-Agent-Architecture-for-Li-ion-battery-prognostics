// O laudo do SLM chega como UMA string markdown-ish: blocos cercados por ```,
// paragrafos separados por linha em branco, listas com "*"/"-" e enfase com **
// ou crase. Dentro de JSON.stringify isso vira "\n" literal e fica ilegivel —
// aqui o texto e quebrado em blocos para o componente montar os elementos.

const CERCA = /```([\w-]*)[ \t]*\r?\n?([\s\S]*?)```/g;
const ITEM_LISTA = /^\s*[-*•]\s+/;
const INLINE = /\*\*([^*]+)\*\*|`([^`]+)`/g;

// o bloco cercado costuma ser o json de eventos: reindenta quando faz parse,
// senao preserva exatamente o que o modelo escreveu
function normalizarCodigo(texto, lingua) {
  const bruto = texto.replace(/\s+$/, "");
  if (lingua && lingua !== "json") return bruto;
  try {
    return JSON.stringify(JSON.parse(bruto), null, 2);
  } catch {
    return bruto;
  }
}

export function blocosDoLaudo(texto = "") {
  const blocos = [];
  let cursor = 0;
  for (const m of texto.matchAll(CERCA)) {
    if (m.index > cursor) blocos.push({ tipo: "prosa", texto: texto.slice(cursor, m.index) });
    blocos.push({ tipo: "codigo", texto: normalizarCodigo(m[2], m[1]) });
    cursor = m.index + m[0].length;
  }
  if (cursor < texto.length) blocos.push({ tipo: "prosa", texto: texto.slice(cursor) });
  return blocos.filter((b) => b.texto.trim() !== "");
}

// Quebra a prosa em paragrafos e listas. Linhas consecutivas de "*"/"-" viram
// um <ul> unico mesmo sem linha em branco antes; as demais mantem a quebra
// simples de linha, que no laudo separa itens de raciocinio.
export function paragrafosDaProsa(texto) {
  const partes = [];
  for (const bloco of texto.split(/\r?\n\s*\r?\n/)) {
    let atual = null;
    for (const linha of bloco.split(/\r?\n/)) {
      if (!linha.trim()) continue;
      const tipo = ITEM_LISTA.test(linha) ? "lista" : "paragrafo";
      if (atual?.tipo !== tipo) {
        atual = tipo === "lista" ? { tipo, itens: [] } : { tipo, linhas: [] };
        partes.push(atual);
      }
      if (tipo === "lista") atual.itens.push(linha.replace(ITEM_LISTA, ""));
      else atual.linhas.push(linha.trim());
    }
  }
  return partes;
}

// **negrito** e `codigo` numa linha -> pedacos que o componente vira elementos
export function fatiarInline(linha) {
  const partes = [];
  let cursor = 0;
  for (const m of linha.matchAll(INLINE)) {
    if (m.index > cursor) partes.push({ tipo: "texto", texto: linha.slice(cursor, m.index) });
    partes.push(m[1] != null ? { tipo: "forte", texto: m[1] } : { tipo: "codigo", texto: m[2] });
    cursor = m.index + m[0].length;
  }
  if (cursor < linha.length) partes.push({ tipo: "texto", texto: linha.slice(cursor) });
  return partes;
}
