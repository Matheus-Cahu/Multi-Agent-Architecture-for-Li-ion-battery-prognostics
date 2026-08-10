// Caminho relativo de proposito: no dev o proxy do Vite encaminha para a API,
// e num build servido pelo proprio express nao ha nada para reconfigurar.
export async function buscarRelatorios(limite = 200, sinal) {
  const r = await fetch(`/reports?limite=${limite}`, { signal: sinal });

  let dados;
  try {
    dados = await r.json();
  } catch {
    // API fora do ar costuma devolver HTML de erro, nao JSON
    throw new Error(`resposta invalida da API (HTTP ${r.status})`);
  }

  if (!r.ok) throw new Error(dados.detalhe || dados.erro || `HTTP ${r.status}`);
  return dados;
}
