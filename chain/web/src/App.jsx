import { useState } from "react";
import { CardRelatorio } from "./componentes/CardRelatorio.jsx";
import { Resumo } from "./componentes/Resumo.jsx";
import { useRelatorios } from "./lib/useRelatorios.js";

export default function App() {
  const [auto, setAuto] = useState(false);
  const [abertos, setAbertos] = useState(() => new Set());
  const { dados, erro, carregando, recarregar } = useRelatorios({ auto });

  function alternar(reportId) {
    setAbertos((atual) => {
      const proximo = new Set(atual);
      proximo.has(reportId) ? proximo.delete(reportId) : proximo.add(reportId);
      return proximo;
    });
  }

  const entradas = dados?.entradas ?? [];

  return (
    <div className="container">
      <header>
        <div>
          <h1>ReportAnchor — entradas do banco</h1>
          <div className="sub">
            Cada relatório do Mongo é re-hasheado e comparado com a âncora on-chain.
          </div>
        </div>
        <div className="acoes">
          <label className="data">
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> auto (5s)
          </label>
          <button onClick={recarregar}>Recarregar</button>
        </div>
      </header>

      {/* o erro nao substitui a lista: dados antigos seguem visiveis se a API cair */}
      {erro && <div className="aviso violado">Falha ao carregar: {erro}</div>}

      {dados && <Resumo total={dados.total} entradas={entradas} />}

      {carregando ? (
        <div className="carregando">carregando…</div>
      ) : entradas.length === 0 ? (
        !erro && <div className="vazio">Nenhum relatório ancorado ainda.</div>
      ) : (
        <div className="lista">
          {entradas.map((entrada) => (
            <CardRelatorio
              key={entrada.reportId}
              entrada={entrada}
              aberto={abertos.has(entrada.reportId)}
              onAlternar={() => alternar(entrada.reportId)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
