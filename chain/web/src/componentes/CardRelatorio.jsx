import { dataLocal, rotulo } from "../lib/formato.js";
import { Relatorio } from "./Relatorio.jsx";

function Campo({ rotulo: nome, children }) {
  return (
    <div className="campo">
      <div className="rotulo">{nome}</div>
      <div className="valor">{children}</div>
    </div>
  );
}

// Hashes so aparecem lado a lado quando ha divergencia para explicar; nos demais
// casos um unico hash basta e o card fica mais legivel.
function Hashes({ entrada }) {
  const { status, hashOnChain, hashRecalculado } = entrada;
  if (status !== "violado" && status !== "sem-ancora") {
    return <Campo rotulo="hash ancorado">{hashRecalculado}</Campo>;
  }
  return (
    <>
      <div className={`aviso ${status}`}>
        {status === "violado"
          ? "O conteúdo no Mongo não corresponde ao hash ancorado na chain — foi alterado após a ancoragem."
          : "Nenhuma âncora encontrada na chain para este reportId."}
      </div>
      <Campo rotulo="hash on-chain (esperado)">{hashOnChain ?? "—"}</Campo>
      <Campo rotulo="hash recalculado (obtido)">{hashRecalculado}</Campo>
    </>
  );
}

export function CardRelatorio({ entrada, aberto, onAlternar }) {
  const violado = entrada.status === "violado";

  return (
    <div className={`card${violado ? " violado-borda" : ""}${aberto ? " aberto" : ""}`}>
      <button className="cabecalho" type="button" onClick={onAlternar} aria-expanded={aberto}>
        <span className="id">{entrada.reportId}</span>
        <span className={`pill ${entrada.status}`}>{rotulo(entrada.status)}</span>
        <span className="data">{dataLocal(entrada.criadoEm)}</span>
      </button>

      {aberto && (
        <div className="corpo">
          <Hashes entrada={entrada} />
          <Campo rotulo="transação">{entrada.txHash ?? "—"}</Campo>
          <Relatorio relatorio={entrada.relatorio} />
        </div>
      )}
    </div>
  );
}
