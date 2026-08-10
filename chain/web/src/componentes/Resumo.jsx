import { contarPorStatus, rotulo } from "../lib/formato.js";

export function Resumo({ total, entradas }) {
  return (
    <div className="resumo">
      <span className="pill">{total} no banco</span>
      {contarPorStatus(entradas).map(([status, n]) => (
        <span key={status} className={`pill ${status}`}>
          {n} {rotulo(status)}
        </span>
      ))}
    </div>
  );
}
