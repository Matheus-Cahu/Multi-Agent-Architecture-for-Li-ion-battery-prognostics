import { useState } from "react";
import { rotuloCampo, valorCurto } from "../lib/formato.js";
import { Laudo } from "./Laudo.jsx";

// escalares curtos viram metricas na grade; texto com quebras (o laudo) e
// objetos/arrays ganham bloco proprio embaixo
const eEscalarCurto = ([, v]) =>
  (v === null || typeof v !== "object") && !(typeof v === "string" && (v.includes("\n") || v.length > 80));

export function Relatorio({ relatorio }) {
  const [bruto, setBruto] = useState(false);

  // o payload e livre (a API aceita qualquer objeto): sem forma conhecida,
  // cai no JSON indentado de sempre
  const objeto = relatorio && typeof relatorio === "object" && !Array.isArray(relatorio);
  const campos = objeto ? Object.entries(relatorio) : [];
  const metricas = campos.filter(eEscalarCurto);
  const blocos = campos.filter((c) => !eEscalarCurto(c));

  return (
    <div className="campo">
      <div className="cabecalho-campo">
        <div className="rotulo">relatório</div>
        {objeto && (
          <button className="alternar" type="button" onClick={() => setBruto((b) => !b)}>
            {bruto ? "ver formatado" : "ver JSON bruto"}
          </button>
        )}
      </div>

      {!objeto || bruto ? (
        <pre>{JSON.stringify(relatorio, null, 2)}</pre>
      ) : (
        <>
          {metricas.length > 0 && (
            <div className="metricas">
              {metricas.map(([chave, valor]) => (
                <div className="metrica" key={chave}>
                  <div className="rotulo">{rotuloCampo(chave)}</div>
                  <div className="numero">{valorCurto(chave, valor)}</div>
                </div>
              ))}
            </div>
          )}

          {blocos.map(([chave, valor]) => (
            <div className="campo" key={chave}>
              <div className="rotulo">{rotuloCampo(chave)}</div>
              {typeof valor === "string" ? (
                <Laudo texto={valor} />
              ) : (
                <pre>{JSON.stringify(valor, null, 2)}</pre>
              )}
            </div>
          ))}
        </>
      )}
    </div>
  );
}
