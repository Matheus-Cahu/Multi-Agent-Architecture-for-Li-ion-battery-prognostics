import { Fragment } from "react";
import { blocosDoLaudo, fatiarInline, paragrafosDaProsa } from "../lib/laudo.js";

function Linha({ texto }) {
  return fatiarInline(texto).map((parte, i) =>
    parte.tipo === "forte" ? (
      <strong key={i}>{parte.texto}</strong>
    ) : parte.tipo === "codigo" ? (
      <code key={i}>{parte.texto}</code>
    ) : (
      <Fragment key={i}>{parte.texto}</Fragment>
    )
  );
}

function Prosa({ texto }) {
  return paragrafosDaProsa(texto).map((parte, i) =>
    parte.tipo === "lista" ? (
      <ul key={i}>
        {parte.itens.map((item, j) => (
          <li key={j}>
            <Linha texto={item} />
          </li>
        ))}
      </ul>
    ) : (
      <p key={i}>
        {parte.linhas.map((linha, j) => (
          <Fragment key={j}>
            {j > 0 && <br />}
            <Linha texto={linha} />
          </Fragment>
        ))}
      </p>
    )
  );
}

export function Laudo({ texto }) {
  if (!texto?.trim()) return <div className="valor">—</div>;

  return (
    <div className="laudo">
      {blocosDoLaudo(texto).map((bloco, i) =>
        bloco.tipo === "codigo" ? (
          <pre key={i}>{bloco.texto}</pre>
        ) : (
          <Prosa key={i} texto={bloco.texto} />
        )
      )}
    </div>
  );
}
