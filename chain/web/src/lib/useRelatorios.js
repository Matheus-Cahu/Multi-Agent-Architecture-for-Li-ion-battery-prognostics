import { useCallback, useEffect, useRef, useState } from "react";
import { buscarRelatorios } from "./api.js";

// Carrega a listagem e opcionalmente reconsulta em intervalo fixo.
export function useRelatorios({ auto, intervaloMs = 5000 }) {
  const [dados, setDados] = useState(null);
  const [erro, setErro] = useState(null);
  const [carregando, setCarregando] = useState(true);
  const abortRef = useRef(null);

  const recarregar = useCallback(async () => {
    abortRef.current?.abort();               // descarta requisicao anterior ainda em voo
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const novos = await buscarRelatorios(200, ctrl.signal);
      setDados(novos);
      setErro(null);
    } catch (e) {
      if (e.name === "AbortError") return;   // substituida por outra chamada, nao e falha
      setErro(e.message);
    } finally {
      if (!ctrl.signal.aborted) setCarregando(false);
    }
  }, []);

  useEffect(() => {
    recarregar();
    return () => abortRef.current?.abort();
  }, [recarregar]);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(recarregar, intervaloMs);
    return () => clearInterval(id);
  }, [auto, intervaloMs, recarregar]);

  return { dados, erro, carregando, recarregar };
}
