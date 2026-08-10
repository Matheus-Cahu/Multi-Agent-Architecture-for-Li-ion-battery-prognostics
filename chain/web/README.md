# chain-viewer

Visualizador React das entradas ancoradas (Vite + React 19).
Consome `GET /reports` da API em `main.js`.

## Rodar

Com os serviços já no ar (`docker compose up -d`, `npm run deploy`, `npm start` na pasta `chain/`):

```bash
npm install
npm run dev          # http://localhost:5173
```

O `vite.config.js` faz proxy de `/reports` para `http://localhost:3000`, então
não há CORS no dev. Para apontar para outra API:

```bash
API_URL=http://outro-host:3000 npm run dev
```

## Build

```bash
npm run build        # gera dist/
npm run preview      # serve o dist/ (sem proxy: precisa da API na mesma origem)
```

## Estrutura

- `src/App.jsx` — estado da página (auto-refresh, cards abertos)
- `src/componentes/CardRelatorio.jsx` — uma entrada e seu status de integridade
- `src/componentes/Resumo.jsx` — contadores por status
- `src/lib/useRelatorios.js` — busca e reconsulta periódica
- `src/lib/api.js` — chamada a `GET /reports`
- `src/lib/formato.js` — rótulos de status e datas
