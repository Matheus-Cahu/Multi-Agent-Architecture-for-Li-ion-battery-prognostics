#!/usr/bin/env bash
# start.sh — sobe todos os servicos do projeto na ordem certa:
#   1. mongo + hardhat node (docker compose)
#   2. compila e deploya o ReportAnchor, gravando o endereco no .env
#   3. API (main.js, :3000)
#   4. front (vite dev, :5173)
#
# O hardhat node perde o estado da chain a cada restart, entao o deploy e o
# CONTRACT_ADDR do .env sao refeitos em toda execucao — e por isso que o passo 2
# vem antes de subir a API, que le o endereco na carga do modulo.
#
# Uso:
#   ./start.sh            sobe tudo e segue em foreground (Ctrl+C encerra)
#   ./start.sh --parar    derruba os containers e sai
#
# Ctrl+C mata API e front; os containers ficam de pe (o volume do Mongo
# preserva os relatorios). Use --parar para derrubar tambem.

set -euo pipefail
set -m                                   # cada servico em seu proprio grupo de processos

cd "$(dirname "$0")"

PORT_API=${PORT:-3000}
PORT_WEB=5173
PORT_RPC=8545

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
erro() { printf '\n\033[1;31m==> %s\033[0m\n' "$*" >&2; }

if [[ "${1:-}" == "--parar" ]]; then
  log "derrubando containers"
  docker compose down
  exit 0
fi

for cmd in docker node npm curl; do
  command -v "$cmd" >/dev/null || { erro "$cmd nao encontrado no PATH"; exit 1; }
done

# porta ocupada por um processo que nao e nosso -> aborta com mensagem clara,
# em vez de deixar o servico morrer com EADDRINUSE no meio do log
porta_ocupada() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
for p in "$PORT_API" "$PORT_WEB"; do
  if porta_ocupada "$p"; then
    erro "porta $p ja esta em uso — encerre o processo anterior (talvez outro ./start.sh)"
    exit 1
  fi
done

# ---------- 1. containers ----------
log "subindo mongo e hardhat node (docker compose)"
docker compose up -d

espera() {  # descricao, tentativas, comando...
  local desc=$1 tentativas=$2; shift 2
  printf 'esperando %s' "$desc"
  for ((i = 0; i < tentativas; i++)); do
    if "$@" >/dev/null 2>&1; then printf ' ok\n'; return 0; fi
    printf '.'; sleep 2
  done
  printf '\n'
  erro "$desc nao respondeu — veja 'docker compose logs'"
  return 1
}

rpc_pronto() {
  curl -sf -X POST -H 'Content-Type: application/json' \
    --data '{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}' \
    "http://127.0.0.1:$PORT_RPC"
}
mongo_pronto() {
  docker compose exec -T mongo mongosh --quiet --eval 'db.runCommand({ping:1})'
}

# o container do hardhat roda `npm install` antes do node, entao a primeira
# subida demora bem mais que as seguintes
espera "o RPC da chain (:$PORT_RPC)" 90 rpc_pronto
espera "o Mongo (:27017)" 30 mongo_pronto

# ---------- 2. contrato ----------
[[ -d node_modules ]]     || { log "instalando dependencias da API"; npm install; }
[[ -d web/node_modules ]] || { log "instalando dependencias do front"; npm --prefix web install; }

log "compilando o contrato"
npx hardhat compile

log "deployando o ReportAnchor"
saida_deploy=$(npx hardhat run scripts/deploy.js --network localhost)
echo "$saida_deploy"
endereco=$(grep -oE '0x[a-fA-F0-9]{40}' <<<"$saida_deploy" | tail -1)
[[ -n "$endereco" ]] || { erro "nao consegui extrair o endereco do contrato do deploy"; exit 1; }

# grava no .env (o npm start carrega esse arquivo com --env-file)
touch .env
if grep -q '^CONTRACT_ADDR=' .env; then
  sed -i "s|^CONTRACT_ADDR=.*|CONTRACT_ADDR=$endereco|" .env
else
  printf 'CONTRACT_ADDR=%s\n' "$endereco" >>.env
fi
log "CONTRACT_ADDR=$endereco gravado no .env"

# ---------- 3 e 4. API e front ----------
pids=()
inicia() {  # nome, comando...
  local nome=$1; shift
  ( "$@" 2>&1 | sed -u "s/^/[$nome] /" ) &
  pids+=("$!")
}

encerra() {
  trap - INT TERM EXIT
  log "encerrando API e front (containers seguem no ar; ./start.sh --parar derruba)"
  for pid in "${pids[@]}"; do
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap encerra INT TERM EXIT

log "subindo API (:$PORT_API) e front (:$PORT_WEB)"
inicia api node --env-file=.env main.js
inicia web npm --prefix web run dev

cat <<EOF

  API    http://localhost:$PORT_API
  front  http://localhost:$PORT_WEB
  RPC    http://localhost:$PORT_RPC
  Mongo  mongodb://localhost:27017

  Ctrl+C para encerrar.

EOF

wait -n            # se um dos dois cair, o trap derruba o outro
