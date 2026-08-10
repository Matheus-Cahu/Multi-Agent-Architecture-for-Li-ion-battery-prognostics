// main.js — API intermediadora: ancora hash na chain, guarda dados no Mongo,
// valida integridade na leitura. Ordem de escrita: CHAIN primeiro, MONGO depois.

const express = require("express");
const path = require("path");
const crypto = require("crypto");
const { ethers } = require("ethers");
const { MongoClient } = require("mongodb");

// ---------- configuracao (via env, com defaults de desenvolvimento) ----------
const PORT            = process.env.PORT            || 3000;
const RPC_URL         = process.env.RPC_URL         || "http://localhost:8545";
const MONGO_URL       = process.env.MONGO_URL       || "mongodb://localhost:27017";
const CONTRACT_ADDR   = process.env.CONTRACT_ADDR;   // definido apos o deploy
// conta #0 padrao do hardhat node (chave publica de teste, sem valor real)
const PRIVATE_KEY     = process.env.PRIVATE_KEY     ||
  "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

const ABI = [
  "function anchor(bytes32 reportId, bytes32 contentHash) external",
  "function getHash(bytes32 reportId) external view returns (bytes32)"
];

// ---------- canonicalizacao: serializacao DETERMINISTICA (recursiva) ----------
// ordena chaves em todos os niveis, para que o mesmo objeto produza sempre
// os mesmos bytes -> o mesmo hash. Usada nas DUAS pontas (ancorar e validar).
function stableStringify(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(stableStringify).join(",") + "]";
  const keys = Object.keys(value).sort();
  return "{" + keys.map(k => JSON.stringify(k) + ":" + stableStringify(value[k])).join(",") + "}";
}

function hashCanonico(objeto) {
  const canonical = stableStringify(objeto);
  return "0x" + crypto.createHash("sha256").update(canonical).digest("hex");
}

// ---------- setup de chain e banco ----------
// validado antes de construir o contrato: com CONTRACT_ADDR indefinido a ethers
// lancaria "invalid value for Contract target" ja na carga do modulo
if (!CONTRACT_ADDR) {
  console.error("defina CONTRACT_ADDR (endereco do contrato, saida de npm run deploy)");
  process.exit(1);
}

const provider = new ethers.JsonRpcProvider(RPC_URL);
const signer   = new ethers.Wallet(PRIVATE_KEY, provider);
const contract = new ethers.Contract(CONTRACT_ADDR, ABI, signer);

const mongo = new MongoClient(MONGO_URL);
let colecao;

// ---------- API ----------
const app = express();
app.use(express.json({ limit: "5mb" }));
// pagina de visualizacao: build do front (npm --prefix web run build) em web/dist.
// __dirname para nao depender do cwd. Em dev o Vite serve a pagina em :5173 e
// so encaminha /reports para ca — este static vale para o modo produção.
app.use(express.static(path.join(__dirname, "web", "dist")));

// ESCRITA: recebe relatorio -> hash -> ancora na CHAIN -> grava no MONGO
app.post("/reports", async (req, res) => {
  const relatorio = req.body;
  if (!relatorio || typeof relatorio !== "object") {
    return res.status(400).json({ erro: "corpo deve ser um objeto JSON" });
  }

  const contentHash = hashCanonico(relatorio);
  const reportId = "0x" + crypto.randomBytes(32).toString("hex"); // id unico (bytes32)

  // 1) CHAIN PRIMEIRO — se falhar, nada e gravado no Mongo (nao ha dado orfao)
  let txHash;
  try {
    const tx = await contract.anchor(reportId, contentHash);
    await tx.wait();               // espera a transacao ser minerada
    txHash = tx.hash;
  } catch (e) {
    return res.status(502).json({ erro: "falha ao ancorar na chain", detalhe: e.message });
  }

  // 2) MONGO DEPOIS — se falhar aqui, sobra uma ancora orfa (inofensiva); logamos
  try {
    await colecao.insertOne({ reportId, contentHash, txHash, relatorio, criadoEm: new Date() });
  } catch (e) {
    console.error(`ANCORA ORFA: reportId=${reportId} txHash=${txHash} — Mongo falhou:`, e.message);
    return res.status(500).json({ erro: "ancorado na chain, mas falhou ao persistir", reportId, txHash });
  }

  res.status(201).json({ reportId, txHash, contentHash });
});

const HASH_ZERO = "0x" + "0".repeat(64);

// LISTAGEM: alimenta a pagina de visualizacao. Diferente do GET por id, nao
// esconde os divergentes — mostra o status de integridade de cada entrada.
app.get("/reports", async (req, res) => {
  const limite = Math.min(parseInt(req.query.limite, 10) || 50, 200);

  try {
    const docs = await colecao.find({}, { sort: { criadoEm: -1 }, limit: limite }).toArray();

    const entradas = await Promise.all(docs.map(async (doc) => {
      const hashRecalculado = hashCanonico(doc.relatorio);
      let hashOnChain, status;
      try {
        hashOnChain = await contract.getHash(doc.reportId);
        if (hashOnChain === HASH_ZERO)                                    status = "sem-ancora";
        else if (hashOnChain.toLowerCase() === hashRecalculado.toLowerCase()) status = "integro";
        else                                                              status = "violado";
      } catch (e) {
        status = "chain-indisponivel";                 // nao confundir com adulteracao
      }
      return {
        reportId: doc.reportId, txHash: doc.txHash, criadoEm: doc.criadoEm,
        hashOnChain, hashRecalculado, status, relatorio: doc.relatorio
      };
    }));

    res.json({ total: await colecao.countDocuments(), entradas });
  } catch (e) {
    res.status(502).json({ erro: "falha ao listar relatorios", detalhe: e.message });
  }
});

// LEITURA: recupera do Mongo -> recalcula hash -> compara com a CHAIN -> so devolve se bater
app.get("/reports/:id", async (req, res) => {
  const reportId = req.params.id;

  // express 4 nao captura rejeicao de handler async: sem try/catch o processo cai
  try {
    const doc = await colecao.findOne({ reportId });
    if (!doc) return res.status(404).json({ erro: "relatorio nao encontrado" });

    const hashRecalculado = hashCanonico(doc.relatorio);          // hash do que esta no Mongo
    const hashOnChain     = await contract.getHash(reportId);      // hash imutavel na chain

    if (hashRecalculado.toLowerCase() !== hashOnChain.toLowerCase()) {
      // divergencia = o conteudo no Mongo foi adulterado apos a ancoragem
      return res.status(409).json({
        erro: "INTEGRIDADE VIOLADA: conteudo diverge da ancora on-chain",
        esperado: hashOnChain, obtido: hashRecalculado
      });
    }

    res.json({ reportId, txHash: doc.txHash, relatorio: doc.relatorio });
  } catch (e) {
    res.status(502).json({ erro: "falha ao consultar relatorio", detalhe: e.message });
  }
});

// ---------- inicializacao ----------
async function start() {
  await mongo.connect();
  colecao = mongo.db("anchor").collection("reports");
  app.listen(PORT, () => console.log(`API ouvindo em http://localhost:${PORT}`));
}
start().catch((e) => { console.error("falha ao iniciar:", e.message); process.exit(1); });
