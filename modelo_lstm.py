"""
Backbone LSTM para comparação com o iTransformer.

Mesma interface: entrada [B, T, N] -> saída [B, S, N].
Arquitetura: encoder LSTM empilhado + cabeça linear direta (multi-output),
que prevê os S passos de uma vez a partir do último estado oculto. A previsão
direta evita decodificação autorregressiva e usa a mesma perda (MSE sobre os
S x N valores) do iTransformer, então a diferença entre os modelos fica na
forma de tokenização: o LSTM mistura todas as variáveis a cada instante
(tokens temporais), o iTransformer trata cada variável como um token.

hidden=176 com 2 camadas e N=7 dá ~409 mil parâmetros, próximo dos ~412 mil
do iTransformer (d_model=128, 4 cabeças, 2 blocos): capacidade comparável.
"""
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_vars: int, seq_len: int, pred_len: int,
                 hidden: int = 176, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len  # não usado; mantido pela interface comum
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.lstm = nn.LSTM(input_size=n_vars, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, pred_len * n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)              # [B, T, hidden]
        y = self.head(out[:, -1])          # [B, S*N]
        return y.view(-1, self.pred_len, self.n_vars)


def n_parametros_esperado(n_vars: int, pred_len: int, hidden: int, layers: int) -> int:
    """Contagem analítica (PyTorch usa dois vetores de viés por camada)."""
    total = 0
    for l in range(layers):
        entrada = n_vars if l == 0 else hidden
        total += 4 * hidden * (entrada + hidden) + 8 * hidden
    return total + hidden * pred_len * n_vars + pred_len * n_vars


if __name__ == "__main__":
    # Autoteste rápido: python modelo_lstm.py
    T, S, N = 96, 24, 7
    m = LSTMForecaster(N, T, S)
    y = m(torch.randn(4, T, N))
    n = sum(p.numel() for p in m.parameters())
    print("saída:", tuple(y.shape), "(esperado (4, 24, 7))")
    print("parâmetros:", n, "| esperado:", n_parametros_esperado(N, S, 176, 2))
