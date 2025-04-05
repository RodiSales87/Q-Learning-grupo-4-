# Q-Learning Game Client - Explicação e Setup

Este documento explica o funcionamento básico do `client.py`, que utiliza Q-Learning para treinar um agente em um ambiente de jogo controlado por socket.

## 🧠 Estrutura do `client.py`

### 1. Classe `QLearningAgent`
Contém a lógica do agente inteligente. Os principais componentes são:

- `alpha`: taxa de aprendizado (0 < alpha ≤ 1)
- `gamma`: fator de desconto futuro (0 < gamma ≤ 1)
- `epsilon`: taxa de exploração (probabilidade de agir aleatoriamente)

#### Métodos importantes:
- `choose_action(state_idx)`: aplica a política ε-greedy para escolher uma ação.
- `update_q_table(...)`: atualiza a Q-table com base na equação Q-learning.
- `state_to_index(state_bin)`: converte um estado binário (ex: `0b1000111`) em um índice entre 0 e 95.
- `train(socket_conn, episodes, max_steps)`: faz o loop de aprendizado principal.
- `save_q_table(filename)`: salva a Q-table após o treinamento.

### 2. Função `main()`
Executa o processo completo:
1. Lê os argumentos via `argparse`:
   - `--episodes`: número de episódios de treino.
   - `--port`: porta para conectar ao servidor do jogo.
2. Cria o socket.
3. Executa o treinamento.
4. Salva a Q-table.
5. Fecha a conexão.

## ⚙️ Setup Inicial

### Pré-requisitos
- Python 3.7+
- Socket server rodando localmente na porta especificada (ex: 2037)

### Instalação
Crie um ambiente virtual e instale dependências:
```bash
python -m venv env
source env/bin/activate  # Windows: env\\Scripts\\activate
pip install -r requirements.txt  # se existir
