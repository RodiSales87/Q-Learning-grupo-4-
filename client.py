import numpy as np
import random
import time
import socket
import argparse
from connection import connect, get_state_reward

class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.actions = ["left", "right", "jump"]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
        self.num_states = 96  # 24 plataformas × 4 direções
        self.num_actions = len(self.actions)
        self.q_table = np.zeros((self.num_states, self.num_actions))
    
    def state_to_index(self, state_bin):
        """Converte estado binário (7 bits) para índice (0-95)"""
        platform = int(state_bin[:5], 2)
        direction = int(state_bin[5:], 2)
        return platform * 4 + direction
    
    def choose_action(self, state_idx):
        """Escolhe ação usando política ε-greedy"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state_idx])]
    
    def update_q_table(self, state_idx, action, reward, next_state_idx):
        """Atualiza Q-table usando Q-learning"""
        action_idx = self.action_to_idx[action]
        current_q = self.q_table[state_idx, action_idx]
        max_next_q = np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action_idx] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def train(self, socket_conn, episodes, max_steps):
        """Executa o treinamento com limite de steps e tratamento de falhas"""
        total_success = 0

        for episode in range(episodes):
            try:
                # Tenta obter estado inicial
                state_bin, reward = get_state_reward(socket_conn, "jump")
                state_idx = self.state_to_index(state_bin)
                episode_success = False
                for step in range(max_steps):
                    action = self.choose_action(state_idx)

                    try:
                        next_state_bin, reward = get_state_reward(socket_conn, action)
                    except Exception as e:
                        print(f"⚠️ Episódio {episode+1}, passo {step+1}: erro ao receber resposta -> {e}")
                        break  # Interrompe o episódio atual, mas continua o loop de treino

                    next_state_idx = self.state_to_index(next_state_bin)
                    self.update_q_table(state_idx, action, reward, next_state_idx)
                    # Dentro do loop de treino:
                    # Primeiro ajusta a recompensa com base na progressão
                    if action == "jump" and next_state_idx > state_idx:
                        reward += 6
                    if next_state_idx == state_idx:
                        reward -= 1  # ou 0.5
                    if next_state_idx > state_idx:
                        reward += 5
                    elif next_state_idx < state_idx:
                        reward -= 0.5
                    if next_state_bin == '0b0000000':
                        reward = -20  # ou -30 no lugar de -100
                        print(f"☠️ Episódio {episode+1}: morreu no passo {step+1}")
                        break
                    # se está na mesma plataforma, mas muda só direção — penaliza mais
                    if state_bin[:5] == next_state_bin[:5] and state_bin[5:] != next_state_bin[5:]:
                        reward -= 1.5  # penaliza ficar rodando no mesmo lugar

                    # Depois atualiza a Q-table com a recompensa ajustada
                    self.update_q_table(state_idx, action, reward, next_state_idx)

                    if not episode_success and next_state_bin.startswith('10101'):
                        print(f"🎉 Episódio {episode+1}: objetivo alcançado em {step+1} passos!")
                        total_success += 1
                        episode_success = True

                    state_idx = next_state_idx
                    print(f"[{episode+1}/{episodes}] step {step+1} | ação: {action} | estado: {next_state_bin} | recompensa: {reward}")

                
                self.epsilon = max(0.01, self.epsilon * 0.98)


            except Exception as e:
                print(f"🚨 Falha no início do episódio {episode+1}: {e}")
                continue  # pula para o próximo episódio
            
        if total_success > 0:
            print(f"✅ {total_success} episódios chegaram ao objetivo final!")
        else:
            print("⚠️ Nenhum sucesso até o momento. Pode ser necessário mais episódios.")

    def save_q_table(self, filename="q_tableTeste.txt"):
        """Salva Q-table no formato correto da entrega"""
        with open(filename, "w") as f:
            for row in self.q_table:
                f.write(" ".join(f"{q:.6f}" for q in row) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episódios de treino")
    parser.add_argument("--port", type=int, default=2037, help="Porta do servidor do jogo")
    args = parser.parse_args()

    print("Iniciando cliente Q-Learning...")

    try:
        print(f"Conectando na porta {args.port}...")
        socket_conn = connect(args.port)

        if not socket_conn:
            print("🚫 Não foi possível conectar ao jogo. Encerrando.")
            return
        
        agent = QLearningAgent()
        print(f"Iniciando treinamento por {args.episodes} episódios...")
        agent.train(socket_conn, episodes=args.episodes, max_steps=150)

        
        agent.save_q_table()
        print("Q-table salva em q_table.txt")
    
    except Exception as e:
        print(f"Erro fatal: {str(e)}")
    finally:
        if 'socket_conn' in locals() and isinstance(socket_conn, socket.socket):
            socket_conn.close()
            print("Conexão encerrada")

if __name__ == "__main__":
    main()
