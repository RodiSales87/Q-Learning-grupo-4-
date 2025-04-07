import numpy as np
import random
import time
import socket
import argparse
from connection import connect, get_state_reward
import os
class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=4.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.qtable_path = r'C:\Projetos\Qlearning-main\q_table.txt'

        self.actions = ["left", "right", "jump"]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}

        self.num_states = 96
        self.num_actions = len(self.actions)

        # Novo: carregar Q-table se existir
        if os.path.exists(self.qtable_path):
            self.q_table = np.loadtxt(self.qtable_path)
            print("✅ Q-table carregada do disco.")
        else:
            self.q_table = np.zeros((self.num_states, self.num_actions))
            print("📄 Nova Q-table criada.")
    def state_to_index(self, state_bin):
        platform = int(state_bin[:5], 2)
        direction = int(state_bin[5:], 2)
        return platform * 4 + direction

    def choose_action(self, state_idx, rotation_counter):
        if self.epsilon > random.random():
            # Limita rotações (left/right) se número excedido
            if rotation_counter >= 1:
                return "jump"
            return random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state_idx])]

    def update_q_table(self, state_idx, action, reward, next_state_idx):
        action_idx = self.action_to_idx[action]
        current_q = self.q_table[state_idx, action_idx]
        max_next_q = np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action_idx] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

   

    def train(self, socket_conn, episodes, max_steps):
        total_success = 0

        for episode in range(episodes):
            try:
                plataforma_avancada = random.choice([True, False])
                if plataforma_avancada:
                    state_bin = self.alcançar_plataforma(socket_conn, plataforma_desejada=16)
                else:
                    state_bin, reward = get_state_reward(socket_conn, "jump")
                state_idx = self.state_to_index(state_bin)
                episode_success = False
                rotation_counter = 0

                for step in range(max_steps):
                    action = self.choose_action(state_idx, rotation_counter)

                    if action in ["left", "right"]:
                        rotation_counter += 1
                    else:
                        rotation_counter = 0

                    try:
                        next_state_bin, reward = get_state_reward(socket_conn, action)
                    except Exception as e:
                        print(f"⚠️ Episódio {episode+1}, passo {step+1}: erro ao receber resposta -> {e}")
                        break

                    next_state_idx = self.state_to_index(next_state_bin)

                    if action == "jump" and next_state_idx > state_idx:
                        reward += 8
                    if next_state_idx > state_idx:
                        reward += 5
                    elif next_state_idx < state_idx:
                        reward -= 1
                    if next_state_idx == state_idx:
                        reward -= 0.5
                    if state_bin[:5] == next_state_bin[:5] and state_bin[5:] != next_state_bin[5:]:
                        reward -= 1.5
                    if next_state_bin == '0b0000000':
                        reward = -30
                        print(f"☠️ Episódio {episode+1}: morreu no passo {step+1}")
                        break

                    self.update_q_table(state_idx, action, reward, next_state_idx)

                    if not episode_success and next_state_bin.startswith('10101'):
                        print(f"🎉 Episódio {episode+1}: objetivo alcançado em {step+1} passos!")
                        total_success += 1
                        episode_success = True

                    state_idx = next_state_idx
                    state_bin = next_state_bin

                    print(f"[{episode+1}/{episodes}] step {step+1} | ação: {action} | estado: {next_state_bin} | recompensa: {reward}")

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            except Exception as e:
                print(f"🚨 Falha no início do episódio {episode+1}: {e}")
                continue

        print(f"\n✅ Total de episódios com sucesso: {total_success}")

    def alcançar_plataforma(self, socket_conn, plataforma_desejada):
        comandos = ["jump", "jump", "jump", "jump", "left", "jump", "right", "jump", "left", "jump","right", "jump", "left", "jump", "jump", "jump", "left", "jump", "jump", "jump", "right" ]  # exemplo genérico
        for i in range(30):
            estado_bin, _ = get_state_reward(socket_conn, comandos[i % len(comandos)])
            plataforma = int(estado_bin[:5], 2)
            if plataforma >= plataforma_desejada:
                print(f"✅ Plataforma {plataforma_desejada} alcançada (atual: {plataforma}) após {i+1} ações.")
                return estado_bin
        print(f"⚠️ Não foi possível alcançar a plataforma {plataforma_desejada} após {30} tentativas.")
        return estado_bin  # Retorna onde chegou mesmo assim
    
    def save_q_table(self, filename="q_table.txt"):
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
        agent.train(socket_conn, episodes=args.episodes, max_steps=50)

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
