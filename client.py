import numpy as np
import random
import time
import socket
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
    
    def train(self, socket_conn, episodes=1000):
        """Executa o treinamento"""
        for episode in range(episodes):
            try:
                # Estado inicial
                state_bin, reward = get_state_reward(socket_conn, "jump")  # Corrigido: socket_conn é o primeiro argumento
                state_idx = self.state_to_index(state_bin)
                
                terminal = False
                while not terminal:
                    # Escolhe e executa ação
                    action = self.choose_action(state_idx)
                    next_state_bin, reward = get_state_reward(socket_conn, action)  # Corrigido: socket_conn é o primeiro argumento
                    next_state_idx = self.state_to_index(next_state_bin)
                    
                    # Atualiza Q-table
                    self.update_q_table(state_idx, action, reward, next_state_idx)
                    
                    # Verifica se chegou ao objetivo (plataforma 21)
                    if next_state_bin.startswith('10101'):
                        terminal = True
                        print(f"Episódio {episode+1}: Objetivo alcançado!")
                    
                    # Atualiza estado
                    state_idx = next_state_idx
                    state_bin = next_state_bin
                    
                    # Pequena pausa
                    time.sleep(0.01)
                
                if (episode + 1) % 100 == 0:
                    print(f"Progresso: {episode+1}/{episodes} episódios")
            
            except Exception as e:
                print(f"Erro no episódio {episode+1}: {str(e)}")
                break
        
    def save_q_table(self, filename="q_table.txt"):
        """Salva Q-table no formato especificado"""
        np.savetxt(filename, self.q_table, fmt='%.6f')

def main():
    print("Iniciando cliente Q-Learning...")
    
    try:
        # Conecta ao jogo
        print("Conectando na porta 2037...")
        socket_conn = connect(2037)
        
        # Treina agente
        agent = QLearningAgent()
        print("Iniciando treinamento...")
        agent.train(socket_conn, episodes=1000)
        
        # Salva resultados
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