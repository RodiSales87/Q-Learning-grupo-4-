import numpy as np
import random
import os
from connection import connect, get_state_reward

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.7, epsilon_min=0.05, epsilon_decay=0.997):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.actions = ["left", "right", "jump"]
        self.num_states = 96
        
        self.base_starting_platform = 0
        self.current_starting_platform = 0
        self.platform_attempts = 0
        self.platform_max_attempts = 500
        
        self.max_consecutive_rotations = 3
        self.max_repeated_actions = 3
        
        self.qtable_path = 'q_table.txt'
        if os.path.exists(self.qtable_path):
            self.q_table = np.loadtxt(self.qtable_path)
            print("✅ Q-table carregada do disco.")
        else:
            self.q_table = np.zeros((self.num_states, len(self.actions)))
            print("📄 Nova Q-table criada.")

    def state_to_index(self, state_bin):
        state_bin = state_bin[2:]
        platform = int(state_bin[:5], 2)
        direction = int(state_bin[5:], 2)
        return platform * 3 + direction

    def binary_to_platform(self, state_bin):
        return int(state_bin[2:7], 2)

    def is_on_target_platform(self, state_bin, target_platform):
        return self.binary_to_platform(state_bin) == target_platform

    def choose_action(self, state_idx, action_history):
        if len(action_history) >= self.max_repeated_actions:
            last_actions = action_history[-self.max_repeated_actions:]
            if all(a == last_actions[0] for a in last_actions):
                allowed_actions = [a for a in self.actions if a != last_actions[0]]
                return random.choice(allowed_actions)

        if len(action_history) >= self.max_consecutive_rotations:
            last_actions = action_history[-self.max_consecutive_rotations:]
            if all(a in ["left", "right"] for a in last_actions):
                return "jump"

        if random.random() < self.epsilon:
            action_weights = [1.0, 1.0, 1.5]
            recent_actions = action_history[-2:] if len(action_history) >= 2 else []
            for i, action in enumerate(self.actions):
                if action in recent_actions:
                    action_weights[i] *= 0.7
            return random.choices(self.actions, weights=action_weights)[0]
        
        return self.actions[np.argmax(self.q_table[state_idx])]

    def update_q_table(self, state_idx, action, reward, next_state_idx):
        action_idx = self.actions.index(action)
        current_q = self.q_table[state_idx, action_idx]
        max_next_q = np.max(self.q_table[next_state_idx])
        
        if state_idx == next_state_idx:
            reward -= 2.0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_idx, action_idx] = new_q

    def move_to_starting_platform(self, socket_conn, target_platform):
        state_bin, _ = get_state_reward(socket_conn, "jump")
        
        if self.is_on_target_platform(state_bin, target_platform):
            return state_bin
            
        print(f"\n⏫ Movendo para plataforma inicial {target_platform}...")
        attempts = 0
        max_attempts = 30
        
        while not self.is_on_target_platform(state_bin, target_platform) and attempts < max_attempts:
            state_idx = self.state_to_index(state_bin)
            action = self.choose_action(state_idx, [])
            state_bin, _ = get_state_reward(socket_conn, action)
            new_platform = self.binary_to_platform(state_bin)
            
            print(f"▲ Plataforma atual: {new_platform}", end='\r')
            attempts += 1
        
        final_platform = self.binary_to_platform(state_bin)
        print(f"\n🎯 Posicionado na plataforma {final_platform} (alvo: {target_platform})")
        return state_bin

    def train(self, socket_conn):
        platform = self.base_starting_platform
        total_success = 0

        self.current_starting_platform = platform
        platform_success = 0
        print(f"\n🏁 INICIANDO PLATAFORMA {platform} (500 tentativas)")

        for attempt in range(1, self.platform_max_attempts + 1):
            try:
                state_bin = self.move_to_starting_platform(socket_conn, platform)
                state_idx = self.state_to_index(state_bin)
                action_history = []
                same_state_count = 0
                steps = 0

                while True:
                    action = self.choose_action(state_idx, action_history)
                    action_history.append(action)

                    if len(action_history) > 5:
                        action_history.pop(0)

                    next_state_bin, reward = get_state_reward(socket_conn, action)
                    next_state_idx = self.state_to_index(next_state_bin)
                    steps += 1

                    current_platform = self.binary_to_platform(state_bin)
                    next_platform = self.binary_to_platform(next_state_bin)

                    if next_platform > current_platform:
                        reward += 10 * (next_platform - current_platform)
                    elif next_platform < current_platform:
                        reward -= 5

                    if next_state_bin == '0b0000000':
                        break

                    if state_idx == next_state_idx:
                        same_state_count += 1
                        if same_state_count > 2:
                            reward -= 5
                    else:
                        same_state_count = 0

                    self.update_q_table(state_idx, action, reward, next_state_idx)

                    if next_state_bin.startswith('0b10101'):
                        platform_success += 1
                        total_success += 1
                        break

                    state_idx = next_state_idx
                    state_bin = next_state_bin
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    
                    print(f"[Plat.{platform}] Tentativa {attempt}/500 | Passo {steps} | Sucessos: {platform_success}", end='\r')

            except Exception as e:
                print(f"⚠️ Erro: {str(e)}")
                continue

        print(f"\n📊 Plataforma {platform} concluída: {platform_success}/500 sucessos")
        print(f"\n✅ Treinamento completo! Sucessos totais: {total_success}")

    def save_q_table(self):
        np.savetxt(self.qtable_path, self.q_table, fmt="%.6f")
        print(f"💾 Q-table salva em {self.qtable_path}")

def main():
    print("🚀 Iniciando treinamento com 500 tentativas na plataforma alvo...")
    socket_conn = connect(2037)
    if not socket_conn:
        print("🚫 Conexão falhou.")
        return

    try:
        agent = QLearningAgent()
        agent.train(socket_conn)
        agent.save_q_table()
    finally:
        socket_conn.close()
        print("🔌 Conexão encerrada")

if __name__ == "__main__":
    main()
