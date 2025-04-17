import torch
import torch.optim as optim
import numpy as np
import chess
import os
import chess_env
import model1


class Trainer:
    def __init__(self, model, env, episodes=100, lr=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.model = model
        self.env = env
        self.episodes = episodes
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.rewards = []
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        if not os.path.exists('models'):
            os.makedirs('models')

    def move_to_index(self, move):
        """
        Map a move to an index from 0 to 19, assuming there are 20 possible moves.
        """
        # Ensure this function generates an index within the range of the model's output
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion if move.promotion else 0
        index = from_square * 73 + to_square + (promotion * 64)

        # Ensure that the index is within the bounds of the model's action space (0-19)
        return index % 20  # We ensure the index is within range 0-19, assuming 20 actions

    def index_to_move(self, index, legal_moves):
        """
        Map a predicted index to a legal move.
        """
        for move in legal_moves:
            if self.move_to_index(move) == index:
                return move
        return np.random.choice(legal_moves)  # fallback if invalid index

    def train(self):
        print("Training started...")
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state_input = self.fen_to_input(state)
                state_input = torch.tensor(state_input, dtype=torch.float32).unsqueeze(0)

                legal_moves = self.env.get_legal_moves()
                legal_indices = [self.move_to_index(move) for move in legal_moves]

                # Epsilon-greedy
                if np.random.rand() < self.epsilon:
                    move = np.random.choice(legal_moves)
                else:
                    with torch.no_grad():
                        q_values = self.model(state_input).squeeze().numpy()
                        best_index = max(legal_indices, key=lambda idx: q_values[idx])
                        move = self.index_to_move(best_index, legal_moves)

                next_state, reward, done = self.env.step(move)
                total_reward += reward

                next_state_input = self.fen_to_input(next_state)
                next_state_input = torch.tensor(next_state_input, dtype=torch.float32).unsqueeze(0)

                target_q_values = self.model(state_input).clone()
                with torch.no_grad():
                    next_q_values = self.model(next_state_input).squeeze().numpy()

                    # Ensure that the next_q_values are only for legal moves
                    legal_next_q_values = [next_q_values[idx] for idx in legal_indices if idx < len(next_q_values)]

                    # If no legal moves are left (which should not happen), set max_next_q to 0
                    if len(legal_next_q_values) == 0:
                        max_next_q = 0.0
                    else:
                        max_next_q = max(legal_next_q_values) if not done else 0.0

                move_index = self.move_to_index(move)
                target_value = reward + 0.9 * max_next_q
                target_q_values[0][move_index] = target_value

                loss = self.criterion(self.model(state_input), target_q_values.detach())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.rewards.append(total_reward)
            print(
                f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")

            if (episode + 1) % 100 == 0:
                torch.save(self.model.state_dict(), f"models/trained_model_{episode + 1}.pth")
                print(f"Model saved to 'models/trained_model_{episode + 1}.pth'")

        torch.save(self.model.state_dict(), "models/trained_model_final.pth")
        np.savetxt("models/training_rewards.csv", self.rewards, delimiter=",")
        print("Training complete. Final model and rewards saved.")

    def fen_to_input(self, fen):
        board = chess.Board(fen)
        input_vector = np.zeros(64)
        board_str = board.board_fen().replace('/', '')
        i = 0
        for char in board_str:
            if char.isdigit():
                i += int(char)
            else:
                input_vector[i] = 1 if char.isupper() else -1
                i += 1
        return input_vector


# When creating the model, you need to pass the number of legal moves
# Modify the trainer.py to initialize the model like this:

env = chess_env.ChessEnv()
legal_moves = env.get_legal_moves()
num_legal_moves = len(legal_moves)
model = model1.Model1(num_legal_moves)
trainer = Trainer(model, env, episodes=100)
trainer.train()
