# File 4: nn_ai.py
import numpy as np
import chess
from model import create_chess_model, custom_mse, custom_cce
from dataset import board_to_input, move_to_policy
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras import losses
from keras._tf_keras.keras.models import load_model



class NeuralNetworkAI:
    def __init__(self, model_path):
        self.model = load_model(
            model_path,
            custom_objects={
                'custom_mse': custom_mse,
                'custom_cce': custom_cce,
                'mse': losses.MeanSquaredError(),
                'categorical_crossentropy': losses.CategoricalCrossentropy()
            }
        )
        self.temperature = 0.5 # Controls exploration vs exploitation

    def calculate_move(self, board):
        # Convert board to input tensor
        input_tensor = board_to_input(board)
        input_batch = np.expand_dims(input_tensor, axis=0)

        # Get model predictions
        policy_pred, value_pred = self.model.predict(input_batch)
        policy_pred = policy_pred[0]

        # Get legal moves
        legal_moves = list(board.legal_moves)

        # Filter policy for legal moves
        legal_policy = []
        for move in legal_moves:
            idx = move_to_policy(move, board)
            if idx is not None:
                legal_policy.append((move, policy_pred[idx]))

        # Apply temperature scaling
        moves, probs = zip(*legal_policy)
        probs = np.power(probs, 1 / self.temperature)
        probs /= np.sum(probs)

        # Select move
        selected_idx = np.random.choice(len(moves), p=probs)
        return moves[selected_idx]