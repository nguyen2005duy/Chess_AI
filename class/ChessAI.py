import chess
from nn_ai import NeuralNetworkAI


class ChessAI:
    def __init__(self, model_path='chess_model.h5'):
        self.ai = NeuralNetworkAI(model_path)

    def calculate_move(self, board):
        return self.ai.calculate_move(board)