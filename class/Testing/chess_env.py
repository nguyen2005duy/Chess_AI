### chess_env.py
import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.get_state()

    def get_state(self):
        return self.board.fen()

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def step(self, move):
        reward = 0
        capture = self.board.is_capture(move)
        self.board.push(move)

        # Check for game over
        if self.board.is_game_over():
            result = self.board.result()
            if result == '1-0':  # White wins
                reward = 1
            elif result == '0-1':  # Black wins
                reward = -1
            else:  # Draw
                reward = 0
            done = True
        elif capture:  # Reward for capturing a piece
            reward = 0.1
            done = False
        elif self.board.is_checkmate():  # Handle checkmate state if relevant
            reward = -1
            done = True
        else:
            done = False

        return self.get_state(), reward, done

    def render(self):
        print(self.board)
