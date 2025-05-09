import torch
import torch.nn as nn
import numpy as np
import chess


class QNetwork(nn.Module):
    def __init__(self, input_size=832, hidden_sizes=[1024, 512, 256], output_size=1747):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.output(x)
        return x


class ChessModel:
    def __init__(self, model_path='models/best_move_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = QNetwork()
        self.load_model(model_path)
        self.q_network.eval()  # Set to evaluation mode

    def load_model(self, path):
        try:
            self.q_network.load_state_dict(torch.load(path, map_location=self.device))
            self.q_network.to(self.device)
            print(f"Model successfully loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing with untrained model")

    def board_to_tensor(self, board):
        """
        Convert a python-chess board to a tensor representation compatible with the network.
        """
        feature = np.zeros((8, 8, 13), dtype=np.float32)
        piece_to_index = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        # Fill in piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - (square // 8)  # Flip row to match original implementation
                col = square % 8
                piece_type = piece.piece_type
                color_offset = 0 if piece.color == chess.WHITE else 6
                feature[row, col, piece_to_index[piece_type] + color_offset] = 1

        # Set active color layer
        active_color = 1 if board.turn == chess.WHITE else 0
        feature[:, :, 12] = active_color

        return torch.tensor(feature.flatten(), dtype=torch.float32).to(self.device)

    def encode_move(self, move):
        """
        Encodes a python-chess move into a unique index that works with the best_move_model.

        This encoding may need adjustment to match how the model was trained.
        """
        from_square = move.from_square
        to_square = move.to_square

        # Simple encoding for the 1747 output size (likely just from_square * 64 + to_square)
        # This assumes no promotion handling for simplicity
        move_index = from_square * 64 + to_square

        # Make sure we don't exceed the output size
        if move_index >= 1747:
            move_index = 0  # Default to first move if out of bounds

        return move_index

    def get_move_scores(self, board, legal_moves=None):
        """
        Returns a list of scores for each legal move.
        """
        if legal_moves is None:
            legal_moves = list(board.legal_moves)

        if not legal_moves:
            return []

        with torch.no_grad():
            board_tensor = self.board_to_tensor(board)
            q_values = self.q_network(board_tensor)

            move_scores = []
            for move in legal_moves:
                move_index = self.encode_move(move)
                # Make sure we don't exceed array bounds
                if move_index < q_values.size(0):
                    score = q_values[move_index].item()
                    move_scores.append((move, score))
                else:
                    # Fallback for any move that can't be properly encoded
                    move_scores.append((move, 0))

            return move_scores

    def get_best_move(self, board):
        """
        Returns the best move according to the Q-network.
        """
        legal_moves = list(board.legal_moves)
        move_scores = self.get_move_scores(board, legal_moves)

        if not move_scores:
            return None

        # Sort by score in descending order
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[0][0]  # Return the move with the highest score