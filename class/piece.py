import os
import chess


class WrappedPiece:
    def __init__(self, chess_piece, row, col):
        self.chess_piece = chess_piece  # python-chess piece
        self.row = row
        self.col = col
        self.color = 'white' if chess_piece.color else 'black'
        self.name = self.get_piece_name(chess_piece)
        self.value = self.get_piece_value()
        self.moved = False
        self.set_texture()
        self.texture_rect = None

    def get_piece_name(self, piece):
        piece_map = {
            chess.PAWN: 'pawn',
            chess.KNIGHT: 'knight',
            chess.BISHOP: 'bishop',
            chess.ROOK: 'rook',
            chess.QUEEN: 'queen',
            chess.KING: 'king'
        }
        return piece_map[piece.piece_type]

    def get_piece_value(self):
        value_map = {
            'pawn': 1.0,
            'knight': 3.0,
            'bishop': 3.001,
            'rook': 5.0,
            'queen': 9.0,
            'king': 100000
        }
        sign = 1 if self.color == 'white' else -1
        return sign * value_map[self.name]

    def set_texture(self, size=80):
        self.texture = os.path.join(
            f'../assets/images/imgs-{size}px/{self.color}_{self.name}.png'
        )

    def add_moves(self, move):
        self.moves.append(move)
