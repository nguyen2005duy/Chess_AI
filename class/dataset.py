# File 2: dataset.py
import chess
import chess.pgn
import numpy as np


def board_to_input(board):
    """Convert python-chess board to neural network input tensor"""
    input_tensor = np.zeros((8, 8, 18), dtype=np.float32)

    # Piece channels (0-11)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_types.index(piece.piece_type) + (6 if piece.color else 0)
            row, col = divmod(square, 8)
            input_tensor[7 - row][col][channel] = 1.0

    # Game state channels (12-17)
    # Channel 12: Current player color (1 for white, 0 for black)
    input_tensor[:, :, 12] = 1.0 if board.turn else 0.0

    # Castling rights
    input_tensor[:, :, 13] = int(board.has_kingside_castling_rights(chess.WHITE))
    input_tensor[:, :, 14] = int(board.has_queenside_castling_rights(chess.WHITE))
    input_tensor[:, :, 15] = int(board.has_kingside_castling_rights(chess.BLACK))
    input_tensor[:, :, 16] = int(board.has_queenside_castling_rights(chess.BLACK))

    # En passant
    if board.ep_square:
        ep_row = 7 - chess.square_rank(board.ep_square)
        ep_col = chess.square_file(board.ep_square)
        input_tensor[ep_row][ep_col][17] = 1.0

    return input_tensor


def move_to_policy(move, board):
    """Convert chess move to policy vector index"""
    from_square = move.from_square
    to_square = move.to_square

    dx = chess.square_file(to_square) - chess.square_file(from_square)
    dy = chess.square_rank(to_square) - chess.square_rank(from_square)

    # Handle promotions
    if move.promotion:
        promotion_type = move.promotion - 1
        direction = 64 + (dy + 1) * 3 + (dx + 1)
        return direction * 9 + promotion_type

    # Queen moves
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    for d, (ddx, ddy) in enumerate(directions):
        if dx == ddx and dy == ddy:
            distance = max(abs(dx), abs(dy)) - 1
            return d * 56 + distance

    # Knight moves
    knight_moves = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    for i, (kdx, kdy) in enumerate(knight_moves):
        if dx == kdx and dy == kdy:
            return 448 + i

    return None