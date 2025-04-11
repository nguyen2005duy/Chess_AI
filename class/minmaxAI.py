import chess


class minmaxAI:
    def __init__(self):
        pass

    def evaluate_board(self, board):
        # Material values
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.1,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100000
        }

        value = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                sign = 1 if piece.color == chess.WHITE else -1
                value += sign * piece_values[piece.piece_type]

        return value

    def calculate_move(self):
        pass

    # Tim move co value nho nhat cho ai di chuyen
    def ai_move(self, board):
        #  board = chess.Board()
        final_move = ["", float('inf')]
        for move in board.legal_moves:
            board.push(move)
            value = self.evaluate_board(board)
            if final_move[1] > value:
                final_move = [move, value]
            board.pop()
        return final_move[0]
