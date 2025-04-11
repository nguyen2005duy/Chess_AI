import chess


class minmaxAI:
    def __init__(self):
        # neu lag giam cai nay xuong 1
        self.max_depth = 3

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

    def calculate_move(self, board: chess.Board):
        best_move = None
        best_value = float('inf')
        moves = sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True)
        for move in moves:
            board.push(move)
            value = self.maximize(board, float('-inf'), float('inf'), 1, self.max_depth)
            board.pop()

            if value < best_value:
                best_value = value
                best_move = move

        return best_move

    # Tim move co value nho nhat cho ai di chuyen (Random)
    def ai_move(self, board: chess.Board):
        #  board = chess.Board()
        final_move = ["", float('inf')]
        for move in board.legal_moves:
            board.push(move)
            value = self.evaluate_board(board)
            if final_move[1] > value:
                final_move = [move, value]
            board.pop()
        return final_move[0]

    def maximize(self, board: chess.Board, alpha, beta, curr_depth: int, max_depth: int):
        if curr_depth >= max_depth:
            return self.evaluate_board(board)
        max_value = float('-inf')
        moves = sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True)
        for move in moves:
            board.push(move)
            max_value = max(max_value, self.minimize(board, alpha, beta, curr_depth + 1, max_depth))
            board.pop()
            if max_value >= beta:
                return max_value
            alpha = max(alpha, max_value)
        return max_value

    def minimize(self, board: chess.Board, alpha, beta, curr_depth: int, max_depth: int):
        if curr_depth >= max_depth:
            return self.evaluate_board(board)
        min_value = float('inf')
        moves = sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True)
        for move in moves:
            board.push(move)
            min_value = min(min_value, self.maximize(board, alpha, beta, curr_depth + 1, max_depth))
            board.pop()
            if min_value <= alpha:
                return min_value
            beta = min(beta, min_value)
        return min_value
