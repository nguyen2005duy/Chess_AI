import chess
from .bitboards import (
    WHITE_PAWNS, WHITE_KNIGHTS, WHITE_BISHOPS, WHITE_ROOKS, WHITE_QUEENS, WHITE_KINGS,
    BLACK_PAWNS, BLACK_KNIGHTS, BLACK_BISHOPS, BLACK_ROOKS, BLACK_QUEENS, BLACK_KINGS,
    WHITE_PIECES, BLACK_PIECES, ALL_PIECES, SIDE_TO_MOVE
)
from . import bitboards

# Move Ordering Heuristics Constants
TT_MOVE_SCORE = 100000
PROMOTION_BASE_SCORE = 90000
CAPTURE_BASE_SCORE = 80000
KILLER_FIRST_SCORE = 25000
KILLER_SECOND_SCORE = 24000
CASTLE_MOVE_SCORE = 22000

CENTER_CONTROL_BONUS = 1000
PAWN_ADVANCE_BONUS = 500
PIECE_DEVELOPMENT_BONUS = 750
THREAT_CREATION_BONUS = 600
HISTORY_SCALE_FACTOR = 10

CENTER_SQUARES = {28, 29, 36, 37}
EXTENDED_CENTER = {18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45}

KILLER_MOVES = [[None, None] for _ in range(64)]
HISTORY_HEURISTIC = [[0] * 64 for _ in range(7)]
BUTTERFLY_HISTORY = [[0] * 4096 for _ in range(7)]

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000
}


def get_piece_type_at(square):
    bb = 1 << square

    if not (ALL_PIECES & bb):
        return None

    if (WHITE_PAWNS | BLACK_PAWNS) & bb:
        return chess.PAWN
    if (WHITE_KNIGHTS | BLACK_KNIGHTS) & bb:
        return chess.KNIGHT
    if (WHITE_BISHOPS | BLACK_BISHOPS) & bb:
        return chess.BISHOP
    if (WHITE_ROOKS | BLACK_ROOKS) & bb:
        return chess.ROOK
    if (WHITE_QUEENS | BLACK_QUEENS) & bb:
        return chess.QUEEN
    if (WHITE_KINGS | BLACK_KINGS) & bb:
        return chess.KING
    return None


def is_capture(move):
    to_square = move.to_square
    from_square = move.from_square
    to_bb = 1 << to_square
    opponent_pieces = BLACK_PIECES if SIDE_TO_MOVE == chess.WHITE else WHITE_PIECES

    if (opponent_pieces & to_bb) != 0:
        return True

    if hasattr(bitboards, 'EN_PASSANT_SQUARE') and bitboards.EN_PASSANT_SQUARE == to_square:
        if SIDE_TO_MOVE == chess.WHITE:
            return (WHITE_PAWNS & (1 << from_square)) != 0
        else:
            return (BLACK_PAWNS & (1 << from_square)) != 0

    return False


def get_mvv_lva_score(move):
    to_square = move.to_square
    from_square = move.from_square

    victim_type = get_piece_type_at(to_square)
    attacker_type = get_piece_type_at(from_square)

    if hasattr(bitboards, 'EN_PASSANT_SQUARE') and to_square == bitboards.EN_PASSANT_SQUARE:
        victim_type = chess.PAWN

    victim_value = PIECE_VALUES.get(victim_type, 0) if victim_type else 0
    attacker_value = PIECE_VALUES.get(attacker_type, 0) if attacker_type else 0

    return 10 * victim_value - attacker_value


def score_move(move, ply, tt_move_hint):
    if move is None or move.from_square == move.to_square:
        return -1000000

    if tt_move_hint and move == tt_move_hint:
        return TT_MOVE_SCORE

    if move.promotion:
        promo_bonus = {
            chess.QUEEN: 900,
            chess.ROOK: 500,
            chess.BISHOP: 330,
            chess.KNIGHT: 320
        }.get(move.promotion, 0)
        return PROMOTION_BASE_SCORE + promo_bonus

    if is_capture(move):
        mvv_lva = get_mvv_lva_score(move)
        return CAPTURE_BASE_SCORE + mvv_lva

    if ply < len(KILLER_MOVES):
        if move == KILLER_MOVES[ply][0]:
            return KILLER_FIRST_SCORE
        elif move == KILLER_MOVES[ply][1]:
            return KILLER_SECOND_SCORE

    from_square = move.from_square
    to_square = move.to_square
    moving_piece_type = get_piece_type_at(from_square)
    score = 0

    if moving_piece_type == chess.KING and abs(from_square - to_square) == 2:
        return CASTLE_MOVE_SCORE

    if moving_piece_type is not None:
        history_score = HISTORY_HEURISTIC[moving_piece_type][to_square]
        butterfly_idx = (from_square << 6) | to_square
        butterfly_score = BUTTERFLY_HISTORY[moving_piece_type][butterfly_idx]

        score += (history_score + butterfly_score * 2) // HISTORY_SCALE_FACTOR

        if to_square in CENTER_SQUARES:
            score += 300
        elif to_square in EXTENDED_CENTER:
            score += 100

        if moving_piece_type in (chess.KNIGHT, chess.BISHOP):
            from_rank = from_square // 8
            from_file = from_square % 8
            to_rank = to_square // 8
            to_file = to_square % 8

            if (SIDE_TO_MOVE == chess.WHITE and from_rank == 0 and to_rank > from_rank) or \
                    (SIDE_TO_MOVE == chess.BLACK and from_rank == 7 and to_rank < from_rank):
                score += 200

    return score


def order_moves(move_list, ply, tt_move_hint=None):
    if not move_list:
        return []

    if len(move_list) == 1:
        return move_list.copy()

    ordered_moves = []
    remaining_moves = []

    if tt_move_hint in move_list:
        ordered_moves.append(tt_move_hint)
        remaining_moves = [m for m in move_list if m != tt_move_hint]
    else:
        remaining_moves = move_list.copy()

    scored_moves = [(move, score_move(move, ply, tt_move_hint)) for move in remaining_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    ordered_moves.extend(move for move, _ in scored_moves)

    return ordered_moves


def update_killer_moves(move, ply):
    if move is None or ply >= len(KILLER_MOVES) or is_capture(move) or move.promotion:
        return

    if move == KILLER_MOVES[ply][0]:
        return

    KILLER_MOVES[ply][1] = KILLER_MOVES[ply][0]
    KILLER_MOVES[ply][0] = move


def update_history_heuristic(move, depth):
    if move is None or is_capture(move) or move.promotion:
        return

    piece_type = get_piece_type_at(move.from_square)
    if piece_type is None:
        return

    history_bonus = min(depth * depth, 400)
    to_square = move.to_square
    from_square = move.from_square

    HISTORY_HEURISTIC[piece_type][to_square] += history_bonus

    butterfly_idx = (from_square << 6) | to_square
    BUTTERFLY_HISTORY[piece_type][butterfly_idx] += history_bonus * 2

    if history_bonus > 0 and depth > 3:
        for pt in range(1, 7):
            for sq in range(64):
                HISTORY_HEURISTIC[pt][sq] = HISTORY_HEURISTIC[pt][sq] * 253 // 256

        for pt in range(1, 7):
            for idx in range(4096):
                if BUTTERFLY_HISTORY[pt][idx] > 0:
                    BUTTERFLY_HISTORY[pt][idx] = BUTTERFLY_HISTORY[pt][idx] * 253 // 256
