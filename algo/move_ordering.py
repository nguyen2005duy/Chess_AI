import chess
from .bitboards import (
    WHITE_PAWNS, WHITE_KNIGHTS, WHITE_BISHOPS, WHITE_ROOKS, WHITE_QUEENS, WHITE_KINGS,
    BLACK_PAWNS, BLACK_KNIGHTS, BLACK_BISHOPS, BLACK_ROOKS, BLACK_QUEENS, BLACK_KINGS,
    WHITE_PIECES, BLACK_PIECES, ALL_PIECES, SIDE_TO_MOVE
)
from . import bitboards  # Also import the full module for accessing EN_PASSANT_SQUARE

# --- Move Ordering Heuristics ---

# Constants for move scoring with even stronger prioritization
TT_MOVE_SCORE = 100000  # Highest priority - significantly increased
PROMOTION_BASE_SCORE = 90000  # Queen promotion adds to this
CAPTURE_BASE_SCORE = 80000  # Base score for all captures
KILLER_FIRST_SCORE = 25000  # Increased killer move importance
KILLER_SECOND_SCORE = 24000
COUNTER_MOVE_SCORE = 23000  # Increased counter move importance
CASTLE_MOVE_SCORE = 22000

# Lower priority bonuses - scaled down relative to the core priorities above
CENTER_CONTROL_BONUS = 1000
PAWN_ADVANCE_BONUS = 500
PIECE_DEVELOPMENT_BONUS = 750
THREAT_CREATION_BONUS = 600
HISTORY_SCALE_FACTOR = 10  # Scaling factor to properly balance history scores

# Center squares definition (e4, d4, e5, d5)
CENTER_SQUARES = {28, 29, 36, 37}
# Extended center (c3-f3, c6-f6, c3-c6, f3-f6)
EXTENDED_CENTER = {18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45}

# Killer moves storage: stores two killer moves per ply
KILLER_MOVES = [[None, None] for _ in range(64)]  # Max depth 64

# Counter move table: move key -> counter_move
COUNTER_MOVES = {}

# History heuristic table: [piece_type][to_square]
HISTORY_HEURISTIC = [[0] * 64 for _ in range(7)]  # 0-indexed piece types

# Butterfly history: [piece_type][from_to] for more accurate history
BUTTERFLY_HISTORY = [[0] * 4096 for _ in range(7)]  # 64*64 possible from-to combinations

# Most recent best move at each depth
BEST_MOVES_BY_POSITION = {}

# Precomputed piece values for MVV-LVA calculation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000
}


# --- Helper Functions ---

def get_piece_type_at(square):
    """
    Fast helper function to get piece type at a square using bitboards.
    """
    bb = 1 << square

    # Early exit for empty squares
    if not (ALL_PIECES & bb):
        return None

    # Check each piece type
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


def get_piece_color_at(square):
    """
    Fast helper function to get piece color at a square using bitboards.
    """
    bb = 1 << square

    if WHITE_PIECES & bb:
        return chess.WHITE
    if BLACK_PIECES & bb:
        return chess.BLACK
    return None


def is_capture(move):
    """
    Determines if a move is a capture using bitboards.
    """
    to_square = move.to_square
    from_square = move.from_square
    to_bb = 1 << to_square
    opponent_pieces = BLACK_PIECES if SIDE_TO_MOVE == chess.WHITE else WHITE_PIECES

    # Standard capture check
    if (opponent_pieces & to_bb) != 0:
        return True

    # En passant capture check
    if hasattr(bitboards, 'EN_PASSANT_SQUARE') and bitboards.EN_PASSANT_SQUARE == to_square:
        # Check if moving piece is a pawn
        if SIDE_TO_MOVE == chess.WHITE:
            return (WHITE_PAWNS & (1 << from_square)) != 0
        else:
            return (BLACK_PAWNS & (1 << from_square)) != 0

    return False


def get_position_hash():
    """
    Create a simple hash of the current position for indexing the BEST_MOVES_BY_POSITION table.
    This is a simplified version that uses the main bitboards.
    """
    # For simplicity, just use the main piece bitboards
    h = 0
    h ^= WHITE_PAWNS
    h ^= WHITE_KNIGHTS * 3
    h ^= WHITE_BISHOPS * 5
    h ^= WHITE_ROOKS * 7
    h ^= WHITE_QUEENS * 11
    h ^= WHITE_KINGS * 13
    h ^= BLACK_PAWNS * 17
    h ^= BLACK_KNIGHTS * 19
    h ^= BLACK_BISHOPS * 23
    h ^= BLACK_ROOKS * 29
    h ^= BLACK_QUEENS * 31
    h ^= BLACK_KINGS * 37
    h ^= int(SIDE_TO_MOVE) * 41

    # Include en passant square if available
    if hasattr(bitboards, 'EN_PASSANT_SQUARE') and bitboards.EN_PASSANT_SQUARE is not None:
        h ^= bitboards.EN_PASSANT_SQUARE * 43

    return h & 0xFFFFFFFF  # Keep it a reasonable size


def get_mvv_lva_score(move):
    """
    Calculate Most Valuable Victim - Least Valuable Aggressor score.
    Higher values for capturing valuable pieces with less valuable ones.
    """
    to_square = move.to_square
    from_square = move.from_square

    # Get piece types
    victim_type = get_piece_type_at(to_square)
    attacker_type = get_piece_type_at(from_square)

    # For en passant, the victim is always a pawn but on a different square
    if hasattr(bitboards, 'EN_PASSANT_SQUARE') and to_square == bitboards.EN_PASSANT_SQUARE:
        victim_type = chess.PAWN

    # Get values
    victim_value = PIECE_VALUES.get(victim_type, 0) if victim_type else 0
    attacker_value = PIECE_VALUES.get(attacker_type, 0) if attacker_type else 0

    # MVV-LVA formula: 10 * victim_value - attacker_value
    # This ensures capturing a queen with a pawn scores higher than with a rook
    return 10 * victim_value - attacker_value


def score_move(move, ply, tt_move_hint):
    """
    Assign a score to a move for sorting in search.
    Higher scores are tried first.
    """
    # Handle null move or invalid move
    if move is None or move.from_square == move.to_square:
        return -1000000  # Very low score

    # 1. TT move gets highest priority (massively increased priority)
    if tt_move_hint and move == tt_move_hint:
        return TT_MOVE_SCORE

    # Check if this move was previously best in the same position (huge boost)
    pos_hash = get_position_hash()
    if pos_hash in BEST_MOVES_BY_POSITION and BEST_MOVES_BY_POSITION[pos_hash] == move:
        return TT_MOVE_SCORE - 1  # Almost as high as TT move

    # 2. Promotions with enhanced scoring
    if move.promotion:
        promo_bonus = {
            chess.QUEEN: 900,
            chess.ROOK: 500,
            chess.BISHOP: 330,
            chess.KNIGHT: 320
        }.get(move.promotion, 0)
        return PROMOTION_BASE_SCORE + promo_bonus

    # 3. Captures with improved MVV-LVA scoring
    if is_capture(move):
        mvv_lva = get_mvv_lva_score(move)
        return CAPTURE_BASE_SCORE + mvv_lva

    # 4. Killer moves (important quiet moves)
    if ply < len(KILLER_MOVES):
        if move == KILLER_MOVES[ply][0]:
            return KILLER_FIRST_SCORE
        elif move == KILLER_MOVES[ply][1]:
            return KILLER_SECOND_SCORE

    # 5. Counter moves (responses to opponent's last move)
    from_to_key = (move.from_square, move.to_square)
    if from_to_key in COUNTER_MOVES:
        return COUNTER_MOVE_SCORE

    # Extract move information for further scoring
    from_square = move.from_square
    to_square = move.to_square
    moving_piece_type = get_piece_type_at(from_square)
    score = 0

    # 6. Castling (strategic priority for king safety)
    if moving_piece_type == chess.KING and abs(from_square - to_square) == 2:
        return CASTLE_MOVE_SCORE

    # 7. History heuristic scoring (highly scaled down relative to main categories)
    if moving_piece_type is not None:
        # Use history tables with careful scaling
        history_score = HISTORY_HEURISTIC[moving_piece_type][to_square]
        butterfly_idx = (from_square << 6) | to_square
        butterfly_score = BUTTERFLY_HISTORY[moving_piece_type][butterfly_idx]

        # Scale history scores much lower than other priorities
        score += (history_score + butterfly_score * 2) // HISTORY_SCALE_FACTOR

        # Center control (minimal bonus)
        if to_square in CENTER_SQUARES:
            score += 300
        elif to_square in EXTENDED_CENTER:
            score += 100

        # Minor piece development in early game (very small bonus)
        if moving_piece_type in (chess.KNIGHT, chess.BISHOP):
            # Simple development check - moving to more active square
            from_rank = from_square // 8
            from_file = from_square % 8
            to_rank = to_square // 8
            to_file = to_square % 8

            # Moving from back rank toward center (minimal development bonus)
            if (SIDE_TO_MOVE == chess.WHITE and from_rank == 0 and to_rank > from_rank) or \
                    (SIDE_TO_MOVE == chess.BLACK and from_rank == 7 and to_rank < from_rank):
                score += 200

    return score


def order_moves(move_list, ply, tt_move_hint=None):
    """
    Order moves for alpha-beta search efficiency.
    """
    if not move_list:
        return []

    # Quick returns for small move lists
    if len(move_list) == 1:
        return move_list.copy()

    # Special handling to ensure TT move is first
    ordered_moves = []
    remaining_moves = []

    # Always try TT move first
    if tt_move_hint in move_list:
        ordered_moves.append(tt_move_hint)
        remaining_moves = [m for m in move_list if m != tt_move_hint]
    else:
        remaining_moves = move_list.copy()

    # Score remaining moves and sort by score
    scored_moves = [(move, score_move(move, ply, tt_move_hint)) for move in remaining_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    # Add sorted moves to ordered list
    ordered_moves.extend(move for move, _ in scored_moves)

    return ordered_moves


def update_killer_moves(move, ply):
    """
    Store a quiet move that caused a beta cutoff.
    """
    if move is None or ply >= len(KILLER_MOVES) or is_capture(move) or move.promotion:
        return

    # Skip if already first killer
    if move == KILLER_MOVES[ply][0]:
        return

    # Shift existing killer and add new one
    KILLER_MOVES[ply][1] = KILLER_MOVES[ply][0]
    KILLER_MOVES[ply][0] = move


def update_counter_move(prev_move, move):
    """
    Store counter moves - responses that caused cutoffs.
    """
    if prev_move is None or move is None or is_capture(move) or move.promotion:
        return

    # Create a key based on the previous move's from-to squares
    key = (prev_move.from_square, prev_move.to_square)
    COUNTER_MOVES[key] = move


def update_history_heuristic(move, depth):
    """
    Update history heuristic tables for quiet moves that caused cutoffs.
    """
    if move is None or is_capture(move) or move.promotion:
        return

    piece_type = get_piece_type_at(move.from_square)
    if piece_type is None:
        return

    # Calculate bonus based on depth squared
    history_bonus = min(depth * depth, 400)
    to_square = move.to_square
    from_square = move.from_square

    # Update regular history
    HISTORY_HEURISTIC[piece_type][to_square] += history_bonus

    # Update butterfly history (more specific from-to combination)
    butterfly_idx = (from_square << 6) | to_square
    BUTTERFLY_HISTORY[piece_type][butterfly_idx] += history_bonus * 2

    # Age all history values
    if history_bonus > 0 and depth > 3:
        for pt in range(1, 7):  # All piece types
            for sq in range(64):  # All squares
                HISTORY_HEURISTIC[pt][sq] = HISTORY_HEURISTIC[pt][sq] * 253 // 256

        # Age butterfly history selectively
        for pt in range(1, 7):
            for idx in range(4096):
                if BUTTERFLY_HISTORY[pt][idx] > 0:
                    BUTTERFLY_HISTORY[pt][idx] = BUTTERFLY_HISTORY[pt][idx] * 253 // 256


def update_best_move_in_position(move, depth):
    """
    Store the best move found in a paosition.
    """
    if move is None or depth < 2:  # Only store moves from reasonable depths
        return

    pos_hash = get_position_hash()
    BEST_MOVES_BY_POSITION[pos_hash] = move

    # Limit table size
    if len(BEST_MOVES_BY_POSITION) > 100000:
        # Simple cleanup - just clear oldest entries
        keys = list(BEST_MOVES_BY_POSITION.keys())
        for i in range(len(keys) // 2):  # Remove half the entries
            del BEST_MOVES_BY_POSITION[keys[i]]


def clear_history_tables():
    """
    Clear history tables between games or periodically.
    """
    global HISTORY_HEURISTIC, BUTTERFLY_HISTORY, COUNTER_MOVES, BEST_MOVES_BY_POSITION

    HISTORY_HEURISTIC = [[0] * 64 for _ in range(7)]
    BUTTERFLY_HISTORY = [[0] * 4096 for _ in range(7)]
    COUNTER_MOVES = {}
    BEST_MOVES_BY_POSITION = {}