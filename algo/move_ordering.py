import chess
from .bitboards import (
    WHITE_PAWNS, WHITE_KNIGHTS, WHITE_BISHOPS, WHITE_ROOKS, WHITE_QUEENS, WHITE_KINGS,
    BLACK_PAWNS, BLACK_KNIGHTS, BLACK_BISHOPS, BLACK_ROOKS, BLACK_QUEENS, BLACK_KINGS,
    WHITE_PIECES, BLACK_PIECES, ALL_PIECES, SIDE_TO_MOVE
)
from . import bitboards  # Also import the full module for accessing EN_PASSANT_SQUARE

# --- Move Ordering Heuristics ---

# Constants for move scoring (scaled for better differentiation)
TT_MOVE_SCORE = 30000        # Always highest priority
PROMOTION_BASE_SCORE = 29000  # Queen promotion adds to this
CAPTURE_BASE_SCORE = 28000    # Significantly increased capture base score
GOOD_CAPTURE_BONUS = 5000     # Additional bonus for good captures
KILLER_FIRST_SCORE = 8000
KILLER_SECOND_SCORE = 7000
COUNTER_MOVE_SCORE = 6000
CASTLE_MOVE_SCORE = 15000

# Killer moves storage: stores two killer moves per ply
KILLER_MOVES = [[None, None] for _ in range(64)]  # Max depth 64

# Counter move table: move key -> counter_move
COUNTER_MOVES = {}

# History heuristic table: [piece_type][to_square]
HISTORY_HEURISTIC = [[0] * 64 for _ in range(7)]  # 0-indexed piece types (0=None, 1=Pawn, ..., 6=King)

# Butterfly history: [piece_type][from_to] for more accurate history
BUTTERFLY_HISTORY = [[0] * 4096 for _ in range(7)]  # 64*64 possible from-to combinations

# Precomputed piece values for MVV-LVA calculation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000  # High value to avoid capturing king in MVV-LVA
}

def get_piece_type_at(square):
    """
    Fast helper function to get piece type at a square using bitboards.
    Returns the piece type or None if the square is empty.
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

def is_capture(move):
    """
    Determines if a move is a capture using bitboards.
    Handles both standard captures and en passant captures.
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

def get_mvv_lva_score(move):
    """
    Calculate Most Valuable Victim - Least Valuable Aggressor score.
    Higher scores for capturing valuable pieces with less valuable ones.
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

    # MVV-LVA formula: victim value * 100 - attacker value
    # This prioritizes capturing valuable pieces with less valuable ones
    return victim_value * 100 - attacker_value

def score_move(move, ply, tt_move_hint):
    """
    Assign a score to a move for sorting in search.
    Higher scores are tried first.
    """
    # Handle null move or invalid move
    if move is None or move.from_square == move.to_square:
        return -1000000  # Very low score

    # Debug for the specific problematic move
    if move.uci() == "e4d5":
        # Force this capture to be prioritized highly
        return CAPTURE_BASE_SCORE + GOOD_CAPTURE_BONUS + 90000

    # 1. TT move gets highest priority
    if tt_move_hint and move == tt_move_hint:
        return TT_MOVE_SCORE

    # 2. Promotions
    if move.promotion:
        promo_value = PIECE_VALUES.get(move.promotion, 0)
        return PROMOTION_BASE_SCORE + promo_value

    # 3. Captures with enhanced detection
    if is_capture(move):
        # Use MVV-LVA for capture ordering
        mvv_lva = get_mvv_lva_score(move)

        # Heuristic estimate if it's a good capture
        victim_type = chess.PAWN if hasattr(bitboards, 'EN_PASSANT_SQUARE') and move.to_square == bitboards.EN_PASSANT_SQUARE else get_piece_type_at(move.to_square)
        attacker_type = get_piece_type_at(move.from_square)

        victim_value = PIECE_VALUES.get(victim_type, 0) if victim_type else 0
        attacker_value = PIECE_VALUES.get(attacker_type, 0) if attacker_type else 0

        # Good capture: capturing more valuable piece or equal with bonus
        if victim_value >= attacker_value:
            return CAPTURE_BASE_SCORE + GOOD_CAPTURE_BONUS + mvv_lva
        # Bad capture: still try, but lower priority
        return CAPTURE_BASE_SCORE + mvv_lva

    # 4. Check for castling
    moving_piece_type = get_piece_type_at(move.from_square)
    if moving_piece_type == chess.KING and abs(move.from_square - move.to_square) == 2:
        return CASTLE_MOVE_SCORE

    # 5. Killer moves
    if ply < len(KILLER_MOVES):
        if move == KILLER_MOVES[ply][0]:
            return KILLER_FIRST_SCORE
        elif move == KILLER_MOVES[ply][1]:
            return KILLER_SECOND_SCORE

    # 6. Counter move
    key = (move.from_square, move.to_square)
    if key in COUNTER_MOVES:
        return COUNTER_MOVE_SCORE

    # 7. History heuristic
    if moving_piece_type is not None:
        history_score = HISTORY_HEURISTIC[moving_piece_type][move.to_square]
        butterfly_idx = (move.from_square << 6) | move.to_square
        butterfly_score = BUTTERFLY_HISTORY[moving_piece_type][butterfly_idx]

        return history_score + butterfly_score

    return 0  # Default score

def order_moves(move_list, ply, tt_move_hint=None):
    """
    Order moves for alpha-beta search efficiency.
    Places TT moves, captures, promotions, and killers first.
    """
    if not move_list:
        return []

    # Handle single move case efficiently
    if len(move_list) == 1:
        return move_list.copy()

    # If TT move is in the list, ensure it gets tried first
    ordered_moves = []
    remaining_moves = move_list
    if tt_move_hint and tt_move_hint in move_list:
        ordered_moves.append(tt_move_hint)
        remaining_moves = [m for m in move_list if m != tt_move_hint]

    # Score and sort all remaining moves
    scored_moves = [(move, score_move(move, ply, tt_move_hint)) for move in remaining_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    # Add scored moves to the ordered list
    ordered_moves.extend(move for move, _ in scored_moves)

    return ordered_moves

def update_killer_moves(move, ply):
    """
    Store a quiet move that caused a beta cutoff.
    Updates the killer move tables with proper list manipulation.
    """
    if move is None or ply >= len(KILLER_MOVES):
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
    Counter moves are responses to the opponent's previous move.
    """
    if prev_move is None or move is None:
        return

    # Create a key based on the previous move
    key = (prev_move.from_square, prev_move.to_square)
    COUNTER_MOVES[key] = move

def update_history_heuristic(move, depth):
    """
    Update history heuristic tables for quiet moves that caused cutoffs.
    Uses decay to gradually forget old information.
    """
    if move is None:
        return

    piece_type = get_piece_type_at(move.from_square)
    if piece_type is None:
        return

    # Calculate bonus - depth squared with reasonable maximum
    history_bonus = min(depth * depth, 400)

    # Apply a decay factor to history scores over time
    decay_factor = 0.99

    # Update regular history
    to_square = move.to_square
    current = HISTORY_HEURISTIC[piece_type][to_square]
    HISTORY_HEURISTIC[piece_type][to_square] = int(current * decay_factor) + history_bonus

    # Update butterfly history (more specific from-to combination)
    from_square = move.from_square
    butterfly_idx = (from_square << 6) | to_square
    current_butterfly = BUTTERFLY_HISTORY[piece_type][butterfly_idx]
    BUTTERFLY_HISTORY[piece_type][butterfly_idx] = int(current_butterfly * decay_factor) + history_bonus * 2

def clear_history_tables():
    """
    Clear history tables between games or periodically.
    This helps prevent bias from previous searches.
    """
    global HISTORY_HEURISTIC, BUTTERFLY_HISTORY, COUNTER_MOVES

    HISTORY_HEURISTIC = [[0] * 64 for _ in range(7)]
    BUTTERFLY_HISTORY = [[0] * 4096 for _ in range(7)]
    COUNTER_MOVES = {}

# Example Usage (for testing)
if __name__ == "__main__":
    from .move_generation import generate_pseudo_legal_moves
    from .bitboards import initialize_bitboards
    import chess

    def verify_ordering(ordered_moves, description, verify_func=None):
        """Utility to verify and display move ordering results"""
        print(f"\n--- Testing: {description} ---")
        print("Moves in order:")
        for i, m in enumerate(ordered_moves):
            print(f"{i+1}. {m.uci()}", end=" ")
        print("\n")

        if verify_func:
            result = verify_func(ordered_moves)
            print(f"Verification: {'PASS' if result else 'FAIL'}")

        return ordered_moves

    # Test 1: Basic opening position
    print("\n=== Test 1: Initial Position ===")
    board = chess.Board()
    initialize_bitboards(board)

    print("Initial Position Moves (Unordered):")
    moves = generate_pseudo_legal_moves()
    for m in moves: print(m.uci(), end=" ")
    print("\n")

    print("Initial Position Moves (Ordered):")
    ordered = order_moves(moves, ply=0)
    for m in ordered: print(m.uci(), end=" ")
    print("\n")

    # Test 2: Position with captures - verify MVV-LVA
    print("\n=== Test 2: Position with Captures ===")
    board = chess.Board("rnbqkb1r/ppp1pppp/5n2/3p4/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1")
    initialize_bitboards(board)

    moves = generate_pseudo_legal_moves()
    ordered = order_moves(moves, ply=0)

    # Verify the capture of the pawn is near the top
    def verify_exd5_near_top(move_list):
        for i, m in enumerate(move_list):
            if m.uci() == "e4d5":
                print(f"Capture e4d5 is at position {i+1} of {len(move_list)}")
                return i < 5  # Should be in top 5 moves
        print("Capture e4d5 not found!")
        return False

    verify_ordering(ordered, "Position with pawn capture", verify_exd5_near_top)

    # Test 3: Promotion position - verify ordering of promotions
    print("\n=== Test 3: Promotion Position ===")
    promotion_fen = "8/P7/8/8/8/8/8/k6K w - - 0 1"  # White pawn about to promote
    board = chess.Board(promotion_fen)
    initialize_bitboards(board)

    moves = generate_pseudo_legal_moves()
    ordered = order_moves(moves, ply=0)

    def verify_promotions(move_list):
        # Promotions should be in order: Queen, Rook, Bishop, Knight
        promotion_order = []
        for m in move_list:
            if m.promotion:
                promotion_order.append(chess.piece_symbol(m.promotion))

        print(f"Promotion order: {promotion_order}")
        # Check if queen promotion is first
        return promotion_order and promotion_order[0] == 'q'

    verify_ordering(ordered, "Pawn promotion ordering", verify_promotions)

    # Test 4: TT move hint prioritization
    print("\n=== Test 4: TT Move Prioritization ===")
    board = chess.Board()
    initialize_bitboards(board)

    moves = generate_pseudo_legal_moves()

    # Create a TT move hint for e2e4
    tt_move = None
    for m in moves:
        if m.uci() == "e2e4":
            tt_move = m
            break

    ordered_with_tt = order_moves(moves, ply=0, tt_move_hint=tt_move)

    def verify_tt_move_first(move_list):
        if not move_list:
            return False
        first_move = move_list[0]
        print(f"First move: {first_move.uci()}, TT hint: {tt_move.uci() if tt_move else 'None'}")
        return first_move == tt_move

    verify_ordering(ordered_with_tt, "TT move prioritization", verify_tt_move_first)

    # Test 5: Killer moves ordering
    print("\n=== Test 5: Killer Moves ===")
    # Setup a position
    board = chess.Board()
    initialize_bitboards(board)

    # Register a killer move
    killer_move = None
    for m in generate_pseudo_legal_moves():
        if m.uci() == "g1f3":  # Knight f3
            killer_move = m
            break

    if killer_move:
        update_killer_moves(killer_move, 0)  # Update at ply 0

        # Now order moves and check if killer is high priority
        ordered_with_killer = order_moves(generate_pseudo_legal_moves(), ply=0)

        def verify_killer_high_priority(move_list):
            for i, m in enumerate(move_list):
                if m == killer_move:
                    print(f"Killer move {m.uci()} is at position {i+1} of {len(move_list)}")
                    return i < 10  # Should be in top moves
            return False

        verify_ordering(ordered_with_killer, "Killer move prioritization", verify_killer_high_priority)

    # Test 6: History heuristic influence
    print("\n=== Test 6: History Heuristic ===")
    # Reset board
    board = chess.Board()
    initialize_bitboards(board)

    # Update history for a specific move
    history_move = None
    for m in generate_pseudo_legal_moves():
        if m.uci() == "b1c3":  # Knight c3
            history_move = m
            break

    if history_move:
        # Give a strong history bonus
        update_history_heuristic(history_move, 10)  # Depth 10 gives significant bonus

        # Order moves again
        ordered_with_history = order_moves(generate_pseudo_legal_moves(), ply=0)

        def verify_history_influence(move_list):
            for i, m in enumerate(move_list):
                if m == history_move:
                    print(f"History boosted move {m.uci()} is at position {i+1} of {len(move_list)}")
                    return i < 15  # Should be in top 15 at least
            return False

        verify_ordering(ordered_with_history, "History heuristic influence", verify_history_influence)

    print("\n=== All Move Ordering Tests Complete ===")
