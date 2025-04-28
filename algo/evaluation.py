import chess
from . import bitboards
from .magic_bitboards import KING_ATTACKS, KNIGHT_ATTACKS, get_rook_attack, get_bishop_attack

# --- Constants ---
MAX_PLY = 64 # Maximum search depth (shared constant)
MATE_SCORE = 999999 # Increased Mate Score
MATE_THRESHOLD = MATE_SCORE - MAX_PLY # Threshold to identify mate scores
DRAW_SCORE = 0     # Score for draws

# --- Evaluation Components ---

# Material values
MATERIAL_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Piece-Square Tables (PSTs)
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [ #
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

# Mirrored PST for black pieces (flip ranks)
MIRRORED_PST = {}
for piece_type, table in PST.items():
    MIRRORED_PST[piece_type] = [
        table[f + 8 * (7 - r)]
        for r in range(8)
        for f in range(8)
    ]

def calculate_material_score():
    """Calculates the material score based on current bitboards."""
    score = 0
    score += bitboards.WHITE_PAWNS.bit_count() * MATERIAL_VALUES[chess.PAWN]
    score += bitboards.WHITE_KNIGHTS.bit_count() * MATERIAL_VALUES[chess.KNIGHT]
    score += bitboards.WHITE_BISHOPS.bit_count() * MATERIAL_VALUES[chess.BISHOP]
    score += bitboards.WHITE_ROOKS.bit_count() * MATERIAL_VALUES[chess.ROOK]
    score += bitboards.WHITE_QUEENS.bit_count() * MATERIAL_VALUES[chess.QUEEN]
    score -= bitboards.BLACK_PAWNS.bit_count() * MATERIAL_VALUES[chess.PAWN]
    score -= bitboards.BLACK_KNIGHTS.bit_count() * MATERIAL_VALUES[chess.KNIGHT]
    score -= bitboards.BLACK_BISHOPS.bit_count() * MATERIAL_VALUES[chess.BISHOP]
    score -= bitboards.BLACK_ROOKS.bit_count() * MATERIAL_VALUES[chess.ROOK]
    score -= bitboards.BLACK_QUEENS.bit_count() * MATERIAL_VALUES[chess.QUEEN]
    return score

def calculate_pst_score():
    """Calculates the PST score based on current bitboards."""
    score = 0
    for piece_type, table in PST.items():
        bb = getattr(bitboards, f"WHITE_{chess.piece_name(piece_type).upper()}S", 0)
        for square in chess.SquareSet(bb): # Use SquareSet iteration
            score += table[square]
        bb = getattr(bitboards, f"BLACK_{chess.piece_name(piece_type).upper()}S", 0)
        mirror_table = MIRRORED_PST[piece_type]
        for square in chess.SquareSet(bb): # Use SquareSet iteration
            score -= mirror_table[square]
    return score

def calculate_mobility_score():
    """Calculates a basic mobility score based on current bitboards."""
    white_mobility = 0
    black_mobility = 0
    mobility_weight = 5
    for square in chess.SquareSet(bitboards.WHITE_KNIGHTS): # Use SquareSet iteration
        white_mobility += (KNIGHT_ATTACKS[square] & ~bitboards.WHITE_PIECES).bit_count() 
    for square in chess.SquareSet(bitboards.WHITE_BISHOPS): # Use SquareSet iteration
        white_mobility += (get_bishop_attack(square, bitboards.ALL_PIECES) & ~bitboards.WHITE_PIECES).bit_count()
    for square in chess.SquareSet(bitboards.WHITE_ROOKS): # Use SquareSet iteration
        white_mobility += (get_rook_attack(square, bitboards.ALL_PIECES) & ~bitboards.WHITE_PIECES).bit_count()
    for square in chess.SquareSet(bitboards.WHITE_QUEENS): # Use SquareSet iteration
        white_mobility += ((get_rook_attack(square, bitboards.ALL_PIECES) | get_bishop_attack(square, bitboards.ALL_PIECES)) & ~bitboards.WHITE_PIECES).bit_count()
    for square in chess.SquareSet(bitboards.BLACK_KNIGHTS): # Use SquareSet iteration
        black_mobility += (KNIGHT_ATTACKS[square] & ~bitboards.BLACK_PIECES).bit_count()
    for square in chess.SquareSet(bitboards.BLACK_BISHOPS): # Use SquareSet iteration
        black_mobility += (get_bishop_attack(square, bitboards.ALL_PIECES) & ~bitboards.BLACK_PIECES).bit_count()
    for square in chess.SquareSet(bitboards.BLACK_ROOKS): # Use SquareSet iteration
        black_mobility += (get_rook_attack(square, bitboards.ALL_PIECES) & ~bitboards.BLACK_PIECES).bit_count()
    for square in chess.SquareSet(bitboards.BLACK_QUEENS): # Use SquareSet iteration
        black_mobility += ((get_rook_attack(square, bitboards.ALL_PIECES) | get_bishop_attack(square, bitboards.ALL_PIECES)) & ~bitboards.BLACK_PIECES).bit_count()
    return (white_mobility - black_mobility) * mobility_weight

def calculate_game_phase():
    """Calculates the game phase (0=Endgame, 256=Middlegame) based on remaining material."""
    max_material = (
        (4 * MATERIAL_VALUES[chess.KNIGHT]) + (4 * MATERIAL_VALUES[chess.BISHOP]) +
        (4 * MATERIAL_VALUES[chess.ROOK]) + (2 * MATERIAL_VALUES[chess.QUEEN])
    )
    current_material = 0
    current_material += (bitboards.WHITE_KNIGHTS | bitboards.BLACK_KNIGHTS).bit_count() * MATERIAL_VALUES[chess.KNIGHT]
    current_material += (bitboards.WHITE_BISHOPS | bitboards.BLACK_BISHOPS).bit_count() * MATERIAL_VALUES[chess.BISHOP]
    current_material += (bitboards.WHITE_ROOKS | bitboards.BLACK_ROOKS).bit_count() * MATERIAL_VALUES[chess.ROOK]
    current_material += (bitboards.WHITE_QUEENS | bitboards.BLACK_QUEENS).bit_count() * MATERIAL_VALUES[chess.QUEEN]
    if max_material == 0: return 0
    phase_ratio = min(1.0, max(0.0, current_material / max_material))
    phase = int(phase_ratio * 256)
    return min(256, max(0, phase))

# --- Heuristic Score Calculation Function ---

def calculate_heuristic_score():
    """
    Calculates the heuristic evaluation score based on the current global bitboard state.
    This function does NOT check for terminal states.
    Returns the score from White's perspective.
    DEBUG: Simplified to only material score for testing mate search.
    """
    # Calculate evaluation components using the current global bitboards
    material_score = calculate_material_score()
    pst_score = calculate_pst_score() # Enable PST
    mobility_score = calculate_mobility_score() # Enable Mobility
    # TODO: Add other heuristic components (pawn structure, king safety etc.)

    # Combine scores (simple addition for now, consider tapering later)
    final_score = material_score + pst_score + mobility_score
    # Tapered Evaluation Logic (Keep commented out for now)
    # mg_score = material_score + pst_score + mobility_score # Add PST/Mobility to MG
    # eg_score = material_score + pst_score # Maybe less mobility in EG? Or king activity instead?
    # game_phase = calculate_game_phase()
    # final_score = (mg_score * game_phase + eg_score * (256 - game_phase)) // 256

    # Return score from White's perspective.
    return final_score


# --- Insufficient Material Check ---
def has_sufficient_material(color):
    """ Checks if a side *might* have sufficient mating material. More complex than python-chess. """
    # This is a basic check. Doesn't handle complex fortresses etc.
    if color == chess.WHITE:
        if bitboards.WHITE_PAWNS or bitboards.WHITE_ROOKS or bitboards.WHITE_QUEENS:
            return True # Pawns, Rooks, or Queens are sufficient
        # Check for Bishop + Knight or multiple minors
        knight_count = bitboards.WHITE_KNIGHTS.bit_count()
        bishop_count = bitboards.WHITE_BISHOPS.bit_count()
        if knight_count >= 1 and bishop_count >= 1: return True # B+N is sufficient
        if knight_count >= 2: return True # 2 Knights *can* mate, though tricky
        if bishop_count >= 2: # Need to check if bishops are on different colors
            light_bishops = bitboards.WHITE_BISHOPS & chess.BB_LIGHT_SQUARES
            dark_bishops = bitboards.WHITE_BISHOPS & chess.BB_DARK_SQUARES
            if light_bishops and dark_bishops: return True # Two bishops on different colors
    else: # Black
        if bitboards.BLACK_PAWNS or bitboards.BLACK_ROOKS or bitboards.BLACK_QUEENS:
            return True
        knight_count = bitboards.BLACK_KNIGHTS.bit_count()
        bishop_count = bitboards.BLACK_BISHOPS.bit_count()
        if knight_count >= 1 and bishop_count >= 1: return True
        if knight_count >= 2: return True
        light_bishops = bitboards.BLACK_BISHOPS & chess.BB_LIGHT_SQUARES
        dark_bishops = bitboards.BLACK_BISHOPS & chess.BB_DARK_SQUARES
        if light_bishops and dark_bishops: return True

    return False # Otherwise, likely insufficient

def is_insufficient_material_draw():
   """ Checks if the position is a draw due to insufficient mating material for BOTH sides. """
   return not has_sufficient_material(chess.WHITE) and not has_sufficient_material(chess.BLACK)

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    from .board_state import initialize_board_state
    from . import bitboards

    # Function to run and print heuristic evaluation for a given FEN
    def test_heuristic_eval(fen, description):
        print(f"\n--- {description} ---")
        print(f"FEN: {fen}")
        board_obj = chess.Board(fen)
        print(board_obj)
        # Initialize global bitboards for this position
        initialize_board_state(board_obj)
        # Calculate heuristic score
        eval_score = calculate_heuristic_score()
        # Adjust for side to move if needed for comparison, but raw score is White's perspective
        perspective_score = eval_score * (1 if bitboards.SIDE_TO_MOVE == chess.WHITE else -1)
        print(f"Heuristic Score (White's Perspective): {eval_score}")
        # print(f"Heuristic Score (Side-to-Move Perspective): {perspective_score}")

        is_mate = board_obj.is_checkmate()
        is_stale = board_obj.is_stalemate()
        is_insuf = board_obj.is_insufficient_material()
        print(f"Terminal State Check: Mate={is_mate}, Stalemate={is_stale}, InsufficientMat={is_insuf}")
        # Use the updated MATE_SCORE constant
        if is_mate: print(f"  (Expected Eval: {'-' if board_obj.turn == chess.WHITE else '+'}MATE_SCORE={MATE_SCORE})")
        if is_stale or is_insuf: print("  (Expected Eval: DRAW_SCORE)")


        return eval_score

    # --- Basic Positions ---
    print("--- Basic Position Tests ---")
    test_heuristic_eval(chess.STARTING_FEN, "Initial Position")
    board_temp = chess.Board()
    board_temp.push_san("e4")
    test_heuristic_eval(board_temp.fen(), "After e4")
    board_temp.push_san("d5")
    test_heuristic_eval(board_temp.fen(), "After e4 d5")

    # --- Specific Test Cases ---
    test_heuristic_eval("8/8/8/8/8/k7/P7/K6R w - - 0 1", "Endgame (R+P vs K)")

    print("\n--- Heuristic Sanity Checks (Terminal states handled by search) ---")

    # 1. Material Imbalance (White up Queen)
    board_material = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    board_material.remove_piece_at(chess.D8)
    test_heuristic_eval(board_material.fen(), "Material Imbalance (White up Q)")

    # 2. Checkmate (Fool's Mate) - Heuristic score is irrelevant, search handles terminal
    test_heuristic_eval("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3", "Checkmate (Fool's Mate)")

    # 3. Checkmate (Back Rank Mate) - Heuristic score is irrelevant
    board_back_rank = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    board_back_rank.push_san("Ra8#")
    test_heuristic_eval(board_back_rank.fen(), "Checkmate (Back Rank Mate)")

    # 4. Stalemate - Heuristic score is irrelevant
    stalemate_fen = "k7/8/8/8/8/8/7R/K7 b - - 0 1"
    test_heuristic_eval(stalemate_fen, "Stalemate")

    # 5. Insufficient Material (King vs King) - Heuristic score might be 0
    kvk_fen = "k7/8/8/8/8/8/8/K7 w - - 0 1"
    test_heuristic_eval(kvk_fen, "Insufficient Material (KvK)")

    # 6. Insufficient Material (King+Bishop vs King)
    kvkb_fen = "k7/8/8/8/8/8/B7/K7 w - - 0 1"
    test_heuristic_eval(kvkb_fen, "Insufficient Material (KvKB)")

    # 7. Insufficient Material (King+Knight vs King)
    kvkn_fen = "k7/8/8/8/8/8/N7/K7 w - - 0 1"
    test_heuristic_eval(kvkn_fen, "Insufficient Material (KvKN)")

    # 8. Insufficient Material (KvKB vs KvKB same color)
    kvkb_same_fen = "kb6/8/8/8/8/8/B7/K7 w - - 0 1"
    test_heuristic_eval(kvkb_same_fen, "Insufficient Material (KvKB vs KvKB same color)")
