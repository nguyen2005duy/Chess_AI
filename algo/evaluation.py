import chess
from . import bitboards
from .magic_bitboards import KING_ATTACKS, KNIGHT_ATTACKS, get_rook_attack, get_bishop_attack

# --- Constants ---
MAX_PLY = 64  # Maximum search depth (shared constant)
MATE_SCORE = 999999  # Increased Mate Score
MATE_THRESHOLD = MATE_SCORE - MAX_PLY  # Threshold to identify mate scores
DRAW_SCORE = 0  # Score for draws

# --- Evaluation Components ---

# Material values
MATERIAL_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 325,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Piece-Square Tables (PSTs)
PST = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ],
    chess.ROOK: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ],
    chess.QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ],
    chess.KING: [  #
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 40, 10, 0, 0, 10, 40, 20
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
PASSED_PAWN_BONUSES_MG = [0, 5, 10, 20, 35, 60, 100]
PASSED_PAWN_BONUSES_EG = [0, 10, 30, 45, 70, 120, 200]

# Isolated pawn penalties (less severe in opening/middlegame)
ISOLATED_PAWN_PENALTY_BY_COUNT_MG = [0, -8, -16, -25, -35, -40, -45, -50, -55]
ISOLATED_PAWN_PENALTY_BY_COUNT_EG = [0, -12, -25, -40, -50, -55, -60, -65, -70]

# King pawn shield (more important in middlegame, less in endgame)
KING_PAWN_SHIELD_SCORES_MG = [15, 20, 15, 10, 15, 10]
KING_PAWN_SHIELD_SCORES_EG = [5, 8, 5, 3, 5, 3]

# Piece mobility (knights need more mobility in endgame)
MOBILITY_WEIGHTS_MG = {
    chess.KNIGHT: 4,
    chess.BISHOP: 5,
    chess.ROOK: 3,
    chess.QUEEN: 2
}

MOBILITY_WEIGHTS_EG = {
    chess.KNIGHT: 6,
    chess.BISHOP: 5,
    chess.ROOK: 4,
    chess.QUEEN: 3
}

# Other position evaluation parameters
DOUBLED_PAWN_PENALTY_MG = -12
DOUBLED_PAWN_PENALTY_EG = -20

BACKWARD_PAWN_PENALTY_MG = -15
BACKWARD_PAWN_PENALTY_EG = -25

PAWN_CHAIN_BONUS_MG = 10
PAWN_CHAIN_BONUS_EG = 5

PAWN_ISLAND_PENALTY_MG = -12
PAWN_ISLAND_PENALTY_EG = -20

BISHOP_PAIR_BONUS_MG = 25
BISHOP_PAIR_BONUS_EG = 50

OPEN_FILE_BONUS_MG = 20
OPEN_FILE_BONUS_EG = 30

UNCASTLED_KING_BASE_PENALTY_MG = 70
UNCASTLED_KING_BASE_PENALTY_EG = 30

OPEN_FILE_NEXT_TO_KING_PENALTY_MG = 25
OPEN_FILE_NEXT_TO_KING_PENALTY_EG = 10

OPEN_FILE_ON_KING_PENALTY_MG = 40
OPEN_FILE_ON_KING_PENALTY_EG = 20


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


def calculate_white_pst_score():
    """Calculates the PST score based on current bitboards."""
    score = 0
    for piece_type, table in PST.items():
        bb = getattr(bitboards, f"WHITE_{chess.piece_name(piece_type).upper()}S", 0)
        for square in chess.SquareSet(bb):  # Use SquareSet iteration
            score += table[square]
    return score


def calculate_black_pst_score():
    """Calculates the PST score based on current bitboards."""
    score = 0
    for piece_type, table in PST.items():
        bb = getattr(bitboards, f"BLACK_{chess.piece_name(piece_type).upper()}S", 0)
        mirror_table = MIRRORED_PST[piece_type]
        for square in chess.SquareSet(bb):  # Use SquareSet iteration
            score -= mirror_table[square]
    return score


def calculate_passed_pawn_and_isolation_score():
    """Calculate scores for passed pawns and isolated pawns."""
    score = 0
    phase = calculate_game_phase()

    # Define file masks for isolation detection
    file_masks = [0] * 8
    adjacent_file_masks = [0] * 8
    for file_idx in range(8):
        file_mask = 0
        for rank_idx in range(8):
            file_mask |= (1 << (rank_idx * 8 + file_idx))
        file_masks[file_idx] = file_mask

        # Create adjacent file masks
        adj_mask = 0
        if file_idx > 0:
            adj_mask |= file_masks[file_idx - 1]
        if file_idx < 7:
            adj_mask |= file_masks[file_idx + 1]
        adjacent_file_masks[file_idx] = adj_mask

    # White passed pawns and isolated pawns
    white_isolated_count = 0
    for square in chess.SquareSet(bitboards.WHITE_PAWNS):
        # Check for passed pawn (no black pawns ahead on same or adjacent files)
        file_idx = square % 8
        rank_idx = square // 8

        # Create passed pawn mask for this square
        passed_mask = 0
        for r in range(rank_idx + 1, 8):  # Squares ahead for white
            passed_mask |= 1 << (r * 8 + file_idx)  # Same file
            if file_idx > 0:
                passed_mask |= 1 << (r * 8 + file_idx - 1)  # Left file
            if file_idx < 7:
                passed_mask |= 1 << (r * 8 + file_idx + 1)  # Right file

        # If no black pawns in this mask, it's a passed pawn
        if not (bitboards.BLACK_PAWNS & passed_mask):
            # Calculate bonus based on rank with phase-based tapering
            squares_from_promotion = 7 - rank_idx
            mg_bonus = PASSED_PAWN_BONUSES_MG[min(6, squares_from_promotion)]
            eg_bonus = PASSED_PAWN_BONUSES_EG[min(6, squares_from_promotion)]
            score += interpolate(mg_bonus, eg_bonus, phase)

        # Check for isolated pawn
        if not (bitboards.WHITE_PAWNS & adjacent_file_masks[file_idx]):
            white_isolated_count += 1

    # Black passed pawns and isolated pawns (similar logic with sign flipped)
    black_isolated_count = 0
    # ... [Similar code for black pawns] ...
    for square in chess.SquareSet(bitboards.BLACK_PAWNS):
        # Check for passed pawn
        file_idx = square % 8
        rank_idx = square // 8

        # Create passed pawn mask for this square
        passed_mask = 0
        for r in range(0, rank_idx):  # Squares ahead for black
            passed_mask |= 1 << (r * 8 + file_idx)  # Same file
            if file_idx > 0:
                passed_mask |= 1 << (r * 8 + file_idx - 1)  # Left file
            if file_idx < 7:
                passed_mask |= 1 << (r * 8 + file_idx + 1)  # Right file

        # If no white pawns in this mask, it's a passed pawn
        if not (bitboards.WHITE_PAWNS & passed_mask):
            squares_from_promotion = rank_idx
            mg_bonus = PASSED_PAWN_BONUSES_MG[min(6, squares_from_promotion)]
            eg_bonus = PASSED_PAWN_BONUSES_EG[min(6, squares_from_promotion)]
            score -= interpolate(mg_bonus, eg_bonus, phase)

        # Check for isolated pawn
        if not (bitboards.BLACK_PAWNS & adjacent_file_masks[file_idx]):
            black_isolated_count += 1

    # Add isolated pawn penalties with tapering
    mg_white_penalty = ISOLATED_PAWN_PENALTY_BY_COUNT_MG[min(8, white_isolated_count)]
    eg_white_penalty = ISOLATED_PAWN_PENALTY_BY_COUNT_EG[min(8, white_isolated_count)]
    mg_black_penalty = ISOLATED_PAWN_PENALTY_BY_COUNT_MG[min(8, black_isolated_count)]
    eg_black_penalty = ISOLATED_PAWN_PENALTY_BY_COUNT_EG[min(8, black_isolated_count)]

    score += interpolate(mg_white_penalty, eg_white_penalty, phase)
    score -= interpolate(mg_black_penalty, eg_black_penalty, phase)

    return score


def calculate_mobility_score():
    """Calculates a balanced mobility score based on current bitboards."""
    white_mobility = 0
    black_mobility = 0
    phase = calculate_game_phase()

    # Get the appropriate mobility weights for the current phase
    weights = {}
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        weights[piece_type] = interpolate(
            MOBILITY_WEIGHTS_MG[piece_type],
            MOBILITY_WEIGHTS_EG[piece_type],
            phase
        )

    # Calculate white mobility with phase-adjusted weights
    for square in chess.SquareSet(bitboards.WHITE_KNIGHTS):
        white_mobility += (KNIGHT_ATTACKS[square] & ~bitboards.WHITE_PIECES).bit_count() * weights[chess.KNIGHT]
    # ... [Similar code for other pieces] ...
    for square in chess.SquareSet(bitboards.WHITE_BISHOPS):
        white_mobility += (get_bishop_attack(square, bitboards.ALL_PIECES) & ~bitboards.WHITE_PIECES).bit_count() * \
                          weights[chess.BISHOP]
    for square in chess.SquareSet(bitboards.WHITE_ROOKS):
        white_mobility += (get_rook_attack(square, bitboards.ALL_PIECES) & ~bitboards.WHITE_PIECES).bit_count() * \
                          weights[chess.ROOK]
    for square in chess.SquareSet(bitboards.WHITE_QUEENS):
        white_mobility += ((get_rook_attack(square, bitboards.ALL_PIECES) | get_bishop_attack(square,
                                                                                              bitboards.ALL_PIECES)) & ~bitboards.WHITE_PIECES).bit_count() * \
                          weights[chess.QUEEN]

    # Calculate black mobility
    for square in chess.SquareSet(bitboards.BLACK_KNIGHTS):
        black_mobility += (KNIGHT_ATTACKS[square] & ~bitboards.BLACK_PIECES).bit_count() * weights[chess.KNIGHT]
    # ... [Similar code for other pieces] ...
    for square in chess.SquareSet(bitboards.BLACK_BISHOPS):
        black_mobility += (get_bishop_attack(square, bitboards.ALL_PIECES) & ~bitboards.BLACK_PIECES).bit_count() * \
                          weights[chess.BISHOP]
    for square in chess.SquareSet(bitboards.BLACK_ROOKS):
        black_mobility += (get_rook_attack(square, bitboards.ALL_PIECES) & ~bitboards.BLACK_PIECES).bit_count() * \
                          weights[chess.ROOK]
    for square in chess.SquareSet(bitboards.BLACK_QUEENS):
        black_mobility += ((get_rook_attack(square, bitboards.ALL_PIECES) | get_bishop_attack(square,
                                                                                              bitboards.ALL_PIECES)) & ~bitboards.BLACK_PIECES).bit_count() * \
                          weights[chess.QUEEN]

    return white_mobility - black_mobility


def get_king_pawn_shield_square_set(king_square, color):
    """Generate squares that should contain pawns to shield the king."""
    file_idx = king_square % 8
    rank_idx = king_square // 8
    shield_squares = []

    # Only evaluate pawn shield if king is on the kingside or queenside
    if file_idx <= 2 or file_idx >= 5:
        # Define shield squares based on king position
        if color == chess.WHITE:
            base_rank = rank_idx - 1  # One rank in front of white king
            if base_rank >= 0:  # Make sure we don't go off the board
                # Add three squares in front of the king
                if file_idx > 0:  # Left square
                    shield_squares.append(base_rank * 8 + file_idx - 1)
                shield_squares.append(base_rank * 8 + file_idx)  # Center square
                if file_idx < 7:  # Right square
                    shield_squares.append(base_rank * 8 + file_idx + 1)

                # Add the same three squares one more rank ahead if possible
                if base_rank - 1 >= 0:
                    if file_idx > 0:
                        shield_squares.append((base_rank - 1) * 8 + file_idx - 1)
                    shield_squares.append((base_rank - 1) * 8 + file_idx)
                    if file_idx < 7:
                        shield_squares.append((base_rank - 1) * 8 + file_idx + 1)
        else:  # BLACK
            base_rank = rank_idx + 1  # One rank in front of black king
            if base_rank <= 7:  # Make sure we don't go off the board
                # Add three squares in front of the king
                if file_idx > 0:  # Left square
                    shield_squares.append(base_rank * 8 + file_idx - 1)
                shield_squares.append(base_rank * 8 + file_idx)  # Center square
                if file_idx < 7:  # Right square
                    shield_squares.append(base_rank * 8 + file_idx + 1)

                # Add the same three squares one more rank ahead if possible
                if base_rank + 1 <= 7:
                    if file_idx > 0:
                        shield_squares.append((base_rank + 1) * 8 + file_idx - 1)
                    shield_squares.append((base_rank + 1) * 8 + file_idx)
                    if file_idx < 7:
                        shield_squares.append((base_rank + 1) * 8 + file_idx + 1)

    return shield_squares


def calculate_game_phase():
    """Calculates the game phase (0=Endgame, 256=Middlegame) based on remaining material."""
    PawnPhase = 0
    KnightPhase = 1
    BishopPhase = 1
    RookPhase = 2
    QueenPhase = 4

    # Total phase value at game start (excluding pawns and kings)
    total_phase = PawnPhase * 16 + KnightPhase * 4 + BishopPhase * 4 + RookPhase * 4 + QueenPhase * 2

    # Calculate current phase based on remaining material
    current_phase = total_phase

    # Subtract the phase value for each piece that's no longer on the board
    current_phase -= PawnPhase * (16 - (bitboards.WHITE_PAWNS | bitboards.BLACK_PAWNS).bit_count())
    current_phase -= KnightPhase * (4 - (bitboards.WHITE_KNIGHTS | bitboards.BLACK_KNIGHTS).bit_count())
    current_phase -= BishopPhase * (4 - (bitboards.WHITE_BISHOPS | bitboards.BLACK_BISHOPS).bit_count())
    current_phase -= RookPhase * (4 - (bitboards.WHITE_ROOKS | bitboards.BLACK_ROOKS).bit_count())
    current_phase -= QueenPhase * (2 - (bitboards.WHITE_QUEENS | bitboards.BLACK_QUEENS).bit_count())

    # Convert to a 0-256 scale
    return (current_phase * 256 + (total_phase // 2)) // total_phase

def calculate_rook_bonuses():
    score = 0
    # open_file_bonus = 25
    phase = calculate_game_phase()
    open_file_bonus = interpolate(OPEN_FILE_BONUS_MG, OPEN_FILE_BONUS_EG, phase)
    # White rooks on open files
    for square in chess.SquareSet(bitboards.WHITE_ROOKS):
        file_idx = square % 8
        file_mask = 0
        for rank in range(8):
            file_mask |= 1 << (rank * 8 + file_idx)
        if not (bitboards.WHITE_PAWNS & file_mask):
            if not (bitboards.BLACK_PAWNS & file_mask):
                score += open_file_bonus  # Open file
            else:
                score += open_file_bonus // 2  # Semi-open file
    # Same logic for black rooks (subtract)
    for square in chess.SquareSet(bitboards.BLACK_ROOKS):
        file_idx = square % 8
        file_mask = 0
        for rank in range(8):
            file_mask |= 1 << (rank * 8 + file_idx)
        if not (bitboards.BLACK_PAWNS & file_mask):
            if not (bitboards.WHITE_PAWNS & file_mask):
                score -= open_file_bonus  # Open file
            else:
                score -= open_file_bonus // 2  # Semi-open file
    return score


def calculate_pawn_structure_score():
    """Calculate scores for pawn structure: doubled pawns, backward pawns, pawn chains, and pawn islands."""
    score = 0

    # Calculate doubled pawns penalty
    doubled_pawn_score = calculate_doubled_pawns_score()

    # Calculate backward pawns penalty
    backward_pawn_score = calculate_backward_pawns_score()

    # Calculate pawn chains and pawn islands score
    pawn_chains_islands_score = calculate_pawn_chains_and_islands_score()

    # Combine scores
    score = doubled_pawn_score + backward_pawn_score + pawn_chains_islands_score

    return score


def calculate_doubled_pawns_score():
    """Calculate penalty for doubled pawns."""
    score = 0
    phase = calculate_game_phase()
    doubled_pawn_penalty = interpolate(DOUBLED_PAWN_PENALTY_MG, DOUBLED_PAWN_PENALTY_EG, phase)
    # Create file masks for doubled pawn detection
    file_masks = [0] * 8
    for file_idx in range(8):
        file_mask = 0
        for rank_idx in range(8):
            file_mask |= (1 << (rank_idx * 8 + file_idx))
        file_masks[file_idx] = file_mask

    # Check white doubled pawns
    for file_idx in range(8):
        white_pawns_in_file = (bitboards.WHITE_PAWNS & file_masks[file_idx]).bit_count()
        if white_pawns_in_file > 1:
            # Central doubled pawns are less bad than wing doubled pawns
            center_file_factor = 0.8 if 2 <= file_idx <= 5 else 1.0
            score += int(doubled_pawn_penalty * (white_pawns_in_file - 1) * center_file_factor)

    # Check black doubled pawns
    for file_idx in range(8):
        black_pawns_in_file = (bitboards.BLACK_PAWNS & file_masks[file_idx]).bit_count()
        if black_pawns_in_file > 1:
            center_file_factor = 0.8 if 2 <= file_idx <= 5 else 1.0
            score -= int(doubled_pawn_penalty * (black_pawns_in_file - 1) * center_file_factor)

    return score


def calculate_backward_pawns_score():
    """Calculate penalty for backward pawns."""
    score = 0
    # backward_pawn_penalty  = -20
    phase = calculate_game_phase()
    backward_pawn_penalty = interpolate(BACKWARD_PAWN_PENALTY_MG, BACKWARD_PAWN_PENALTY_EG, phase)

    # Create file masks
    file_masks = [0] * 8
    for file_idx in range(8):
        file_mask = 0
        for rank_idx in range(8):
            file_mask |= (1 << (rank_idx * 8 + file_idx))
        file_masks[file_idx] = file_mask

    # Check white backward pawns
    for square in chess.SquareSet(bitboards.WHITE_PAWNS):
        file_idx = square % 8
        rank_idx = square // 8

        # Skip pawns on the edge files (they can't have supporting pawns on both sides)
        if file_idx == 0 or file_idx == 7:
            continue

        # A pawn is backward if:
        # 1. It has no supporting pawns (no friendly pawns on adjacent files that are on the same rank or behind)
        # 2. It cannot safely advance (square in front is attacked by enemy pawn)

        # Check for supporting pawns
        has_support = False

        # Left adjacent file support check
        left_file_support_mask = file_masks[file_idx - 1]
        for check_rank in range(rank_idx, 8):  # Check same rank and behind
            left_support_square = check_rank * 8 + (file_idx - 1)
            if bitboards.WHITE_PAWNS & (1 << left_support_square):
                has_support = True
                break

        # Right adjacent file support check if not already supported
        if not has_support:
            right_file_support_mask = file_masks[file_idx + 1]
            for check_rank in range(rank_idx, 8):  # Check same rank and behind
                right_support_square = check_rank * 8 + (file_idx + 1)
                if bitboards.WHITE_PAWNS & (1 << right_support_square):
                    has_support = True
                    break

        # Check if the pawn is blocked from advancing
        square_in_front = (rank_idx - 1) * 8 + file_idx
        if rank_idx > 0:
            # Check if square in front is attacked by black pawn
            is_attacked = False

            # Check diagonal attacks from black pawns
            if file_idx > 0:  # Check left diagonal
                left_attack_square = (rank_idx - 1) * 8 + (file_idx - 1)
                if bitboards.BLACK_PAWNS & (1 << left_attack_square):
                    is_attacked = True

            if file_idx < 7 and not is_attacked:  # Check right diagonal
                right_attack_square = (rank_idx - 1) * 8 + (file_idx + 1)
                if bitboards.BLACK_PAWNS & (1 << right_attack_square):
                    is_attacked = True

            # If not supported and attacked, it's a backward pawn
            if not has_support and is_attacked:
                score += backward_pawn_penalty

    # Check black backward pawns (similar logic but reversed direction)
    for square in chess.SquareSet(bitboards.BLACK_PAWNS):
        file_idx = square % 8
        rank_idx = square // 8

        # Skip pawns on the edge files
        if file_idx == 0 or file_idx == 7:
            continue

        # Check for supporting pawns
        has_support = False

        # Left adjacent file support check
        left_file_support_mask = file_masks[file_idx - 1]
        for check_rank in range(0, rank_idx + 1):  # Check same rank and behind (for black)
            left_support_square = check_rank * 8 + (file_idx - 1)
            if bitboards.BLACK_PAWNS & (1 << left_support_square):
                has_support = True
                break

        # Right adjacent file support check if not already supported
        if not has_support:
            right_file_support_mask = file_masks[file_idx + 1]
            for check_rank in range(0, rank_idx + 1):  # Check same rank and behind (for black)
                right_support_square = check_rank * 8 + (file_idx + 1)
                if bitboards.BLACK_PAWNS & (1 << right_support_square):
                    has_support = True
                    break

        # Check if the pawn is blocked from advancing
        if rank_idx < 7:
            square_in_front = (rank_idx + 1) * 8 + file_idx

            # Check if square in front is attacked by white pawn
            is_attacked = False

            # Check diagonal attacks from white pawns
            if file_idx > 0:  # Check left diagonal
                left_attack_square = (rank_idx + 1) * 8 + (file_idx - 1)
                if bitboards.WHITE_PAWNS & (1 << left_attack_square):
                    is_attacked = True

            if file_idx < 7 and not is_attacked:  # Check right diagonal
                right_attack_square = (rank_idx + 1) * 8 + (file_idx + 1)
                if bitboards.WHITE_PAWNS & (1 << right_attack_square):
                    is_attacked = True

            # If not supported and attacked, it's a backward pawn
            if not has_support and is_attacked:
                score -= backward_pawn_penalty

    return score


def calculate_bishop_pair_bonus():
    """Calculate bonus for having the bishop pair."""
    score = 0
    phase = calculate_game_phase()
    bishop_pair_bonus = interpolate(BISHOP_PAIR_BONUS_MG, BISHOP_PAIR_BONUS_EG, phase)

    # Check if white has the bishop pair
    if bitboards.WHITE_BISHOPS.bit_count() >= 2:
        # Check if they're on opposite colors
        white_bishops = list(chess.SquareSet(bitboards.WHITE_BISHOPS))
        if (white_bishops[0] % 2) != (white_bishops[1] % 2):
            score += bishop_pair_bonus

    # Check if black has the bishop pair
    if bitboards.BLACK_BISHOPS.bit_count() >= 2:
        black_bishops = list(chess.SquareSet(bitboards.BLACK_BISHOPS))
        if (black_bishops[0] % 2) != (black_bishops[1] % 2):
            score -= bishop_pair_bonus

    return score
    # score = 0
    # bishop_pair_bonus = 30
    #
    # # Check if white has the bishop pair
    # if bitboards.WHITE_BISHOPS.bit_count() >= 2:
    #     # Check if they're on opposite colors
    #     white_bishops = list(chess.SquareSet(bitboards.WHITE_BISHOPS))
    #     if (white_bishops[0] % 2) != (white_bishops[1] % 2):
    #         score += bishop_pair_bonus
    #
    # # Check if black has the bishop pair
    # if bitboards.BLACK_BISHOPS.bit_count() >= 2:
    #     black_bishops = list(chess.SquareSet(bitboards.BLACK_BISHOPS))
    #     if (black_bishops[0] % 2) != (black_bishops[1] % 2):
    #         score -= bishop_pair_bonus
    #
    # return score


def calculate_pawn_chains_and_islands_score():
    """Calculate scores for pawn chains and pawn islands."""
    score = 0
    phase = calculate_game_phase()
    # Constants
    # pawn_chain_bonus = 8  # Bonus per pawn in a chain
    # pawn_island_penalty = -15  # Penalty per pawn island
    pawn_chain_bonus = interpolate(PAWN_CHAIN_BONUS_MG, PAWN_CHAIN_BONUS_EG, phase)
    pawn_island_penalty = interpolate(PAWN_ISLAND_PENALTY_MG, PAWN_ISLAND_PENALTY_EG, phase)
    # Create file masks
    file_masks = [0] * 8
    for file_idx in range(8):
        file_mask = 0
        for rank_idx in range(8):
            file_mask |= (1 << (rank_idx * 8 + file_idx))
        file_masks[file_idx] = file_mask

    # Calculate white pawn chains
    white_chain_count = 0
    for square in chess.SquareSet(bitboards.WHITE_PAWNS):
        file_idx = square % 8
        rank_idx = square // 8

        # Check if this pawn is supporting another pawn diagonally
        is_supporting = False

        # Check left diagonal support
        if file_idx > 0 and rank_idx > 0:
            left_diag_square = (rank_idx - 1) * 8 + (file_idx - 1)
            if bitboards.WHITE_PAWNS & (1 << left_diag_square):
                is_supporting = True
                white_chain_count += 1

        # Check right diagonal support
        if file_idx < 7 and rank_idx > 0 and not is_supporting:
            right_diag_square = (rank_idx - 1) * 8 + (file_idx + 1)
            if bitboards.WHITE_PAWNS & (1 << right_diag_square):
                is_supporting = True
                white_chain_count += 1

    # Calculate black pawn chains
    black_chain_count = 0
    for square in chess.SquareSet(bitboards.BLACK_PAWNS):
        file_idx = square % 8
        rank_idx = square // 8

        # Check if this pawn is supporting another pawn diagonally
        is_supporting = False

        # Check left diagonal support
        if file_idx > 0 and rank_idx < 7:
            left_diag_square = (rank_idx + 1) * 8 + (file_idx - 1)
            if bitboards.BLACK_PAWNS & (1 << left_diag_square):
                is_supporting = True
                black_chain_count += 1

        # Check right diagonal support
        if file_idx < 7 and rank_idx < 7 and not is_supporting:
            right_diag_square = (rank_idx + 1) * 8 + (file_idx + 1)
            if bitboards.BLACK_PAWNS & (1 << right_diag_square):
                is_supporting = True
                black_chain_count += 1

    # Add pawn chain bonus
    score += white_chain_count * pawn_chain_bonus
    score -= black_chain_count * pawn_chain_bonus

    # Calculate white pawn islands
    white_islands = 0
    files_with_white_pawns = [False] * 8

    for file_idx in range(8):
        if (bitboards.WHITE_PAWNS & file_masks[file_idx]) != 0:
            files_with_white_pawns[file_idx] = True

    # Count islands (groups of consecutive files with pawns)
    in_island = False
    for file_idx in range(8):
        if files_with_white_pawns[file_idx]:
            if not in_island:
                white_islands += 1
                in_island = True
        else:
            in_island = False

    # Calculate black pawn islands
    black_islands = 0
    files_with_black_pawns = [False] * 8

    for file_idx in range(8):
        if (bitboards.BLACK_PAWNS & file_masks[file_idx]) != 0:
            files_with_black_pawns[file_idx] = True

    # Count islands (groups of consecutive files with pawns)
    in_island = False
    for file_idx in range(8):
        if files_with_black_pawns[file_idx]:
            if not in_island:
                black_islands += 1
                in_island = True
        else:
            in_island = False

    # Add pawn island penalty (more than 1 island is penalized)
    if white_islands > 1:
        score += (white_islands - 1) * pawn_island_penalty
    if black_islands > 1:
        score -= (black_islands - 1) * pawn_island_penalty

    return score


def calculate_king_pawn_shield_score():
    """Calculate pawn shield score for kings in opening and middlegame only."""
    score = 0
    # Constants for material weights (extracted from the C# code snippet)
    rookEndgameWeight = 20
    bishopEndgameWeight = 10
    knightEndgameWeight = 10
    queenEndgameWeight = 45
    numWQueens = bitboards.WHITE_QUEENS.bit_count()
    numWRooks = bitboards.WHITE_ROOKS.bit_count()
    numWBishops = bitboards.WHITE_BISHOPS.bit_count()
    numWKnights = bitboards.WHITE_KNIGHTS.bit_count()
    numBQueens = bitboards.BLACK_QUEENS.bit_count()
    numBRooks = bitboards.BLACK_ROOKS.bit_count()
    numBBishops = bitboards.BLACK_BISHOPS.bit_count()
    numBKnights = bitboards.BLACK_KNIGHTS.bit_count()

    # Calculate endgame transition (based on the C# code)
    endgameStartWeight = 2 * rookEndgameWeight + 2 * bishopEndgameWeight + 2 * knightEndgameWeight + queenEndgameWeight
    WendgameWeightSum = numWQueens * queenEndgameWeight + numWRooks * rookEndgameWeight + numWBishops * bishopEndgameWeight + numWKnights * knightEndgameWeight
    BendgameWeightSum = numBQueens * queenEndgameWeight + numBRooks * rookEndgameWeight + numBBishops * bishopEndgameWeight + numBKnights * knightEndgameWeight

    # Calculate phase factors (1.0 = full middlegame, 0.0 = full endgame)
    WphaseT = 1 - min(1, WendgameWeightSum / endgameStartWeight)
    BphaseT = 1 - min(1, BendgameWeightSum / endgameStartWeight)

    # Scale phase to 256 for interpolation (256 = full middlegame, 0 = full endgame)
    Wphase = int(WphaseT * 256)
    Bphase = int(BphaseT * 256)

    # Evaluate Black king safety (from White's perspective: positive score)
    if Wphase > 0:  # Only evaluate if not in full endgame for White
        black_king_square = chess.SquareSet(bitboards.BLACK_KINGS).pop()
        black_king_file = black_king_square % 8

        if black_king_file <= 2 or black_king_file >= 5:  # Kingside or queenside castled position
            shield_squares = get_king_pawn_shield_square_set(black_king_square, chess.BLACK)
            black_penalty = 0

            for i, square in enumerate(shield_squares[:3]):  # First rank of shield
                if not (bitboards.BLACK_PAWNS & (1 << square)):
                    # Check if there's a pawn in the second rank of shield
                    if len(shield_squares) > 3 and i + 3 < len(shield_squares) and (
                            bitboards.BLACK_PAWNS & (1 << shield_squares[i + 3])):
                        # Interpolate between MG and EG values for second rank
                        shield_penalty = interpolate(KING_PAWN_SHIELD_SCORES_MG[i + 3],
                                                     KING_PAWN_SHIELD_SCORES_EG[i + 3],
                                                     Wphase)
                        black_penalty += shield_penalty
                    else:
                        # Interpolate between MG and EG values for first rank
                        shield_penalty = interpolate(KING_PAWN_SHIELD_SCORES_MG[i],
                                                     KING_PAWN_SHIELD_SCORES_EG[i],
                                                     Wphase)
                        black_penalty += shield_penalty

            black_penalty *= black_penalty  # Square the penalty as in original
            score += black_penalty
        else:
            # Uncastled black king penalty - tapered between MG and EG
            enemy_development = 0
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                bb = getattr(bitboards, f"WHITE_{chess.piece_name(piece_type).upper()}S", 0)
                # Count developed minor pieces
                for square in chess.SquareSet(bb):
                    if square // 8 != 7:  # Not on last rank
                        enemy_development += 1

            enemy_development_score = min(1.0, enemy_development / 4.0)

            # Interpolate uncastled king penalty between MG and EG values
            base_uncastled_penalty = interpolate(UNCASTLED_KING_BASE_PENALTY_MG,
                                                 UNCASTLED_KING_BASE_PENALTY_EG,
                                                 Wphase)
            uncastled_penalty = int(base_uncastled_penalty * enemy_development_score)
            score += uncastled_penalty

        # Only check open files if opponent has rooks or queens
        if numWRooks > 0 or numWQueens > 0:
            black_penalty = evaluate_open_files_against_king(black_king_file,
                                                             bitboards.BLACK_PAWNS,
                                                             bitboards.WHITE_PAWNS,
                                                             numWRooks,
                                                             numWQueens,
                                                             Wphase)
            score += black_penalty

    # Evaluate White king safety (from White's perspective: negative score)
    if Bphase > 0:  # Only evaluate if not in full endgame for Black
        white_king_square = chess.SquareSet(bitboards.WHITE_KINGS).pop()
        white_king_file = white_king_square % 8

        if white_king_file <= 2 or white_king_file >= 5:  # Kingside or queenside castled position
            shield_squares = get_king_pawn_shield_square_set(white_king_square, chess.WHITE)
            white_penalty = 0

            for i, square in enumerate(shield_squares[:3]):  # First rank of shield
                if not (bitboards.WHITE_PAWNS & (1 << square)):
                    # Check if there's a pawn in the second rank of shield
                    if len(shield_squares) > 3 and i + 3 < len(shield_squares) and (
                            bitboards.WHITE_PAWNS & (1 << shield_squares[i + 3])):
                        # Interpolate between MG and EG values for second rank
                        shield_penalty = interpolate(KING_PAWN_SHIELD_SCORES_MG[i + 3],
                                                     KING_PAWN_SHIELD_SCORES_EG[i + 3],
                                                     Bphase)
                        white_penalty += shield_penalty
                    else:
                        # Interpolate between MG and EG values for first rank
                        shield_penalty = interpolate(KING_PAWN_SHIELD_SCORES_MG[i],
                                                     KING_PAWN_SHIELD_SCORES_EG[i],
                                                     Bphase)
                        white_penalty += shield_penalty

            white_penalty *= white_penalty  # Square the penalty as in original
            score -= white_penalty
        else:
            # King in center penalty (uncastled) - tapered between MG and EG
            enemy_development = 0
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                bb = getattr(bitboards, f"BLACK_{chess.piece_name(piece_type).upper()}S", 0)
                # Count developed minor pieces (not on back rank)
                for square in chess.SquareSet(bb):
                    if square // 8 != 0:  # Not on first rank
                        enemy_development += 1

            enemy_development_score = min(1.0, enemy_development / 4.0)

            # Interpolate uncastled king penalty between MG and EG values
            base_uncastled_penalty = interpolate(UNCASTLED_KING_BASE_PENALTY_MG,
                                                 UNCASTLED_KING_BASE_PENALTY_EG,
                                                 Bphase)
            uncastled_penalty = int(base_uncastled_penalty * enemy_development_score)
            score -= uncastled_penalty

        # Only check open files if opponent has rooks or queens
        if numBRooks > 0 or numBQueens > 0:
            white_penalty = evaluate_open_files_against_king(white_king_file,
                                                             bitboards.WHITE_PAWNS,
                                                             bitboards.BLACK_PAWNS,
                                                             numBRooks,
                                                             numBQueens,
                                                             Bphase)
            score -= white_penalty

    return score


def evaluate_open_files_against_king(king_file, friendly_pawns, enemy_pawns, enemy_rook_count, enemy_queen_count,
                                     phase):
    """Evaluate penalty for open files near the king."""
    if enemy_rook_count < 1 and enemy_queen_count < 1:
        return 0

    open_file_penalty = 0
    king_file = min(6, max(1, king_file))  # Clamp king file to 1-6 range

    # Check the king's file and adjacent files
    for file_idx in range(king_file - 1, king_file + 2):
        if file_idx < 0 or file_idx > 7:
            continue

        file_mask = 0
        for rank_idx in range(8):
            file_mask |= (1 << (rank_idx * 8 + file_idx))

        is_king_file = file_idx == king_file

        # Check if file has no enemy pawns (potentially open)
        if (enemy_pawns & file_mask) == 0:
            if is_king_file:
                # Interpolate for king's file
                penalty = interpolate(OPEN_FILE_ON_KING_PENALTY_MG,
                                      OPEN_FILE_ON_KING_PENALTY_EG,
                                      phase)
            else:
                # Interpolate for adjacent files
                penalty = interpolate(OPEN_FILE_NEXT_TO_KING_PENALTY_MG,
                                      OPEN_FILE_NEXT_TO_KING_PENALTY_EG,
                                      phase)
            open_file_penalty += penalty

            # Check if file is fully open (no friendly pawns either)
            if (friendly_pawns & file_mask) == 0:
                # Add additional penalty for fully open files
                if is_king_file:
                    penalty = interpolate(OPEN_FILE_ON_KING_PENALTY_MG // 2,
                                          OPEN_FILE_ON_KING_PENALTY_EG // 2,
                                          phase)
                else:
                    penalty = interpolate(OPEN_FILE_NEXT_TO_KING_PENALTY_MG // 2,
                                          OPEN_FILE_NEXT_TO_KING_PENALTY_EG // 2,
                                          phase)
                open_file_penalty += penalty

    # Scale penalty by how dangerous the enemy attacking pieces are
    if enemy_rook_count > 1 or (enemy_rook_count > 0 and enemy_queen_count > 0):
        return open_file_penalty
    else:
        return open_file_penalty // 2


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
    pst_score = calculate_white_pst_score() - calculate_black_pst_score()  # Enable PST
    mobility_score = calculate_mobility_score()  # Enable Mobility
    # TODO: Add other heuristic components (pawn structure, king safety etc.)
    isolated_pawn_and_passed_pawn = calculate_passed_pawn_and_isolation_score()
    king_pawn_shield_score = calculate_king_pawn_shield_score()
    bishop_pair_score = calculate_bishop_pair_bonus()
    pawn_structure_score = calculate_pawn_structure_score()
    rook_score = calculate_rook_bonuses()
    # Combine scores (simple addition for now, consider tapering later)
    final_score = (material_score + pst_score + mobility_score
                   + isolated_pawn_and_passed_pawn
                   + king_pawn_shield_score + pawn_structure_score + bishop_pair_score + rook_score)

    # Return score from White's perspective.
    return final_score


def interpolate(mg_value, eg_value, phase):
    """
    Interpolate between middlegame and endgame values based on game phase.
    phase: 256 = full middlegame, 0 = full endgame
    """
    return (mg_value * phase + eg_value * (256 - phase)) // 256


# --- Insufficient Material Check ---
def has_sufficient_material(color):
    """ Checks if a side *might* have sufficient mating material. More complex than python-chess. """
    # This is a basic check. Doesn't handle complex fortresses etc.
    if color == chess.WHITE:
        if bitboards.WHITE_PAWNS or bitboards.WHITE_ROOKS or bitboards.WHITE_QUEENS:
            return True  # Pawns, Rooks, or Queens are sufficient
        # Check for Bishop + Knight or multiple minors
        knight_count = bitboards.WHITE_KNIGHTS.bit_count()
        bishop_count = bitboards.WHITE_BISHOPS.bit_count()
        if knight_count >= 1 and bishop_count >= 1: return True  # B+N is sufficient
        if knight_count >= 2: return True  # 2 Knights *can* mate, though tricky
        if bishop_count >= 2:  # Need to check if bishops are on different colors
            light_bishops = bitboards.WHITE_BISHOPS & chess.BB_LIGHT_SQUARES
            dark_bishops = bitboards.WHITE_BISHOPS & chess.BB_DARK_SQUARES
            if light_bishops and dark_bishops: return True  # Two bishops on different colors
    else:  # Black
        if bitboards.BLACK_PAWNS or bitboards.BLACK_ROOKS or bitboards.BLACK_QUEENS:
            return True
        knight_count = bitboards.BLACK_KNIGHTS.bit_count()
        bishop_count = bitboards.BLACK_BISHOPS.bit_count()
        if knight_count >= 1 and bishop_count >= 1: return True
        if knight_count >= 2: return True
        light_bishops = bitboards.BLACK_BISHOPS & chess.BB_LIGHT_SQUARES
        dark_bishops = bitboards.BLACK_BISHOPS & chess.BB_DARK_SQUARES
        if light_bishops and dark_bishops: return True

    return False  # Otherwise, likely insufficient


def is_insufficient_material_draw():
    """ Checks if the position is a draw due to insufficient mating material for BOTH sides. """
    return not has_sufficient_material(chess.WHITE) and not has_sufficient_material(chess.BLACK)
