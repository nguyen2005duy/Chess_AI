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
         20, 40, 10,  0,  0, 10, 40, 20
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

PASSED_PAWN_BONUSES = [0, 120, 80, 50, 30, 15, 15]
ISOLATED_PAWN_PENALTY_BY_COUNT = [0, -10, -25, -50, -75, -75, -75, -75, -75]
KING_PAWN_SHIELD_SCORES = [4, 7, 4, 3, 6, 3]

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
        for square in chess.SquareSet(bb): # Use SquareSet iteration
            score += table[square]
    return score
def calculate_black_pst_score():
    """Calculates the PST score based on current bitboards."""
    score = 0
    for piece_type, table in PST.items():
        bb = getattr(bitboards, f"BLACK_{chess.piece_name(piece_type).upper()}S", 0)
        mirror_table = MIRRORED_PST[piece_type]
        for square in chess.SquareSet(bb): # Use SquareSet iteration
            score -= mirror_table[square]
    return score

def calculate_passed_pawn_and_isolation_score():
    """Calculate scores for passed pawns and isolated pawns."""
    score = 0

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
        # Check for passed p    awn (no black pawns ahead on same or adjacent files)
        file_idx = square % 8
        rank_idx = square // 8

        # Create passed pawn mask for this square (all squares ahead on same or adjacent files)
        passed_mask = 0
        for r in range(rank_idx + 1, 8):  # Squares ahead for white
            passed_mask |= 1 << (r * 8 + file_idx)  # Same file
            if file_idx > 0:
                passed_mask |= 1 << (r * 8 + file_idx - 1)  # Left file
            if file_idx < 7:
                passed_mask |= 1 << (r * 8 + file_idx + 1)  # Right file

        # If no black pawns in this mask, it's a passed pawn
        if not (bitboards.BLACK_PAWNS & passed_mask):
            # Calculate bonus based on rank (how far advanced the pawn is)
            squares_from_promotion = 7 - rank_idx
            score += PASSED_PAWN_BONUSES[min(6, squares_from_promotion)]

        # Check for isolated pawn (no friendly pawns on adjacent files)
        if not (bitboards.WHITE_PAWNS & adjacent_file_masks[file_idx]):
            white_isolated_count += 1

    # Black passed pawns and isolated pawns
    black_isolated_count = 0
    for square in chess.SquareSet(bitboards.BLACK_PAWNS):
        # Check for passed pawn
        file_idx = square % 8
        rank_idx = square // 8

        # Create passed pawn mask for this square (all squares ahead on same or adjacent files)
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
            score -= PASSED_PAWN_BONUSES[min(6, squares_from_promotion)]

        # Check for isolated pawn
        if not (bitboards.BLACK_PAWNS & adjacent_file_masks[file_idx]):
            black_isolated_count += 1

    # Add isolated pawn penalties
    score += ISOLATED_PAWN_PENALTY_BY_COUNT[min(8, white_isolated_count)]
    score -= ISOLATED_PAWN_PENALTY_BY_COUNT[min(8, black_isolated_count)]

    return score


def calculate_mobility_score():
    """Calculates a balanced mobility score based on current bitboards."""
    white_mobility_knights = 0
    white_mobility_bishops = 0
    white_mobility_rooks = 0
    white_mobility_queens = 0

    black_mobility_knights = 0
    black_mobility_bishops = 0
    black_mobility_rooks = 0
    black_mobility_queens = 0

    # Fine-tuned mobility weights for different piece types
    knight_mobility_weight = 3
    bishop_mobility_weight = 3
    rook_mobility_weight = 2
    queen_mobility_weight = 1

    # Calculate white mobility
    for square in chess.SquareSet(bitboards.WHITE_KNIGHTS):
        white_mobility_knights += (KNIGHT_ATTACKS[square] & ~bitboards.WHITE_PIECES).bit_count()

    for square in chess.SquareSet(bitboards.WHITE_BISHOPS):
        white_mobility_bishops += (
                    get_bishop_attack(square, bitboards.ALL_PIECES) & ~bitboards.WHITE_PIECES).bit_count()

    for square in chess.SquareSet(bitboards.WHITE_ROOKS):
        white_mobility_rooks += (get_rook_attack(square, bitboards.ALL_PIECES) & ~bitboards.WHITE_PIECES).bit_count()

    for square in chess.SquareSet(bitboards.WHITE_QUEENS):
        white_mobility_queens += ((get_rook_attack(square, bitboards.ALL_PIECES) | get_bishop_attack(square,
                                                                                                     bitboards.ALL_PIECES)) & ~bitboards.WHITE_PIECES).bit_count()

    # Calculate black mobility
    for square in chess.SquareSet(bitboards.BLACK_KNIGHTS):
        black_mobility_knights += (KNIGHT_ATTACKS[square] & ~bitboards.BLACK_PIECES).bit_count()

    for square in chess.SquareSet(bitboards.BLACK_BISHOPS):
        black_mobility_bishops += (
                    get_bishop_attack(square, bitboards.ALL_PIECES) & ~bitboards.BLACK_PIECES).bit_count()

    for square in chess.SquareSet(bitboards.BLACK_ROOKS):
        black_mobility_rooks += (get_rook_attack(square, bitboards.ALL_PIECES) & ~bitboards.BLACK_PIECES).bit_count()

    for square in chess.SquareSet(bitboards.BLACK_QUEENS):
        black_mobility_queens += ((get_rook_attack(square, bitboards.ALL_PIECES) | get_bishop_attack(square,
                                                                                                     bitboards.ALL_PIECES)) & ~bitboards.BLACK_PIECES).bit_count()

    # Calculate the weighted total mobility scores
    white_mobility = (
            white_mobility_knights * knight_mobility_weight +
            white_mobility_bishops * bishop_mobility_weight +
            white_mobility_rooks * rook_mobility_weight +
            white_mobility_queens * queen_mobility_weight
    )

    black_mobility = (
            black_mobility_knights * knight_mobility_weight +
            black_mobility_bishops * bishop_mobility_weight +
            black_mobility_rooks * rook_mobility_weight +
            black_mobility_queens * queen_mobility_weight
    )

    # Calculate phase for mobility scaling
    # phase = calculate_game_phase()

    # Scale mobility importance based on game phase
    # In endgame, mobility is often more important
    # mobility_scaling = 1.0 + ((256 - phase) / 256.0) * 0.3  # Up to 30% more important in endgames

    return int(white_mobility - black_mobility)
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
    total_phase = (
        (4 * KnightPhase) + (4 * BishopPhase) +
        (4 * RookPhase) + (2 * QueenPhase)
    )
    current_material = total_phase
    current_material -= (bitboards.WHITE_KNIGHTS | bitboards.BLACK_KNIGHTS).bit_count() * KnightPhase
    current_material -= (bitboards.WHITE_BISHOPS | bitboards.BLACK_BISHOPS).bit_count() * BishopPhase
    current_material -= (bitboards.WHITE_ROOKS | bitboards.BLACK_ROOKS).bit_count() * RookPhase
    current_material -= (bitboards.WHITE_QUEENS | bitboards.BLACK_QUEENS).bit_count() * QueenPhase

    return (current_material * 256 +(total_phase / 2)) / total_phase


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
    doubled_pawn_penalty = -20  # Penalty per doubled pawn

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
            # Doubled pawns (or tripled) found on this file
            score += doubled_pawn_penalty * (white_pawns_in_file - 1)

    # Check black doubled pawns
    for file_idx in range(8):
        black_pawns_in_file = (bitboards.BLACK_PAWNS & file_masks[file_idx]).bit_count()
        if black_pawns_in_file > 1:
            # Doubled pawns (or tripled) found on this file
            score -= doubled_pawn_penalty * (black_pawns_in_file - 1)

    return score


def calculate_backward_pawns_score():
    """Calculate penalty for backward pawns."""
    score = 0
    backward_pawn_penalty = -15  # Penalty per backward pawn

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


def calculate_pawn_chains_and_islands_score():
    """Calculate scores for pawn chains and pawn islands."""
    score = 0

    # Constants
    pawn_chain_bonus = 5  # Bonus per pawn in a chain
    pawn_island_penalty = -10  # Penalty per pawn island

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

    # Calculate pieces for endgame transition
    numWQueens = bitboards.WHITE_QUEENS.bit_count()
    numWRooks = bitboards.WHITE_ROOKS.bit_count()
    numWBishops = bitboards.WHITE_BISHOPS.bit_count()
    numWKnights = bitboards.WHITE_KNIGHTS.bit_count()
    numBQueens =  bitboards.BLACK_QUEENS.bit_count()
    numBRooks = bitboards.BLACK_ROOKS.bit_count()
    numBBishops =  bitboards.BLACK_BISHOPS.bit_count()
    numBKnights =  bitboards.BLACK_KNIGHTS.bit_count()

    # Calculate endgame transition (based on the C# code)
    endgameStartWeight = 2 * rookEndgameWeight + 2 * bishopEndgameWeight + 2 * knightEndgameWeight + queenEndgameWeight
    WendgameWeightSum = numWQueens * queenEndgameWeight + numWRooks * rookEndgameWeight + numWBishops * bishopEndgameWeight + numWKnights * knightEndgameWeight
    BendgameWeightSum = numBQueens * queenEndgameWeight + numBRooks * rookEndgameWeight + numBBishops * bishopEndgameWeight + numBKnights * knightEndgameWeight
    WendgameT = 1 - min(1, WendgameWeightSum / endgameStartWeight)
    BendgameT = 1 - min(1, BendgameWeightSum / endgameStartWeight)

    # We use the inverse of endgameT as our phase weight (1.0 in opening, gradually less in middlegame)
    if BendgameT < 1:
        phase_weight = 1 - BendgameT
        pawn_shield_weight = phase_weight
        if bitboards.BLACK_QUEENS == 0 or bitboards.BLACK_QUEENS.bit_count() == 0:
            pawn_shield_weight *= 0.6  # Reduce importance if opponent has no queen

        # White king pawn shield
        white_king_square = chess.SquareSet(bitboards.WHITE_KINGS).pop()
        white_king_file = white_king_square % 8

        # Only evaluate shield for kingside/queenside castling positions
        if white_king_file <= 2 or white_king_file >= 5:
            shield_squares = get_king_pawn_shield_square_set(white_king_square, chess.WHITE)
            white_penalty = 0

            for i, square in enumerate(shield_squares[:3]):  # First rank of shield
                if not (bitboards.WHITE_PAWNS & (1 << square)):
                    # Check if there's a pawn in the second rank of shield
                    if len(shield_squares) > 3 and i + 3 < len(shield_squares) and (
                            bitboards.WHITE_PAWNS & (1 << shield_squares[i + 3])):
                        white_penalty += KING_PAWN_SHIELD_SCORES[i + 3]
                    else:
                        white_penalty += KING_PAWN_SHIELD_SCORES[i]

            white_penalty *= white_penalty  # Square the penalty as in the original
            score -= int(white_penalty * pawn_shield_weight)
        else:
            # King in center penalty (uncastled) - only in opening/middlegame
            # Approximate enemy piece development score
            enemy_development = 0
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                bb = getattr(bitboards, f"BLACK_{chess.piece_name(piece_type).upper()}S", 0)
                # Count developed minor pieces (not on back rank)
                for square in chess.SquareSet(bb):
                    if square // 8 != 0:  # Not on first rank
                        enemy_development += 1

            enemy_development_score = min(1.0, enemy_development / 4.0)
            uncastled_penalty = int(50 * enemy_development_score)
            score -= uncastled_penalty * pawn_shield_weight
        white_king_file = white_king_square % 8

        # Only check open files if opponent has rooks or queens
        if numBRooks > 1 or (numBRooks > 0 and numBQueens > 0):
            white_penalty = evaluate_open_files_against_king(white_king_file, bitboards.WHITE_PAWNS,
                                                             bitboards.BLACK_PAWNS,
                                                             bitboards.BLACK_ROOKS.bit_count(),
                                                             bitboards.BLACK_QUEENS.bit_count())
            score -= white_penalty * pawn_shield_weight

    if WendgameT < 1:
        phase_weight = 1 - WendgameT
        # Black king pawn shield - similar logic
        pawn_shield_weight = phase_weight
        if bitboards.WHITE_QUEENS == 0 or bitboards.WHITE_QUEENS.bit_count() == 0:
            pawn_shield_weight *= 0.6  # Reduce importance if opponent has no queen

        black_king_square = chess.SquareSet(bitboards.BLACK_KINGS).pop()
        black_king_file = black_king_square % 8

        if black_king_file <= 2 or black_king_file >= 5:
            shield_squares = get_king_pawn_shield_square_set(black_king_square, chess.BLACK)
            black_penalty = 0

            for i, square in enumerate(shield_squares[:3]):  # First rank of shield
                if not (bitboards.BLACK_PAWNS & (1 << square)):
                    # Check if there's a pawn in the second rank of shield
                    if len(shield_squares) > 3 and i + 3 < len(shield_squares) and (
                            bitboards.BLACK_PAWNS & (1 << shield_squares[i + 3])):
                        black_penalty += KING_PAWN_SHIELD_SCORES[i + 3]
                    else:
                        black_penalty += KING_PAWN_SHIELD_SCORES[i]

            black_penalty *= black_penalty  # Square the penalty
            score += int(black_penalty * pawn_shield_weight)
        else:
            # Uncastled black king penalty - only in opening/middlegame
            enemy_development = 0
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                bb = getattr(bitboards, f"WHITE_{chess.piece_name(piece_type).upper()}S", 0)
                # Count developed minor pieces
                for square in chess.SquareSet(bb):
                    if square // 8 != 7:  # Not on last rank
                        enemy_development += 1

            enemy_development_score = min(1.0, enemy_development / 4.0)
            uncastled_penalty = int(50 * enemy_development_score)
            score += uncastled_penalty * pawn_shield_weight
        black_king_file = black_king_square % 8
        if numWRooks > 1 or (numWRooks > 0 and numWQueens > 0):
            black_penalty = evaluate_open_files_against_king(black_king_file, bitboards.BLACK_PAWNS,
                                                             bitboards.WHITE_PAWNS,
                                                             bitboards.WHITE_ROOKS.bit_count(),
                                                             bitboards.WHITE_QUEENS.bit_count())
            score += black_penalty * pawn_shield_weight



    return score


def evaluate_open_files_against_king(king_file, friendly_pawns, enemy_pawns, enemy_rook_count, enemy_queen_count):
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
            open_file_penalty += 25 if is_king_file else 15
            # Check if file is fully open (no friendly pawns either)
            if (friendly_pawns & file_mask) == 0:
                open_file_penalty += 15 if is_king_file else 10

    # Scale penalty by how dangerous the enemy attacking pieces are
    if enemy_rook_count > 1 or (enemy_rook_count > 0 and enemy_queen_count > 0):
        return open_file_penalty
    else:
        return open_file_penalty

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
    pst_score = calculate_white_pst_score() - calculate_black_pst_score() # Enable PST
    mobility_score = calculate_mobility_score() # Enable Mobility
    # TODO: Add other heuristic components (pawn structure, king safety etc.)
    isolated_pawn_and_passed_pawn = calculate_passed_pawn_and_isolation_score()
    king_pawn_shield_score = calculate_king_pawn_shield_score()

    pawn_structure_score = calculate_pawn_structure_score()
    # rook_score = calculate_rook_evaluation()
    # Combine scores (simple addition for now, consider tapering later)
    final_score = (material_score + pst_score + mobility_score
                   + isolated_pawn_and_passed_pawn
                   + king_pawn_shield_score + pawn_structure_score)

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
