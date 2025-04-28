import chess
from . import bitboards
from .bitboards import initialize_bitboards as init_bitboards_from_board
from .zobrist import (
    ZOBRIST_TABLE, ZOBRIST_BLACK_TO_MOVE, ZOBRIST_CASTLING_RIGHTS, ZOBRIST_EN_PASSANT_FILE,
    PIECE_TO_ZOBRIST_INDEX, calculate_zobrist_hash as calculate_full_zobrist_hash
)

# Store board state history for undoing moves
BOARD_STATE_HISTORY = []

# Store position history (Zobrist hashes) for repetition detection
POSITION_HISTORY = []

# Current Zobrist hash of the board state
CURRENT_ZOBRIST_HASH = 0

# Lookup tables for pieces to reduce conditionals
WHITE_PIECE_BB = {
    chess.PAWN: lambda: bitboards.WHITE_PAWNS,
    chess.KNIGHT: lambda: bitboards.WHITE_KNIGHTS,
    chess.BISHOP: lambda: bitboards.WHITE_BISHOPS,
    chess.ROOK: lambda: bitboards.WHITE_ROOKS,
    chess.QUEEN: lambda: bitboards.WHITE_QUEENS,
    chess.KING: lambda: bitboards.WHITE_KINGS
}

BLACK_PIECE_BB = {
    chess.PAWN: lambda: bitboards.BLACK_PAWNS,
    chess.KNIGHT: lambda: bitboards.BLACK_KNIGHTS,
    chess.BISHOP: lambda: bitboards.BLACK_BISHOPS,
    chess.ROOK: lambda: bitboards.BLACK_ROOKS,
    chess.QUEEN: lambda: bitboards.BLACK_QUEENS,
    chess.KING: lambda: bitboards.BLACK_KINGS
}

# Castling rook move lookup table
CASTLING_ROOK_MOVES = {
    # (to_square, color): (from_square, to_square)
    (chess.G1, chess.WHITE): (chess.H1, chess.F1),
    (chess.C1, chess.WHITE): (chess.A1, chess.D1),
    (chess.G8, chess.BLACK): (chess.H8, chess.F8),
    (chess.C8, chess.BLACK): (chess.A8, chess.D8)
}

def initialize_board_state(board: chess.Board):
    """
    Initializes the bitboards and calculates the initial Zobrist hash
    based on the current state of a python-chess board.
    """
    global CURRENT_ZOBRIST_HASH, POSITION_HISTORY
    init_bitboards_from_board(board)
    CURRENT_ZOBRIST_HASH = calculate_full_zobrist_hash(board)
    # Clear history when initializing
    BOARD_STATE_HISTORY.clear()
    POSITION_HISTORY.clear()
    POSITION_HISTORY.append(CURRENT_ZOBRIST_HASH) # Add initial hash

def _update_zobrist_castling_rights(old_rights, new_rights):
    """Helper to update Zobrist hash for castling rights changes"""
    global CURRENT_ZOBRIST_HASH

    if old_rights == new_rights:
        return

    # XOR out old castling rights
    old_index = 0
    if old_rights & chess.BB_H1: old_index |= 1 << 0 # K
    if old_rights & chess.BB_A1: old_index |= 1 << 1 # Q
    if old_rights & chess.BB_H8: old_index |= 1 << 2 # k
    if old_rights & chess.BB_A8: old_index |= 1 << 3 # q
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_CASTLING_RIGHTS[old_index]

    # XOR in new castling rights
    new_index = 0
    if new_rights & chess.BB_H1: new_index |= 1 << 0 # K
    if new_rights & chess.BB_A1: new_index |= 1 << 1 # Q
    if new_rights & chess.BB_H8: new_index |= 1 << 2 # k
    if new_rights & chess.BB_A8: new_index |= 1 << 3 # q
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_CASTLING_RIGHTS[new_index]

def _update_zobrist_ep_square(old_ep, new_ep):
    """Helper to update Zobrist hash for en passant square changes"""
    global CURRENT_ZOBRIST_HASH

    # XOR out old EP square hash if it existed
    if old_ep is not None:
        CURRENT_ZOBRIST_HASH ^= ZOBRIST_EN_PASSANT_FILE[chess.square_file(old_ep)]

    # XOR in new EP square hash if it exists
    if new_ep is not None:
        CURRENT_ZOBRIST_HASH ^= ZOBRIST_EN_PASSANT_FILE[chess.square_file(new_ep)]

def _move_piece(piece_type, color, from_square, to_square):
    """Helper function to move a piece and update Zobrist hash"""
    global CURRENT_ZOBRIST_HASH

    from_bb = 1 << from_square
    to_bb = 1 << to_square

    # Update the appropriate bitboard directly
    if color == chess.WHITE:
        if piece_type == chess.PAWN:
            bitboards.WHITE_PAWNS &= ~from_bb
            bitboards.WHITE_PAWNS |= to_bb
        elif piece_type == chess.KNIGHT:
            bitboards.WHITE_KNIGHTS &= ~from_bb
            bitboards.WHITE_KNIGHTS |= to_bb
        elif piece_type == chess.BISHOP:
            bitboards.WHITE_BISHOPS &= ~from_bb
            bitboards.WHITE_BISHOPS |= to_bb
        elif piece_type == chess.ROOK:
            bitboards.WHITE_ROOKS &= ~from_bb
            bitboards.WHITE_ROOKS |= to_bb
        elif piece_type == chess.QUEEN:
            bitboards.WHITE_QUEENS &= ~from_bb
            bitboards.WHITE_QUEENS |= to_bb
        elif piece_type == chess.KING:
            bitboards.WHITE_KINGS &= ~from_bb
            bitboards.WHITE_KINGS |= to_bb
    else:  # BLACK
        if piece_type == chess.PAWN:
            bitboards.BLACK_PAWNS &= ~from_bb
            bitboards.BLACK_PAWNS |= to_bb
        elif piece_type == chess.KNIGHT:
            bitboards.BLACK_KNIGHTS &= ~from_bb
            bitboards.BLACK_KNIGHTS |= to_bb
        elif piece_type == chess.BISHOP:
            bitboards.BLACK_BISHOPS &= ~from_bb
            bitboards.BLACK_BISHOPS |= to_bb
        elif piece_type == chess.ROOK:
            bitboards.BLACK_ROOKS &= ~from_bb
            bitboards.BLACK_ROOKS |= to_bb
        elif piece_type == chess.QUEEN:
            bitboards.BLACK_QUEENS &= ~from_bb
            bitboards.BLACK_QUEENS |= to_bb
        elif piece_type == chess.KING:
            bitboards.BLACK_KINGS &= ~from_bb
            bitboards.BLACK_KINGS |= to_bb

    # Update Zobrist hash
    piece_index = PIECE_TO_ZOBRIST_INDEX[(piece_type, color)]
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[piece_index][from_square]
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[piece_index][to_square]

    return piece_type  # Return the piece type that was moved

def _handle_castling(moving_piece_type, side_to_move, from_square, to_square):
    """Helper function to handle castling rook movement"""
    global CURRENT_ZOBRIST_HASH

    if moving_piece_type != chess.KING or abs(from_square - to_square) != 2:
        return False

    castling_key = (to_square, side_to_move)
    if castling_key not in CASTLING_ROOK_MOVES:
        return False

    rook_from, rook_to = CASTLING_ROOK_MOVES[castling_key]

    # Move the rook
    if side_to_move == chess.WHITE:
        bitboards.WHITE_ROOKS &= ~(1 << rook_from)
        bitboards.WHITE_ROOKS |= (1 << rook_to)
    else:
        bitboards.BLACK_ROOKS &= ~(1 << rook_from)
        bitboards.BLACK_ROOKS |= (1 << rook_to)

    # Update Zobrist hash for rook movement
    rook_index = PIECE_TO_ZOBRIST_INDEX[(chess.ROOK, side_to_move)]
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[rook_index][rook_from]
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[rook_index][rook_to]

    return True

def _find_piece_at_square(square, side):
    """Find which piece is at the given square for the given side"""
    square_bit = 1 << square

    piece_bbs = WHITE_PIECE_BB if side == chess.WHITE else BLACK_PIECE_BB

    for piece_type, bb_func in piece_bbs.items():
        if bb_func() & square_bit:
            return piece_type

    return None

def _update_castling_rights(rights, moving_piece_type, moving_side, from_square, to_square, captured_piece_type=None):
    """Helper to update castling rights based on a move"""
    new_rights = rights

    # King moves lose all castling rights for that side
    if moving_piece_type == chess.KING:
        if moving_side == chess.WHITE:
            new_rights &= ~(chess.BB_H1 | chess.BB_A1)  # Clear K and Q
        else:
            new_rights &= ~(chess.BB_H8 | chess.BB_A8)  # Clear k and q

    # Rook moves lose castling rights for that rook
    elif moving_piece_type == chess.ROOK:
        if moving_side == chess.WHITE:
            if from_square == chess.H1: new_rights &= ~chess.BB_H1  # Clear K
            elif from_square == chess.A1: new_rights &= ~chess.BB_A1  # Clear Q
        else:
            if from_square == chess.H8: new_rights &= ~chess.BB_H8  # Clear k
            elif from_square == chess.A8: new_rights &= ~chess.BB_A8  # Clear q

    # Captures on rook squares lose castling rights for that rook
    if captured_piece_type is not None:
        if to_square == chess.H1: new_rights &= ~chess.BB_H1
        elif to_square == chess.A1: new_rights &= ~chess.BB_A1
        elif to_square == chess.H8: new_rights &= ~chess.BB_H8
        elif to_square == chess.A8: new_rights &= ~chess.BB_A8

    return new_rights

def push_move(move: chess.Move):
    """
    Applies a move to the bitboard representation of the board state.
    Updates bitboards, game state variables, and the Zobrist hash incrementally.
    """
    global CURRENT_ZOBRIST_HASH, POSITION_HISTORY

    # --- Handle Null Move (fast path) ---
    if move == chess.Move.null():
        _handle_null_move()
        return

    # --- For regular moves, store original state values ---
    original_castling_rights = bitboards.CASTLING_RIGHTS
    original_ep_square = bitboards.EN_PASSANT_SQUARE
    original_halfmove_clock = bitboards.HALF_MOVE_COUNTER
    original_side_to_move = bitboards.SIDE_TO_MOVE
    original_zobrist_hash = CURRENT_ZOBRIST_HASH

    from_square = move.from_square
    to_square = move.to_square
    from_bb = 1 << from_square
    to_bb = 1 << to_square
    side_that_moved = original_side_to_move
    opponent_side = not side_that_moved

    # Create history entry (captured piece will be set later)
    BOARD_STATE_HISTORY.append({
        'move': move,
        'captured_piece_type': None,
        'prev_castling_rights': original_castling_rights,
        'prev_ep_square': original_ep_square,
        'prev_halfmove_clock': original_halfmove_clock,
        'prev_zobrist_hash': original_zobrist_hash
    })

    # Find the moving piece type by examining bitboards
    # Fast lookup by directly checking the appropriate bitboards
    moving_piece_type = _find_piece_type_fast(from_square, side_that_moved)
    if moving_piece_type is None:
        raise ValueError(f"No piece found at square {chess.square_name(from_square)}")

    # Update Zobrist hash for piece movement (XOR out from source, XOR in at destination)
    piece_index = PIECE_TO_ZOBRIST_INDEX[(moving_piece_type, side_that_moved)]
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[piece_index][from_square] ^ ZOBRIST_TABLE[piece_index][to_square]

    # Move the piece (removing from source, adding to destination)
    _move_piece_fast(moving_piece_type, side_that_moved, from_bb, to_bb)

    # --- Handle captures ---
    captured_piece_type = None
    opponent_bb = bitboards.BLACK_PIECES if side_that_moved == chess.WHITE else bitboards.WHITE_PIECES

    if opponent_bb & to_bb:
        captured_piece_type = _handle_capture(to_square, opponent_side, to_bb)

    # Record the captured piece in history
    BOARD_STATE_HISTORY[-1]['captured_piece_type'] = captured_piece_type

    # --- Handle special moves ---
    # En passant capture
    if moving_piece_type == chess.PAWN and to_square == original_ep_square and original_ep_square is not None:
        _handle_en_passant_capture(side_that_moved, to_square)
        captured_piece_type = chess.PAWN
        BOARD_STATE_HISTORY[-1]['captured_piece_type'] = chess.PAWN

    # Handle promotion
    if move.promotion is not None:
        _handle_promotion(move.promotion, side_that_moved, to_bb)

    # Handle castling
    _handle_castling(moving_piece_type, side_that_moved, from_square, to_square)

    # --- Update game state ---
    # Update aggregate bitboards
    _update_aggregate_bitboards()

    # Update halfmove clock
    bitboards.HALF_MOVE_COUNTER = 0 if moving_piece_type == chess.PAWN or captured_piece_type is not None else bitboards.HALF_MOVE_COUNTER + 1

    # Set en passant square (only for pawn double moves)
    new_ep_square = None
    if moving_piece_type == chess.PAWN and abs(from_square - to_square) == 16:
        new_ep_square = (from_square + to_square) // 2

    # Update EP square in hash and state
    _update_zobrist_ep_square(original_ep_square, new_ep_square)
    bitboards.EN_PASSANT_SQUARE = new_ep_square

    # Update castling rights
    new_castling_rights = _update_castling_rights(
        original_castling_rights, moving_piece_type, side_that_moved,
        from_square, to_square, captured_piece_type
    )
    _update_zobrist_castling_rights(original_castling_rights, new_castling_rights)
    bitboards.CASTLING_RIGHTS = new_castling_rights

    # Update side to move
    bitboards.SIDE_TO_MOVE = opponent_side
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_BLACK_TO_MOVE

    # Update full move counter if needed
    if opponent_side == chess.WHITE:
        bitboards.FULL_MOVE_COUNTER += 1

    # Append hash after all state changes
    POSITION_HISTORY.append(CURRENT_ZOBRIST_HASH)


def _handle_null_move():
    """Helper function to process null moves efficiently"""
    global CURRENT_ZOBRIST_HASH

    original_ep_square = bitboards.EN_PASSANT_SQUARE
    original_side_to_move = bitboards.SIDE_TO_MOVE

    # Store minimal info for undo
    BOARD_STATE_HISTORY.append({
        'move': chess.Move.null(),
        'captured_piece_type': None,
        'prev_castling_rights': bitboards.CASTLING_RIGHTS,
        'prev_ep_square': original_ep_square,
        'prev_halfmove_clock': bitboards.HALF_MOVE_COUNTER,
        'prev_zobrist_hash': CURRENT_ZOBRIST_HASH
    })

    # Update EP square, halfmove and Zobrist hash
    if original_ep_square is not None:
        CURRENT_ZOBRIST_HASH ^= ZOBRIST_EN_PASSANT_FILE[chess.square_file(original_ep_square)]
    bitboards.EN_PASSANT_SQUARE = None
    bitboards.HALF_MOVE_COUNTER += 1

    # Flip side to move
    bitboards.SIDE_TO_MOVE = not original_side_to_move
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_BLACK_TO_MOVE

    # Increment full move counter if needed
    if bitboards.SIDE_TO_MOVE == chess.WHITE:
        bitboards.FULL_MOVE_COUNTER += 1

    POSITION_HISTORY.append(CURRENT_ZOBRIST_HASH)


def _find_piece_type_fast(square, side):
    """Optimized version of _find_piece_at_square for better performance"""
    square_bit = 1 << square

    if side == chess.WHITE:
        if bitboards.WHITE_PAWNS & square_bit: return chess.PAWN
        if bitboards.WHITE_KNIGHTS & square_bit: return chess.KNIGHT
        if bitboards.WHITE_BISHOPS & square_bit: return chess.BISHOP
        if bitboards.WHITE_ROOKS & square_bit: return chess.ROOK
        if bitboards.WHITE_QUEENS & square_bit: return chess.QUEEN
        if bitboards.WHITE_KINGS & square_bit: return chess.KING
    else:
        if bitboards.BLACK_PAWNS & square_bit: return chess.PAWN
        if bitboards.BLACK_KNIGHTS & square_bit: return chess.KNIGHT
        if bitboards.BLACK_BISHOPS & square_bit: return chess.BISHOP
        if bitboards.BLACK_ROOKS & square_bit: return chess.ROOK
        if bitboards.BLACK_QUEENS & square_bit: return chess.QUEEN
        if bitboards.BLACK_KINGS & square_bit: return chess.KING
    return None


def _move_piece_fast(piece_type, side, from_bb, to_bb):
    """Fast piece movement without redundant checks"""
    if side == chess.WHITE:
        if piece_type == chess.PAWN:
            bitboards.WHITE_PAWNS = (bitboards.WHITE_PAWNS & ~from_bb) | to_bb
        elif piece_type == chess.KNIGHT:
            bitboards.WHITE_KNIGHTS = (bitboards.WHITE_KNIGHTS & ~from_bb) | to_bb
        elif piece_type == chess.BISHOP:
            bitboards.WHITE_BISHOPS = (bitboards.WHITE_BISHOPS & ~from_bb) | to_bb
        elif piece_type == chess.ROOK:
            bitboards.WHITE_ROOKS = (bitboards.WHITE_ROOKS & ~from_bb) | to_bb
        elif piece_type == chess.QUEEN:
            bitboards.WHITE_QUEENS = (bitboards.WHITE_QUEENS & ~from_bb) | to_bb
        elif piece_type == chess.KING:
            bitboards.WHITE_KINGS = (bitboards.WHITE_KINGS & ~from_bb) | to_bb
    else:  # BLACK
        if piece_type == chess.PAWN:
            bitboards.BLACK_PAWNS = (bitboards.BLACK_PAWNS & ~from_bb) | to_bb
        elif piece_type == chess.KNIGHT:
            bitboards.BLACK_KNIGHTS = (bitboards.BLACK_KNIGHTS & ~from_bb) | to_bb
        elif piece_type == chess.BISHOP:
            bitboards.BLACK_BISHOPS = (bitboards.BLACK_BISHOPS & ~from_bb) | to_bb
        elif piece_type == chess.ROOK:
            bitboards.BLACK_ROOKS = (bitboards.BLACK_ROOKS & ~from_bb) | to_bb
        elif piece_type == chess.QUEEN:
            bitboards.BLACK_QUEENS = (bitboards.BLACK_QUEENS & ~from_bb) | to_bb
        elif piece_type == chess.KING:
            bitboards.BLACK_KINGS = (bitboards.BLACK_KINGS & ~from_bb) | to_bb


def _handle_capture(square, opponent_side, to_bb):
    """Handle the capture of a piece, returns the captured piece type"""
    global CURRENT_ZOBRIST_HASH

    # Find which piece is being captured
    captured_piece_type = _find_piece_type_fast(square, opponent_side)

    if captured_piece_type is not None:
        # Update Zobrist hash for captured piece
        piece_index = PIECE_TO_ZOBRIST_INDEX[(captured_piece_type, opponent_side)]
        CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[piece_index][square]

        # Remove the captured piece
        if opponent_side == chess.WHITE:
            if captured_piece_type == chess.PAWN:
                bitboards.WHITE_PAWNS &= ~to_bb
            elif captured_piece_type == chess.KNIGHT:
                bitboards.WHITE_KNIGHTS &= ~to_bb
            elif captured_piece_type == chess.BISHOP:
                bitboards.WHITE_BISHOPS &= ~to_bb
            elif captured_piece_type == chess.ROOK:
                bitboards.WHITE_ROOKS &= ~to_bb
            elif captured_piece_type == chess.QUEEN:
                bitboards.WHITE_QUEENS &= ~to_bb
        else:  # BLACK
            if captured_piece_type == chess.PAWN:
                bitboards.BLACK_PAWNS &= ~to_bb
            elif captured_piece_type == chess.KNIGHT:
                bitboards.BLACK_KNIGHTS &= ~to_bb
            elif captured_piece_type == chess.BISHOP:
                bitboards.BLACK_BISHOPS &= ~to_bb
            elif captured_piece_type == chess.ROOK:
                bitboards.BLACK_ROOKS &= ~to_bb
            elif captured_piece_type == chess.QUEEN:
                bitboards.BLACK_QUEENS &= ~to_bb

    return captured_piece_type


def _handle_en_passant_capture(side_that_moved, to_square):
    """Handle en passant capture logic"""
    global CURRENT_ZOBRIST_HASH

    # Determine captured pawn square
    ep_captured_square = to_square - 8 if side_that_moved == chess.WHITE else to_square + 8
    ep_captured_bb = 1 << ep_captured_square

    # Remove the captured pawn and update hash
    if side_that_moved == chess.WHITE:
        bitboards.BLACK_PAWNS &= ~ep_captured_bb
        piece_index = PIECE_TO_ZOBRIST_INDEX[(chess.PAWN, chess.BLACK)]
    else:
        bitboards.WHITE_PAWNS &= ~ep_captured_bb
        piece_index = PIECE_TO_ZOBRIST_INDEX[(chess.PAWN, chess.WHITE)]

    CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[piece_index][ep_captured_square]


def _handle_promotion(promotion_piece, side, to_bb):
    """Handle pawn promotion logic"""
    global CURRENT_ZOBRIST_HASH

    # Remove pawn from target
    if side == chess.WHITE:
        bitboards.WHITE_PAWNS &= ~to_bb
        pawn_index = PIECE_TO_ZOBRIST_INDEX[(chess.PAWN, chess.WHITE)]
    else:
        bitboards.BLACK_PAWNS &= ~to_bb
        pawn_index = PIECE_TO_ZOBRIST_INDEX[(chess.PAWN, chess.BLACK)]

    # Update hash - remove pawn
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[pawn_index][chess.msb(to_bb)]

    # Add promoted piece
    if side == chess.WHITE:
        if promotion_piece == chess.QUEEN:
            bitboards.WHITE_QUEENS |= to_bb
        elif promotion_piece == chess.ROOK:
            bitboards.WHITE_ROOKS |= to_bb
        elif promotion_piece == chess.BISHOP:
            bitboards.WHITE_BISHOPS |= to_bb
        elif promotion_piece == chess.KNIGHT:
            bitboards.WHITE_KNIGHTS |= to_bb
    else:
        if promotion_piece == chess.QUEEN:
            bitboards.BLACK_QUEENS |= to_bb
        elif promotion_piece == chess.ROOK:
            bitboards.BLACK_ROOKS |= to_bb
        elif promotion_piece == chess.BISHOP:
            bitboards.BLACK_BISHOPS |= to_bb
        elif promotion_piece == chess.KNIGHT:
            bitboards.BLACK_KNIGHTS |= to_bb

    # Update hash - add promoted piece
    promoted_index = PIECE_TO_ZOBRIST_INDEX[(promotion_piece, side)]
    CURRENT_ZOBRIST_HASH ^= ZOBRIST_TABLE[promoted_index][chess.msb(to_bb)]


def _update_aggregate_bitboards():
    """Update the combined bitboards after a move"""
    bitboards.WHITE_PIECES = (bitboards.WHITE_PAWNS | bitboards.WHITE_KNIGHTS |
                             bitboards.WHITE_BISHOPS | bitboards.WHITE_ROOKS |
                             bitboards.WHITE_QUEENS | bitboards.WHITE_KINGS)
    bitboards.BLACK_PIECES = (bitboards.BLACK_PAWNS | bitboards.BLACK_KNIGHTS |
                             bitboards.BLACK_BISHOPS | bitboards.BLACK_ROOKS |
                             bitboards.BLACK_QUEENS | bitboards.BLACK_KINGS)
    bitboards.ALL_PIECES = bitboards.WHITE_PIECES | bitboards.BLACK_PIECES


def pop_move():
    """
    Undoes the last move by reversing the changes based on minimal history.
    Restores bitboards and game state incrementally.
    """
    global CURRENT_ZOBRIST_HASH, POSITION_HISTORY

    if not BOARD_STATE_HISTORY:
        print("Error: Cannot undo move, no history available.")
        return

    # Restore state from history
    last_state_info = BOARD_STATE_HISTORY.pop()
    move = last_state_info['move']
    captured_piece_type = last_state_info['captured_piece_type']

    # Remove the hash from position history
    if POSITION_HISTORY:
        POSITION_HISTORY.pop()
    else:
        print("Warning: Tried to pop from empty POSITION_HISTORY.")

    # Handle null move reversal
    if move == chess.Move.null():
        _undo_null_move(last_state_info)
        return

    # Restore saved state values
    bitboards.CASTLING_RIGHTS = last_state_info['prev_castling_rights']
    bitboards.EN_PASSANT_SQUARE = last_state_info['prev_ep_square']
    bitboards.HALF_MOVE_COUNTER = last_state_info['prev_halfmove_clock']
    CURRENT_ZOBRIST_HASH = last_state_info['prev_zobrist_hash']

    # Restore side to move
    bitboards.SIDE_TO_MOVE = not bitboards.SIDE_TO_MOVE
    side = bitboards.SIDE_TO_MOVE

    # Adjust full move counter
    if side == chess.WHITE:
        bitboards.FULL_MOVE_COUNTER -= 1

    # Extract move details
    from_square = move.from_square
    to_square = move.to_square
    from_bb = 1 << from_square
    to_bb = 1 << to_square
    promotion = move.promotion

    # Handle promotion first (special case)
    if promotion is not None:
        _undo_promotion(promotion, side, from_bb, to_bb)
    else:
        # Move the piece back from destination to source
        _undo_regular_move(side, from_bb, to_bb)

    # Restore captured piece if any
    if captured_piece_type is not None:
        _restore_captured_piece(move, captured_piece_type, side, to_bb, last_state_info)

    # Undo castling rook move if needed
    if (_find_piece_type_fast(from_square, side) == chess.KING and
            abs(from_square - to_square) == 2):
        _undo_castling(side, to_square)

    # Recalculate aggregate bitboards
    _update_aggregate_bitboards()


def _undo_null_move(last_state_info):
    """Helper to undo a null move"""
    global CURRENT_ZOBRIST_HASH

    # Restore state directly from saved history
    bitboards.EN_PASSANT_SQUARE = last_state_info['prev_ep_square']
    bitboards.HALF_MOVE_COUNTER = last_state_info['prev_halfmove_clock']
    CURRENT_ZOBRIST_HASH = last_state_info['prev_zobrist_hash']

    # Restore side to move
    bitboards.SIDE_TO_MOVE = not bitboards.SIDE_TO_MOVE

    # Decrement full move number if required
    if bitboards.SIDE_TO_MOVE == chess.WHITE:
        bitboards.FULL_MOVE_COUNTER -= 1


def _undo_promotion(promotion, side, from_bb, to_bb):
    """Helper to undo a promotion move"""
    if side == chess.WHITE:
        # Remove promoted piece
        if promotion == chess.QUEEN:
            bitboards.WHITE_QUEENS &= ~to_bb
        elif promotion == chess.ROOK:
            bitboards.WHITE_ROOKS &= ~to_bb
        elif promotion == chess.BISHOP:
            bitboards.WHITE_BISHOPS &= ~to_bb
        elif promotion == chess.KNIGHT:
            bitboards.WHITE_KNIGHTS &= ~to_bb
        # Put pawn back on original square
        bitboards.WHITE_PAWNS |= from_bb
    else:
        # Remove promoted piece
        if promotion == chess.QUEEN:
            bitboards.BLACK_QUEENS &= ~to_bb
        elif promotion == chess.ROOK:
            bitboards.BLACK_ROOKS &= ~to_bb
        elif promotion == chess.BISHOP:
            bitboards.BLACK_BISHOPS &= ~to_bb
        elif promotion == chess.KNIGHT:
            bitboards.BLACK_KNIGHTS &= ~to_bb
        # Put pawn back on original square
        bitboards.BLACK_PAWNS |= from_bb


def _undo_regular_move(side, from_bb, to_bb):
    """Helper to move a piece back from destination to source"""
    # Fast path: check bitboards directly
    if side == chess.WHITE:
        if bitboards.WHITE_PAWNS & to_bb:
            bitboards.WHITE_PAWNS = (bitboards.WHITE_PAWNS & ~to_bb) | from_bb
        elif bitboards.WHITE_KNIGHTS & to_bb:
            bitboards.WHITE_KNIGHTS = (bitboards.WHITE_KNIGHTS & ~to_bb) | from_bb
        elif bitboards.WHITE_BISHOPS & to_bb:
            bitboards.WHITE_BISHOPS = (bitboards.WHITE_BISHOPS & ~to_bb) | from_bb
        elif bitboards.WHITE_ROOKS & to_bb:
            bitboards.WHITE_ROOKS = (bitboards.WHITE_ROOKS & ~to_bb) | from_bb
        elif bitboards.WHITE_QUEENS & to_bb:
            bitboards.WHITE_QUEENS = (bitboards.WHITE_QUEENS & ~to_bb) | from_bb
        elif bitboards.WHITE_KINGS & to_bb:
            bitboards.WHITE_KINGS = (bitboards.WHITE_KINGS & ~to_bb) | from_bb
    else:  # BLACK
        if bitboards.BLACK_PAWNS & to_bb:
            bitboards.BLACK_PAWNS = (bitboards.BLACK_PAWNS & ~to_bb) | from_bb
        elif bitboards.BLACK_KNIGHTS & to_bb:
            bitboards.BLACK_KNIGHTS = (bitboards.BLACK_KNIGHTS & ~to_bb) | from_bb
        elif bitboards.BLACK_BISHOPS & to_bb:
            bitboards.BLACK_BISHOPS = (bitboards.BLACK_BISHOPS & ~to_bb) | from_bb
        elif bitboards.BLACK_ROOKS & to_bb:
            bitboards.BLACK_ROOKS = (bitboards.BLACK_ROOKS & ~to_bb) | from_bb
        elif bitboards.BLACK_QUEENS & to_bb:
            bitboards.BLACK_QUEENS = (bitboards.BLACK_QUEENS & ~to_bb) | from_bb
        elif bitboards.BLACK_KINGS & to_bb:
            bitboards.BLACK_KINGS = (bitboards.BLACK_KINGS & ~to_bb) | from_bb


def _restore_captured_piece(move, captured_piece_type, side, to_bb, last_state_info):
    """Helper to restore a captured piece"""
    # Check if it was en passant capture
    if (move.from_square & 7) != (move.to_square & 7):
        is_pawn_move = (_find_piece_type_fast(move.from_square, side) == chess.PAWN)
        is_ep_square = (move.to_square == last_state_info['prev_ep_square'])
        is_ep_capture = is_pawn_move and is_ep_square and last_state_info['prev_ep_square'] is not None
    else:
        is_ep_capture = False

    if is_ep_capture:
        # Restore the captured pawn to its original square
        captured_square = move.to_square - 8 if side == chess.WHITE else move.to_square + 8
        captured_bb = 1 << captured_square

        if side == chess.WHITE:
            bitboards.BLACK_PAWNS |= captured_bb
        else:
            bitboards.WHITE_PAWNS |= captured_bb
    else:
        # Regular capture - restore piece to the capture square
        opponent_side = not side
        if opponent_side == chess.WHITE:
            if captured_piece_type == chess.PAWN:
                bitboards.WHITE_PAWNS |= to_bb
            elif captured_piece_type == chess.KNIGHT:
                bitboards.WHITE_KNIGHTS |= to_bb
            elif captured_piece_type == chess.BISHOP:
                bitboards.WHITE_BISHOPS |= to_bb
            elif captured_piece_type == chess.ROOK:
                bitboards.WHITE_ROOKS |= to_bb
            elif captured_piece_type == chess.QUEEN:
                bitboards.WHITE_QUEENS |= to_bb
        else:  # BLACK
            if captured_piece_type == chess.PAWN:
                bitboards.BLACK_PAWNS |= to_bb
            elif captured_piece_type == chess.KNIGHT:
                bitboards.BLACK_KNIGHTS |= to_bb
            elif captured_piece_type == chess.BISHOP:
                bitboards.BLACK_BISHOPS |= to_bb
            elif captured_piece_type == chess.ROOK:
                bitboards.BLACK_ROOKS |= to_bb
            elif captured_piece_type == chess.QUEEN:
                bitboards.BLACK_QUEENS |= to_bb


def _undo_castling(side, to_square):
    """Helper to undo a castling move"""
    castling_key = (to_square, side)
    if castling_key in CASTLING_ROOK_MOVES:
        rook_from, rook_to = CASTLING_ROOK_MOVES[castling_key]

        # Move rook back to its original position
        if side == chess.WHITE:
            bitboards.WHITE_ROOKS = (bitboards.WHITE_ROOKS & ~(1 << rook_to)) | (1 << rook_from)
        else:
            bitboards.BLACK_ROOKS = (bitboards.BLACK_ROOKS & ~(1 << rook_to)) | (1 << rook_from)


def is_repetition(count=3):
    """Checks if the current position has occurred 'count' times."""
    if len(POSITION_HISTORY) <= 1:  # No repetitions possible with 0 or 1 entry
        return False

    if bitboards.HALF_MOVE_COUNTER < 4:  # Need at least 4 halfmoves for a 3-fold repetition
        return False

    # Create a dictionary for faster lookups with just the positions we need to check
    relevant_history_start = max(0, len(POSITION_HISTORY) - bitboards.HALF_MOVE_COUNTER - 1)
    relevant_history = POSITION_HISTORY[relevant_history_start:]

    # Count occurrences of current position
    from collections import Counter
    position_counts = Counter(relevant_history)

    return position_counts[CURRENT_ZOBRIST_HASH] >= count

# Example Usage (for testing)
if __name__ == "__main__":
    board = chess.Board()
    initialize_board_state(board)
    initial_calculated_hash = CURRENT_ZOBRIST_HASH

    print("Initial state:")
    from .bitboards import print_bitboard
    print_bitboard(bitboards.ALL_PIECES)
    print(f"Side to move: {'White' if bitboards.SIDE_TO_MOVE else 'Black'}")
    print(f"Initial Zobrist Hash: {initial_calculated_hash:016x}")


    move_e2e4 = chess.Move.from_uci("e2e4")
    print(f"\nMaking move: {move_e2e4.uci()}")
    push_move(move_e2e4)

    print("State after e2e4:")
    print_bitboard(bitboards.ALL_PIECES)
    print(f"Side to move: {'White' if bitboards.SIDE_TO_MOVE else 'Black'}")
    print(f"Zobrist Hash after e2e4: {CURRENT_ZOBRIST_HASH:016x}")

    move_d7d5 = chess.Move.from_uci("d7d5")
    print(f"\nMaking move: {move_d7d5.uci()}")
    push_move(move_d7d5)

    print("State after d7d5:")
    print_bitboard(bitboards.ALL_PIECES)
    print(f"Side to move: {'White' if bitboards.SIDE_TO_MOVE else 'Black'}")
    print(f"Zobrist Hash after d7d5: {CURRENT_ZOBRIST_HASH:016x}")


    # Undo the last move
    print("\nUndoing last move (d7d5):")
    pop_move()

    print("State after undoing last move:")
    print_bitboard(bitboards.ALL_PIECES)
    print(f"Side to move: {'White' if bitboards.SIDE_TO_MOVE else 'Black'}")
    print(f"Zobrist Hash after undoing last move: {CURRENT_ZOBRIST_HASH:016x}")


    # Undo the first move
    print("\nUndoing first move (e2e4):")
    pop_move()
    restored_hash = CURRENT_ZOBRIST_HASH # Get the hash after undoing moves

    print("State after undoing first move:")
    print_bitboard(bitboards.ALL_PIECES)
    print(f"Side to move: {'White' if bitboards.SIDE_TO_MOVE else 'Black'}")
    print(f"Zobrist Hash after undoing first move: {restored_hash:016x}")

    # Verify hash by comparing restored hash with the initially calculated one
    print(f"\nInitial calculated Zobrist Hash: {initial_calculated_hash:016x}")
    if initial_calculated_hash == restored_hash:
        print("Restored hash matches initially calculated hash.")
    else:
        print("**** HASH MISMATCH AFTER FULL UNDO ****")
        print(f"Restored:   {restored_hash:016x}")
        print(f"Initial:    {initial_calculated_hash:016x}")
        board_reset = chess.Board()
        recalculated_hash_scratch = calculate_full_zobrist_hash(board_reset)
        print(f"Recalculated from scratch: {recalculated_hash_scratch:016x}")
