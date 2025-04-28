import chess
from . import bitboards
from .magic_bitboards import KING_ATTACKS, KNIGHT_ATTACKS, get_rook_attack, get_bishop_attack, is_square_attacked

# --- Constants ---
FULL_MASK = (1 << 64) - 1
NOT_A_FILE = ~chess.BB_FILE_A
NOT_H_FILE = ~chess.BB_FILE_H

# --- Optimized Bitboard Shift Helpers ---
def north(bb): return bb << 8
def south(bb): return bb >> 8
def east(bb): return (bb << 1) & NOT_A_FILE
def west(bb): return (bb >> 1) & NOT_H_FILE
def north_east(bb): return (bb << 9) & NOT_A_FILE
def north_west(bb): return (bb << 7) & NOT_H_FILE
def south_east(bb): return (bb >> 7) & NOT_A_FILE
def south_west(bb): return (bb >> 9) & NOT_H_FILE


def _add_pawn_move(moves, from_sq, to_sq):
    """Helper to add pawn moves, handling promotions."""
    if chess.square_rank(to_sq) in (0, 7):
        # Add promotions in order of likely value (queen first)
        moves.extend([
            chess.Move(from_sq, to_sq, promotion=chess.QUEEN),
            chess.Move(from_sq, to_sq, promotion=chess.KNIGHT),
            chess.Move(from_sq, to_sq, promotion=chess.ROOK),
            chess.Move(from_sq, to_sq, promotion=chess.BISHOP)
        ])
    else:
        moves.append(chess.Move(from_sq, to_sq))


def _process_bitboard_moves(bb, offset, moves, add_move_func):
    """Helper to process bitboard moves efficiently."""
    while bb:
        target_square = chess.lsb(bb)
        from_square = target_square + offset
        add_move_func(moves, from_square, target_square)
        bb &= bb - 1


def _generate_pawn_moves(moves, color, empty_squares, opponent_pieces):
    """Optimized function to generate pseudo-legal pawn moves."""
    # Prepare direction-specific variables based on color
    if color == chess.WHITE:
        pawns = bitboards.WHITE_PAWNS
        single_push_target = north(pawns) & empty_squares
        double_push_target = north(single_push_target & chess.BB_RANK_3) & empty_squares
        captures_left_target = north_west(pawns) & opponent_pieces
        captures_right_target = north_east(pawns) & opponent_pieces
        single_push_offset, double_push_offset = -8, -16
        capture_left_offset, capture_right_offset = -7, -9
        ep_rank_bb = chess.BB_RANK_5
    else:
        pawns = bitboards.BLACK_PAWNS
        single_push_target = south(pawns) & empty_squares
        double_push_target = south(single_push_target & chess.BB_RANK_6) & empty_squares
        captures_left_target = south_west(pawns) & opponent_pieces
        captures_right_target = south_east(pawns) & opponent_pieces
        single_push_offset, double_push_offset = 8, 16
        capture_left_offset, capture_right_offset = 9, 7
        ep_rank_bb = chess.BB_RANK_4

    # Process move types using optimized helper
    _process_bitboard_moves(single_push_target, single_push_offset, moves, _add_pawn_move)
    _process_bitboard_moves(double_push_target, double_push_offset, moves, lambda m, f, t: m.append(chess.Move(f, t)))
    _process_bitboard_moves(captures_left_target, capture_left_offset, moves, _add_pawn_move)
    _process_bitboard_moves(captures_right_target, capture_right_offset, moves, _add_pawn_move)

    # En passant - only process if an EP square exists
    if bitboards.EN_PASSANT_SQUARE is not None:
        ep_target_bb = 1 << bitboards.EN_PASSANT_SQUARE
        attacking_pawns = pawns & ep_rank_bb

        if attacking_pawns:  # Only continue if there are pawns on the right rank
            if color == chess.WHITE:
                attackers = ((ep_target_bb >> 9) & NOT_H_FILE & attacking_pawns) | \
                           ((ep_target_bb >> 7) & NOT_A_FILE & attacking_pawns)
            else:
                attackers = ((ep_target_bb << 9) & NOT_A_FILE & attacking_pawns) | \
                           ((ep_target_bb << 7) & NOT_H_FILE & attacking_pawns)

            _process_bitboard_moves(attackers, 0, moves, lambda m, f, t: m.append(chess.Move(f, bitboards.EN_PASSANT_SQUARE)))


def _generate_piece_moves(piece_bb, attack_func, friendly_pieces, moves):
    """Helper to generate moves for a specific piece type."""
    while piece_bb:
        from_square = chess.lsb(piece_bb)
        targets = attack_func(from_square, bitboards.ALL_PIECES) & ~friendly_pieces
        _process_bitboard_moves(targets, 0, moves, lambda m, f, t: m.append(chess.Move(from_square, t)))
        piece_bb &= piece_bb - 1


def generate_pseudo_legal_moves():
    """
    Optimized function to generate all pseudo-legal moves for the current side to move.
    """
    # Pre-allocate moves list with a reasonable capacity hint
    moves = []
    current_side = bitboards.SIDE_TO_MOVE
    friendly_pieces = bitboards.WHITE_PIECES if current_side == chess.WHITE else bitboards.BLACK_PIECES
    opponent_pieces = bitboards.BLACK_PIECES if current_side == chess.WHITE else bitboards.WHITE_PIECES
    empty_squares = (~bitboards.ALL_PIECES) & FULL_MASK

    # Knight moves
    knights = bitboards.WHITE_KNIGHTS if current_side == chess.WHITE else bitboards.BLACK_KNIGHTS
    _generate_piece_moves(knights, lambda sq, _: KNIGHT_ATTACKS[sq], friendly_pieces, moves)

    # Bishop moves
    bishops = bitboards.WHITE_BISHOPS if current_side == chess.WHITE else bitboards.BLACK_BISHOPS
    _generate_piece_moves(bishops, get_bishop_attack, friendly_pieces, moves)

    # Rook moves
    rooks = bitboards.WHITE_ROOKS if current_side == chess.WHITE else bitboards.BLACK_ROOKS
    _generate_piece_moves(rooks, get_rook_attack, friendly_pieces, moves)

    # Queen moves
    queens = bitboards.WHITE_QUEENS if current_side == chess.WHITE else bitboards.BLACK_QUEENS
    _generate_piece_moves(queens, lambda sq, pieces: get_bishop_attack(sq, pieces) | get_rook_attack(sq, pieces), friendly_pieces, moves)

    # King moves (including castling)
    kings = bitboards.WHITE_KINGS if current_side == chess.WHITE else bitboards.BLACK_KINGS
    if kings:
        king_square = chess.lsb(kings)
        # Regular king moves
        targets = KING_ATTACKS[king_square] & ~friendly_pieces
        _process_bitboard_moves(targets, 0, moves, lambda m, f, t: m.append(chess.Move(king_square, t)))

        # Castling - only check if king is not in check
        opponent_side = chess.BLACK if current_side == chess.WHITE else chess.WHITE
        if not is_square_attacked(king_square, opponent_side):
            if current_side == chess.WHITE:
                # Kingside castling (E1G1)
                if (bitboards.CASTLING_RIGHTS & chess.BB_H1 and
                    not (bitboards.ALL_PIECES & (chess.BB_F1 | chess.BB_G1)) and
                    not is_square_attacked(chess.F1, chess.BLACK) and
                    not is_square_attacked(chess.G1, chess.BLACK)):
                    moves.append(chess.Move.from_uci("e1g1"))
                # Queenside castling (E1C1)
                if (bitboards.CASTLING_RIGHTS & chess.BB_A1 and
                    not (bitboards.ALL_PIECES & (chess.BB_B1 | chess.BB_C1 | chess.BB_D1)) and
                    not is_square_attacked(chess.D1, chess.BLACK) and
                    not is_square_attacked(chess.C1, chess.BLACK)):
                    moves.append(chess.Move.from_uci("e1c1"))
            else:
                # Kingside castling (E8G8)
                if (bitboards.CASTLING_RIGHTS & chess.BB_H8 and
                    not (bitboards.ALL_PIECES & (chess.BB_F8 | chess.BB_G8)) and
                    not is_square_attacked(chess.F8, chess.WHITE) and
                    not is_square_attacked(chess.G8, chess.WHITE)):
                    moves.append(chess.Move.from_uci("e8g8"))
                # Queenside castling (E8C8)
                if (bitboards.CASTLING_RIGHTS & chess.BB_A8 and
                    not (bitboards.ALL_PIECES & (chess.BB_B8 | chess.BB_C8 | chess.BB_D8)) and
                    not is_square_attacked(chess.D8, chess.WHITE) and
                    not is_square_attacked(chess.C8, chess.WHITE)):
                    moves.append(chess.Move.from_uci("e8c8"))

    # Pawn moves - generate these last for better move ordering
    _generate_pawn_moves(moves, current_side, empty_squares, opponent_pieces)

    return moves


def generate_legal_moves():
    """
    Optimized function to generate all legal moves by filtering pseudo-legal moves
    that would leave the king in check.
    """
    from . import board_state

    # Get current state information
    current_side = bitboards.SIDE_TO_MOVE
    opponent_side = chess.BLACK if current_side == chess.WHITE else chess.WHITE
    kings_bb = bitboards.WHITE_KINGS if current_side == chess.WHITE else bitboards.BLACK_KINGS

    # Early return if no king (shouldn't happen in normal chess)
    if not kings_bb:
        return []

    king_square = chess.lsb(kings_bb)
    king_in_check = is_square_attacked(king_square, opponent_side)

    # Optimization: Pre-allocate with estimated capacity
    legal_moves = []

    # Handle check evasion separately for better performance
    pseudo_moves = generate_pseudo_legal_moves()
    for move in pseudo_moves:
        board_state.push_move(move)

        # Check legality depending on whether king moved
        if move.from_square == king_square:
            new_kings_bb = bitboards.WHITE_KINGS if current_side == chess.WHITE else bitboards.BLACK_KINGS
            new_king_square = chess.lsb(new_kings_bb)
            is_legal = not is_square_attacked(new_king_square, opponent_side)
        else:
            is_legal = not is_square_attacked(king_square, opponent_side)

        if is_legal:
            legal_moves.append(move)

        board_state.pop_move()

    return legal_moves


# Example Usage (for testing)
if __name__ == "__main__":
    import time
    from collections import defaultdict

    # This example still uses python-chess board for initialization
    # A true bitboard engine would manage its state purely with bitboards
    board = chess.Board()
    # Import initialization function correctly
    from .board_state import initialize_board_state
    initialize_board_state(board) # Use board_state init for consistency

    def categorize_moves(moves_list):
        """Categorize moves by piece type and move type"""
        result = defaultdict(int)
        captures = 0
        checks = 0
        castling = 0
        promotions = 0

        for move in moves_list:
            piece = board.piece_at(move.from_square)
            if piece:
                result[piece.symbol()] += 1

            # Count special move types
            if board.is_capture(move):
                captures += 1
            if board.gives_check(move):
                checks += 1
            if board.is_castling(move):
                castling += 1
            if move.promotion:
                promotions += 1

        return result, captures, checks, castling, promotions

    def validate_moves(bitboard_moves, title):
        """Compare with python-chess legal moves and report differences"""
        start_time = time.time()

        # Get python-chess legal moves for comparison
        py_legal_moves = list(board.legal_moves)

        # Convert both to sets of UCI strings for comparison
        bb_uci_set = {m.uci() for m in bitboard_moves}
        py_uci_set = {m.uci() for m in py_legal_moves}

        # Find missing and extra moves
        missing_moves = py_uci_set - bb_uci_set
        extra_moves = bb_uci_set - py_uci_set

        # Stats by piece type
        categories, captures, checks, castles, promos = categorize_moves(bitboard_moves)

        # Performance measurement
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # in milliseconds

        # Report results
        print(f"\n=== {title} ===")
        print(f"Total moves generated: {len(bitboard_moves)}")
        print(f"Python-chess moves:    {len(py_legal_moves)}")
        print(f"Accuracy: {100 - (len(missing_moves) + len(extra_moves)) / max(1, len(py_legal_moves)) * 100:.2f}%")
        print(f"Time taken: {duration:.2f} ms")

        if missing_moves:
            print(f"MISSING MOVES ({len(missing_moves)}): {', '.join(sorted(missing_moves))}")
        if extra_moves:
            print(f"EXTRA MOVES ({len(extra_moves)}): {', '.join(sorted(extra_moves))}")

        print("Move breakdown:")
        for piece, count in sorted(categories.items()):
            print(f"  {piece}: {count}")
        print(f"Captures: {captures}, Checks: {checks}, Castling: {castles}, Promotions: {promos}")

        return len(missing_moves) == 0 and len(extra_moves) == 0

    print("--- Testing Pseudo-Legal Moves ---")
    print("Pseudo-legal moves from initial position:")
    pseudo_moves = generate_pseudo_legal_moves()
    print(f"Count: {len(pseudo_moves)}")

    # Test pseudo-legal against legal to see what's being filtered
    legal_from_standard = generate_legal_moves()
    invalid_pseudo = [m.uci() for m in pseudo_moves if m not in legal_from_standard]
    if invalid_pseudo:
        print(f"Pseudo-legal moves that aren't legal: {len(invalid_pseudo)}")
        print(f"Examples: {', '.join(invalid_pseudo[:5])}...")

    board.push_san("e4")
    initialize_board_state(board) # Use board_state init
    print("\nPseudo-legal moves after e4:")
    pseudo_moves_e4 = generate_pseudo_legal_moves()
    print(f"Count: {len(pseudo_moves_e4)}")

    print("\n--- Testing Legal Moves ---")
    # Reset to initial position
    board = chess.Board()
    initialize_board_state(board)

    print("Legal moves from initial position:")
    legal_moves_start = generate_legal_moves()
    all_correct = validate_moves(legal_moves_start, "Starting Position")

    # Test a position with en passant
    fen_ep = "rnbqkbnr/ppp2ppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    board = chess.Board(fen_ep)
    initialize_board_state(board)
    legal_moves_ep = generate_legal_moves()
    ep_correct = validate_moves(legal_moves_ep, "En Passant Position")
    # Specifically check for the e5d6 en passant move
    ep_move = chess.Move.from_uci("e5d6")
    if ep_move in legal_moves_ep:
        print("✓ En passant move correctly generated")
    else:
        print("✗ En passant move missing!")

    # Test castling rights
    fen_castle = "r3k2r/pppqbppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQBPPP/R3K2R w KQkq - 4 8"
    board = chess.Board(fen_castle)
    initialize_board_state(board)
    legal_moves_castle = generate_legal_moves()
    castle_correct = validate_moves(legal_moves_castle, "Castling Position")
    # Check for specific castling moves
    if chess.Move.from_uci("e1g1") in legal_moves_castle and chess.Move.from_uci("e1c1") in legal_moves_castle:
        print("✓ White castling moves correctly generated")
    else:
        print("✗ White castling moves missing or incorrect!")

    # Test a position where king safety matters (mate in 1)
    fen_mate_in_1 = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = chess.Board(fen_mate_in_1)
    initialize_board_state(board)
    legal_moves_mate = generate_legal_moves()
    mate_correct = validate_moves(legal_moves_mate, "Mate-in-1 Position")
    # Check for the Qf7# move
    mate_move = chess.Move.from_uci("h5f7")
    if mate_move in legal_moves_mate:
        print("✓ Checkmate move correctly generated")
    else:
        print("✗ Checkmate move missing!")

    # Test a position with pins
    fen_pinned = "rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3 7"
    board = chess.Board(fen_pinned)
    initialize_board_state(board)
    legal_moves_pinned = generate_legal_moves()
    pin_correct = validate_moves(legal_moves_pinned, "Pinned Knight Position")

    # Check specifically that pinned knight can't move
    knight_moves = [m for m in legal_moves_pinned if board.piece_at(m.from_square) and
                   board.piece_at(m.from_square).piece_type == chess.KNIGHT]
    if any(m.from_square == chess.C3 for m in knight_moves):
        print("✗ ERROR: Pinned knight at c3 can still move!")
    else:
        print("✓ Pinned knight correctly restricted")

    # Display all legal moves in the pinned position for visual inspection
    print("\nLegal moves in pinned position:")
    for move in sorted(legal_moves_pinned, key=lambda m: m.uci()):
        print(move.uci(), end=" ")
    print() # Newline

    # Overall success summary
    if all([all_correct, ep_correct, castle_correct, mate_correct, pin_correct]):
        print("\n✓ All move generation tests PASSED!")
    else:
        print("\n✗ Some move generation tests FAILED!")
