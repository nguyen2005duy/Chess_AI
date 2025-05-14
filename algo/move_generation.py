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
        from_square = target_square + offset if offset else chess.lsb(bb)
        if offset:
            to_square = target_square
        else:
            to_square = chess.lsb(bb >> 1) if bb & (bb - 1) else target_square
            bb >>= 1
        add_move_func(moves, from_square, to_square)
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
        # Pawns that could potentially capture en passant
        if color == chess.WHITE:
            # White pawns on rank 5 that can capture to the EP square
            potential_ep_attackers = pawns & ep_rank_bb
            ep_attackers_left = west(ep_target_bb) & potential_ep_attackers
            ep_attackers_right = east(ep_target_bb) & potential_ep_attackers

            while ep_attackers_left:
                from_square = chess.lsb(ep_attackers_left)
                moves.append(chess.Move(from_square, bitboards.EN_PASSANT_SQUARE))
                ep_attackers_left &= ep_attackers_left - 1

            while ep_attackers_right:
                from_square = chess.lsb(ep_attackers_right)
                moves.append(chess.Move(from_square, bitboards.EN_PASSANT_SQUARE))
                ep_attackers_right &= ep_attackers_right - 1
        else:
            # Black pawns on rank 4 that can capture to the EP square
            potential_ep_attackers = pawns & ep_rank_bb
            ep_attackers_left = west(ep_target_bb) & potential_ep_attackers
            ep_attackers_right = east(ep_target_bb) & potential_ep_attackers

            while ep_attackers_left:
                from_square = chess.lsb(ep_attackers_left)
                moves.append(chess.Move(from_square, bitboards.EN_PASSANT_SQUARE))
                ep_attackers_left &= ep_attackers_left - 1

            while ep_attackers_right:
                from_square = chess.lsb(ep_attackers_right)
                moves.append(chess.Move(from_square, bitboards.EN_PASSANT_SQUARE))
                ep_attackers_right &= ep_attackers_right - 1


def _generate_piece_moves(piece_bb, attack_func, friendly_pieces, moves):
    """Helper to generate moves for a specific piece type."""
    while piece_bb:
        from_square = chess.lsb(piece_bb)
        targets = attack_func(from_square, bitboards.ALL_PIECES) & ~friendly_pieces

        # Process each target square for this piece
        while targets:
            to_square = chess.lsb(targets)
            moves.append(chess.Move(from_square, to_square))
            targets &= targets - 1

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
    _generate_piece_moves(queens, lambda sq, pieces: get_bishop_attack(sq, pieces) | get_rook_attack(sq, pieces),
                          friendly_pieces, moves)

    # King moves (including castling)
    kings = bitboards.WHITE_KINGS if current_side == chess.WHITE else bitboards.BLACK_KINGS
    if kings:
        king_square = chess.lsb(kings)
        # Regular king moves
        targets = KING_ATTACKS[king_square] & ~friendly_pieces
        while targets:
            to_square = chess.lsb(targets)
            moves.append(chess.Move(king_square, to_square))
            targets &= targets - 1

        # Castling - check rights first to avoid unnecessary work
        if bitboards.CASTLING_RIGHTS:
            opponent_side = chess.BLACK if current_side == chess.WHITE else chess.WHITE

            if current_side == chess.WHITE:
                # Kingside castling (E1G1)
                if (bitboards.CASTLING_RIGHTS & chess.BB_H1 and
                        not (bitboards.ALL_PIECES & (chess.BB_F1 | chess.BB_G1)) and
                        not is_square_attacked(chess.F1, chess.BLACK) and
                        not is_square_attacked(chess.G1, chess.BLACK) and
                        not is_square_attacked(king_square, chess.BLACK)):
                    moves.append(chess.Move.from_uci("e1g1"))

                # Queenside castling (E1C1)
                if (bitboards.CASTLING_RIGHTS & chess.BB_A1 and
                        not (bitboards.ALL_PIECES & (chess.BB_B1 | chess.BB_C1 | chess.BB_D1)) and
                        not is_square_attacked(chess.D1, chess.BLACK) and
                        not is_square_attacked(chess.C1, chess.BLACK) and
                        not is_square_attacked(king_square, chess.BLACK)):
                    moves.append(chess.Move.from_uci("e1c1"))
            else:
                # Kingside castling (E8G8)
                if (bitboards.CASTLING_RIGHTS & chess.BB_H8 and
                        not (bitboards.ALL_PIECES & (chess.BB_F8 | chess.BB_G8)) and
                        not is_square_attacked(chess.F8, chess.WHITE) and
                        not is_square_attacked(chess.G8, chess.WHITE) and
                        not is_square_attacked(king_square, chess.WHITE)):
                    moves.append(chess.Move.from_uci("e8g8"))

                # Queenside castling (E8C8)
                if (bitboards.CASTLING_RIGHTS & chess.BB_A8 and
                        not (bitboards.ALL_PIECES & (chess.BB_B8 | chess.BB_C8 | chess.BB_D8)) and
                        not is_square_attacked(chess.D8, chess.WHITE) and
                        not is_square_attacked(chess.C8, chess.WHITE) and
                        not is_square_attacked(king_square, chess.WHITE)):
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

    # Generate and filter pseudo-legal moves
    pseudo_moves = generate_pseudo_legal_moves()
    for move in pseudo_moves:
        # Save state before making the move
        board_state.push_move(move)

        # Check if this move is legal (doesn't leave king in check)
        # If the king moved, we need to find its new location
        if move.from_square == king_square:
            # Update king position after move
            new_kings_bb = bitboards.WHITE_KINGS if current_side == chess.WHITE else bitboards.BLACK_KINGS
            new_king_square = chess.lsb(new_kings_bb)
            is_legal = not is_square_attacked(new_king_square, opponent_side)
        else:
            # King didn't move, check if it's still safe
            is_legal = not is_square_attacked(king_square, opponent_side)

        # If the move is legal, add it to our list
        if is_legal:
            legal_moves.append(move)

        # Restore the board state
        board_state.pop_move()

    return legal_moves


