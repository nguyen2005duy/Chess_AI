import chess, math
import time
from .Opening_Book import *
from . import bitboards # Access global state like SIDE_TO_MOVE
from . import board_state # Access global state like CURRENT_ZOBRIST_HASH
from .board_state import push_move, pop_move, initialize_board_state, is_repetition # Make/Unmake, Repetition Check
from .evaluation import calculate_heuristic_score, MATE_SCORE, DRAW_SCORE, MATE_THRESHOLD, MAX_PLY, is_insufficient_material_draw # Eval, Constants, Draw Check
from .move_generation import generate_legal_moves # Legal move generation
from .move_ordering import order_moves, get_piece_type_at # Move ordering (includes MVV-LVA), piece type helper
from .magic_bitboards import is_square_attacked # For check detection
import chess.syzygy # For EGTB
import chess.polyglot # For opening book
from .transposition_table import ( # Import TT functions and flags
    probe_tt, record_tt, clear_tt, get_tt_stats,
    transposition_table, # <-- Import the dictionary itself
    HASH_FLAG_EXACT, HASH_FLAG_LOWERBOUND, HASH_FLAG_UPPERBOUND
)

from .move_ordering import update_killer_moves, update_history_heuristic # Import heuristic updates

# --- Constants ---
CHECKMATE_SCORE = MATE_SCORE # For clarity

# --- Search Enhancement Constants ---
NMP_REDUCTION = 4       # Depth reduction for Null Move Pruning
NMP_MIN_DEPTH = 3       # Minimum depth to apply NMP
LMR_BASE_DEPTH = 3      # Minimum depth to apply Late Move Reductions
LMR_MIN_MOVE_COUNT = 3  # Number of moves searched at full depth before reducing
LMR_REDUCTION_FACTOR = 0.8 # Factor for calculating LMR
TIME_CHECK_INTERVAL = 4096 # Check time every N nodes
FUTILITY_MARGIN = 150   # Margin for futility pruning
FUTILITY_DEPTH = 3      # Maximum depth for futility pruning
RAZORING_MARGIN = 350   # Margin for razoring
SEE_CAPTURE_MARGIN = -20 # Static Exchange Evaluation threshold for captures
ASPIRATION_WINDOW_SIZE = 25  # Initial aspiration window size 
IID_DEPTH = 4           # Minimum depth for Internal Iterative Deepening 

# --- EGTB Configuration ---
# Specify the path to your Syzygy tablebase files directory
TABLEBASE_PATH = "syzygy"
TABLEBASE_MAX_PIECES = 0 #
SYZYGY_TABLEBASE = None


try:
    SYZYGY_TABLEBASE = chess.syzygy.open_tablebase(TABLEBASE_PATH)
    TABLEBASE_MAX_PIECES = 5 
    print(f"Syzygy tablebases opened from: {TABLEBASE_PATH} (Assumed Max Pieces: {TABLEBASE_MAX_PIECES})")
except FileNotFoundError:
    print(f"Warning: Syzygy tablebases directory not found at {TABLEBASE_PATH}. Search will proceed without EGTB.")
    SYZYGY_TABLEBASE = None
    TABLEBASE_MAX_PIECES = 0
except Exception as e:
    print(f"Warning: Error opening Syzygy tablebases: {e}")
    SYZYGY_TABLEBASE = None
    TABLEBASE_MAX_PIECES = 0


def _create_board_from_bitboards():
    """ Creates a python-chess Board object from the current global bitboard state. """
    board = chess.Board.empty() # Create empty board

    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        # White pieces
        w_bb_name = f"WHITE_{chess.piece_name(piece_type).upper()}S"
        w_bb = getattr(bitboards, w_bb_name, 0)
        for square in chess.scan_forward(w_bb):
            board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
        # Black pieces
        b_bb_name = f"BLACK_{chess.piece_name(piece_type).upper()}S"
        b_bb = getattr(bitboards, b_bb_name, 0)
        for square in chess.scan_forward(b_bb):
            board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))

    # Set state variables
    board.turn = bitboards.SIDE_TO_MOVE
    board.castling_rights = bitboards.CASTLING_RIGHTS
    board.ep_square = bitboards.EN_PASSANT_SQUARE
    board.halfmove_clock = bitboards.HALF_MOVE_COUNTER
    board.fullmove_number = bitboards.FULL_MOVE_COUNTER

    return board

# --- Search Statistics ---
nodes_searched = 0
search_start_time_global = 0.0 
time_limit_global = 0.0
stop_search = False

# --- Custom Exception for Time Limit ---
class TimeUpException(Exception):
    pass

# --- Helper Functions ---
def get_evaluation():
    """ Calculates heuristic score adjusted for the current side to move. """
    raw_score = calculate_heuristic_score()
    perspective_multiplier = 1 if bitboards.SIDE_TO_MOVE == chess.WHITE else -1
    return raw_score * perspective_multiplier

def static_exchange_evaluation(move):
    """
    Simplified Static Exchange Evaluation (SEE).
    Returns an estimate if the move gains material.
    Positive value = material gain, negative = material loss.
    """
    opponent_pieces = bitboards.BLACK_PIECES if bitboards.SIDE_TO_MOVE == chess.WHITE else bitboards.WHITE_PIECES
    is_capture = (opponent_pieces >> move.to_square) & 1

    if not is_capture and not move.promotion:
        return 0  

    # Determine victim piece value
    victim_value = 0
    if is_capture:
        victim_type = get_piece_type_at(move.to_square)
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        if victim_type is not None:
            victim_value = piece_values.get(victim_type, 0)

    # Add promotion value if applicable
    promotion_value = 0
    if move.promotion:
        # Value gained from promotion (queen - pawn)
        promo_piece_values = {
            chess.QUEEN: 800,
            chess.ROOK: 400,  
            chess.BISHOP: 230,  
            chess.KNIGHT: 200  
        }
        promotion_value = promo_piece_values.get(move.promotion, 0)

    # Attacker piece value
    attacker_type = get_piece_type_at(move.from_square)
    attacker_value = 0
    attacker_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0  
    }

    if attacker_type is not None:
        attacker_value = attacker_values.get(attacker_type, 0)

    # Simple estimation - could be refined with full exchange sequence analysis
    # Consider the balance of the exchange
    if is_capture:
        return victim_value - (attacker_value if victim_value < attacker_value else 0)
    else:
        return promotion_value  # For pure promotions, just return the gain value

def is_check():
    """Helper function to check if the current side is in check"""
    king_square = chess.lsb(bitboards.WHITE_KINGS if bitboards.SIDE_TO_MOVE == chess.WHITE else bitboards.BLACK_KINGS)
    return is_square_attacked(king_square, not bitboards.SIDE_TO_MOVE)

def should_stop_search():
    """Check if we should stop the search based on time or stop flag"""
    global stop_search
    if stop_search:
        return True
    if (nodes_searched & (TIME_CHECK_INTERVAL - 1)) == 0:
        if time.time() - search_start_time_global > time_limit_global:
            return True
    return False

# --- Quiescence Search (Phase 3.3 - No TT integration needed here typically) ---
def quiescence_search(ply, alpha, beta):
    """
    Searches capture sequences to stabilize the evaluation at the horizon.
    """
    global nodes_searched
    nodes_searched += 1 # Increment main counter as well

    # --- Time Check ---
    if should_stop_search():
        raise TimeUpException()

    # --- Draw Checks ---
    if ply > 0 and (is_repetition(2) or bitboards.HALF_MOVE_COUNTER >= 100 or is_insufficient_material_draw()):
        return DRAW_SCORE

    # Stand-pat score (evaluation before considering captures)
    stand_pat_score = get_evaluation()

    # Delta pruning - if even capturing the most valuable piece wouldn't improve alpha, exit early
    if stand_pat_score + 900 < alpha:  # 900 = queen value
        return alpha

    # Fail-high (Beta cutoff)
    if stand_pat_score >= beta:
        return beta

    # Update alpha
    if stand_pat_score > alpha:
        alpha = stand_pat_score

    # Generate legal moves and filter for captures/promotions
    legal_moves = generate_legal_moves()
    noisy_moves = []
    for m in legal_moves:
        # Determine if this is a capture or promotion
        opponent_pieces = bitboards.BLACK_PIECES if bitboards.SIDE_TO_MOVE == chess.WHITE else bitboards.WHITE_PIECES
        is_simple_capture = (opponent_pieces >> m.to_square) & 1
        moving_piece_type = get_piece_type_at(m.from_square)
        is_ep_capture = (moving_piece_type == chess.PAWN and
                         m.to_square == bitboards.EN_PASSANT_SQUARE and
                         bitboards.EN_PASSANT_SQUARE is not None)

        if is_simple_capture or is_ep_capture or m.promotion is not None:
            noisy_moves.append(m)

    # Order noisy moves (MVV-LVA for captures, promotions scored high)
    ordered_noisy_moves = order_moves(noisy_moves, ply, None) # Pass ply, no TT hint needed

    for move in ordered_noisy_moves:
        # SEE pruning for bad captures - improved margin based on depth
        see_score = static_exchange_evaluation(move)
        if see_score < SEE_CAPTURE_MARGIN:
            continue

        push_move(move)
        score = -quiescence_search(ply + 1, -beta, -alpha)
        pop_move()

        # Fail-high (Beta cutoff)
        if score >= beta:
            return beta

        # Update alpha
        if score > alpha:
            alpha = score

    return alpha    

# --- Internal Iterative Deepening (for when no TT move is available) ---
def internal_iterative_deepening(depth, ply, alpha, beta, is_pv_node):
    """
    Performs a shallow search to find a good move to try first when no TT move is available.
    Returns a move that can be used as a "best guess".
    """
    if depth < IID_DEPTH or ply > IID_DEPTH:
        return None  # Only use IID at sufficient depth and close to root

    # Perform a reduced depth search
    iid_depth = max(depth // 2, 1)

    # Run the reduced depth search to get the best move into TT
    negamax_search(iid_depth, ply, alpha, beta, is_pv_node, True)

    # After search, get the TT move that was stored
    tt_entry = transposition_table.get(board_state.CURRENT_ZOBRIST_HASH)
    if tt_entry and tt_entry.best_move:
        return tt_entry.best_move

    return None

# --- Negamax Search Part 1: Pruning and Early Exits ---
def negamax_pruning(depth, ply, alpha, beta, is_pv_node, allow_null_move, tt_best_move, in_check):
    """Handles pruning and early exits for negamax search"""
    # --- Futility Pruning for non-PV, non-check nodes ---
    if not is_pv_node and not in_check and depth <= FUTILITY_DEPTH:
        # Get static evaluation
        static_eval = get_evaluation()
        # If static evaluation + margin is still worse than alpha, likely a fail-low node
        if static_eval + FUTILITY_MARGIN * depth < alpha:
            # Return quiescence score instead of alpha to get more accurate bounds
            return quiescence_search(ply, alpha, beta), True  # Return value and prune flag

    # --- Null Move Pruning (NMP) ---
    can_nmp = allow_null_move and not in_check and depth >= NMP_MIN_DEPTH and ply > 0

    # Basic material check for NMP safety (avoid in simple endgames)
    if can_nmp:
        major_pieces = 0
        if bitboards.SIDE_TO_MOVE == chess.WHITE:
            major_pieces = (bitboards.WHITE_ROOKS | bitboards.WHITE_QUEENS).bit_count()
            minor_pieces = (bitboards.WHITE_KNIGHTS | bitboards.WHITE_BISHOPS).bit_count()
        else:
            major_pieces = (bitboards.BLACK_ROOKS | bitboards.BLACK_QUEENS).bit_count()
            minor_pieces = (bitboards.BLACK_KNIGHTS | bitboards.BLACK_BISHOPS).bit_count()

        # More conservative NMP: require at least one major piece or two minor pieces
        if major_pieces == 0 and minor_pieces < 2:
            can_nmp = False # Disable NMP in simple endings

    if can_nmp:
        # Dynamic depth reduction based on depth and evaluation
        R = NMP_REDUCTION + (depth // 4)

        # Make null move
        push_move(chess.Move.null())
        # Search with reduced depth - use a narrower window for efficiency
        null_move_score = -negamax_search(depth - 1 - R, ply + 1, -beta, -beta + 1,
                                         is_pv_node=False, allow_null_move=False)
        # Unmake null move
        pop_move()

        # Fail-high with verification for deep searches
        if null_move_score >= beta:
            # For very deep searches, verify with a reduced-depth search
            if depth >= 10:
                verification_score = negamax_search(depth - R, ply, alpha, beta,
                                                   is_pv_node=False, allow_null_move=False)
                if verification_score >= beta:
                    return beta, True  # Return cutoff value and prune flag
            else:
                return beta, True  # Regular NMP cutoff for shallow depths

    # --- Internal Iterative Deepening if no TT move is available ---
    if tt_best_move is None and is_pv_node and depth >= IID_DEPTH:
        tt_best_move = internal_iterative_deepening(depth, ply, alpha, beta, is_pv_node)

    return None, False  # No pruning opportunity

# --- Alpha-Beta Negamax Search (Main function) ---
# --- Constants for Search Extensions ---
MAX_EXTENSIONS = 3  # Maximum number of extensions allowed per search path
CHECK_EXTENSION = 1  # Extension amount when in check
PAWN_PUSH_EXTENSION = 1  # Extension for pawns approaching promotion
RECAPTURE_EXTENSION = 1  # Extension for recapture moves


# --- Negamax Search with Enhanced Extensions ---
def negamax_search(depth, ply, alpha, beta, is_pv_node=True, allow_null_move=True, num_extensions=0, prev_move=None,
                   prev_was_capture=False):
    """
    Core PVS Alpha-Beta Negamax search function with TT, NMP, LMR, and Extensions.
    Includes Quiescence Search at the leaves.
    Returns the score from the perspective of the side to move.
    """
    global nodes_searched

    # --- Time Check ---
    if should_stop_search():
        raise TimeUpException()

    nodes_searched += 1  # Increment node count *after* time check potentially raises

    # --- Base Cases & Early Exit ---
    # Max depth reached
    if ply >= MAX_PLY:
        return get_evaluation()  # Evaluate at max depth

    # --- Draw Checks ---
    # Check for Repetition (threefold including current position), 50-move, and insufficient material
    if ply > 0 and (is_repetition(2) or bitboards.HALF_MOVE_COUNTER >= 100 or is_insufficient_material_draw()):
        return DRAW_SCORE

    # --- Transposition Table Probe ---
    # Use original depth for TT lookup
    tt_score, tt_best_move, tt_hit_used = probe_tt(depth, ply, alpha, beta)  # Pass ply for mate distance adjustment
    # If probe_tt returned a score that caused a cutoff or was exact, and the score is not None, return it.
    # Allows immediate return for non-root nodes, and potentially for root nodes if TT hit was exact.
    if tt_hit_used and tt_score is not None:
        return tt_score  # Return the score directly if probe_tt deemed it usable

    # Check if in check
    king_square = chess.lsb(bitboards.WHITE_KINGS if bitboards.SIDE_TO_MOVE == chess.WHITE else bitboards.BLACK_KINGS)
    in_check = is_square_attacked(king_square, not bitboards.SIDE_TO_MOVE)

    # --- Razoring (at low depths, if we're far below alpha, go to quiescence) ---
    if not is_pv_node and not in_check and depth <= 3:
        score = get_evaluation()
        if score + RAZORING_MARGIN * depth < alpha:
            # Go directly to quiescence search with adjusted evaluation
            return quiescence_search(ply, alpha, beta)

    # If depth limit reached -> Go into Quiescence Search
    if depth <= 0:
        return quiescence_search(ply, alpha, beta)

    # --- EGTB Probe ---
    # Check only if piece count is within tablebase range
    piece_count = bitboards.ALL_PIECES.bit_count()
    if SYZYGY_TABLEBASE is not None and piece_count <= TABLEBASE_MAX_PIECES:
        try:
            # Create temporary board from bitboards (only when needed)
            temp_board = _create_board_from_bitboards()
            # Probe using the global Tablebase object
            wdl_score = SYZYGY_TABLEBASE.probe_wdl(temp_board)

            # WDL score: 2=Win, 1=Cursed Win, 0=Draw, -1=Blessed Loss, -2=Loss
            # Convert to evaluation score relative to current player
            if wdl_score == 0:
                egtb_score = DRAW_SCORE
            elif wdl_score > 0:  # Win or cursed win
                # Use MATE_SCORE adjusted by ply for faster mates
                egtb_score = MATE_SCORE - ply
            else:  # Loss or blessed loss
                egtb_score = -MATE_SCORE + ply

            record_tt(MAX_PLY, ply, egtb_score, HASH_FLAG_EXACT, None)
            return egtb_score

        except chess.syzygy.MissingTableError:
            pass
        except Exception as e:
            print(f"Warning: EGTB probe failed unexpectedly: {e}")
            pass

    # --- Pruning Strategies ---
    pruning_result, should_prune = negamax_pruning(
        depth, ply, alpha, beta, is_pv_node, allow_null_move, tt_best_move, in_check
    )
    if should_prune:
        return pruning_result

    # Generate Legal Moves
    legal_moves = generate_legal_moves()

    # Checkmate / Stalemate Detection (if not found in TT and no legal moves)
    if not legal_moves:
        if in_check:
            # Checkmate
            mate_score = -CHECKMATE_SCORE + ply  # Score indicating being mated
            record_tt(depth, ply, mate_score, HASH_FLAG_EXACT, None)  # No best move when mated
            return mate_score
        else:
            # Stalemate
            record_tt(depth, ply, DRAW_SCORE, HASH_FLAG_EXACT, None)  # No best move in stalemate
            return DRAW_SCORE

    # --- Recursive Search ---

    # Order Moves
    ordered_moves = order_moves(legal_moves, ply, tt_best_move)  # Pass TT move hint to ordering

    best_score_for_node = -CHECKMATE_SCORE  # Initialize with worst possible score
    best_move_for_node = ordered_moves[0]  # Default to first ordered move
    hash_flag = HASH_FLAG_UPPERBOUND  # Start assuming fail-low (alpha isn't raised), making alpha an upper bound
    move_count = 0
    searched_moves = 0

    for move in ordered_moves:
        move_count += 1

        # Move classification for pruning decisions
        opponent_pieces = bitboards.BLACK_PIECES if bitboards.SIDE_TO_MOVE == chess.WHITE else bitboards.WHITE_PIECES
        is_standard_capture = (opponent_pieces >> move.to_square) & 1
        moving_piece_type_local = get_piece_type_at(move.from_square)
        is_ep_capture = (moving_piece_type_local == chess.PAWN and
                         move.to_square == bitboards.EN_PASSANT_SQUARE and
                         bitboards.EN_PASSANT_SQUARE is not None)
        is_capture = is_standard_capture or is_ep_capture
        is_promotion = move.promotion is not None

        # SEE pruning for bad captures in non-PV nodes
        if not is_pv_node and is_capture and move_count > 1 and not is_promotion:
            see_score = static_exchange_evaluation(move)
            if see_score < SEE_CAPTURE_MARGIN:
                continue  # Skip clearly bad captures

        push_move(move)

        # Check if this move gives check (used for check extensions)
        opp_king_square = chess.lsb(
            bitboards.BLACK_KINGS if bitboards.SIDE_TO_MOVE == chess.WHITE else bitboards.WHITE_KINGS)
        gives_check = is_square_attacked(opp_king_square, bitboards.SIDE_TO_MOVE)

        # --- Search Extensions ---
        extension = 0
        if num_extensions < MAX_EXTENSIONS:
            # Check extension - extend search when in check
            if gives_check:
                extension = CHECK_EXTENSION
            # Pawn push extension - extend search when pawns are close to promotion
            elif moving_piece_type_local == chess.PAWN:
                target_rank = chess.square_rank(move.to_square)
                if (bitboards.SIDE_TO_MOVE == chess.WHITE and target_rank == 6) or \
                        (bitboards.SIDE_TO_MOVE == chess.BLACK and target_rank == 1):
                    extension = PAWN_PUSH_EXTENSION
            # Recapture extension - extend search for recaptures at the same square
            elif prev_move and is_capture and prev_was_capture and move.to_square == prev_move.to_square:
                extension = RECAPTURE_EXTENSION

        # Track that we actually searched this move
        searched_moves += 1

        # --- Late Move Reduction (LMR) ---
        reduction = 0
        # Apply LMR if: sufficient depth, not the first few moves, not a capture/promo, not in check, not giving check, and not a PV node.
        if depth >= LMR_BASE_DEPTH and move_count > LMR_MIN_MOVE_COUNT and not is_capture and not is_promotion and not in_check and not gives_check and not is_pv_node:
            # Calculate reduction based on depth and move count
            reduction = int(LMR_REDUCTION_FACTOR * (depth - 1) + 0.5 * math.log(move_count))
            reduction = min(depth - 2, max(0, reduction))  # Keep reduction in reasonable bounds

        # Use adjusted depth based on extensions
        search_depth = depth - 1 + extension

        # --- Principal Variation Search (PVS) ---
        # First move: Full window search
        if searched_moves == 1:
            score = -negamax_search(search_depth - reduction, ply + 1, -beta, -alpha,
                                    is_pv_node=True, allow_null_move=True,
                                    num_extensions=num_extensions + extension,
                                    prev_move=move, prev_was_capture=is_capture)
        else:
            # Try LMR with null window for efficiency
            score = -negamax_search(search_depth - reduction, ply + 1, -alpha - 1, -alpha,
                                    is_pv_node=False, allow_null_move=True,
                                    num_extensions=num_extensions + extension,
                                    prev_move=move, prev_was_capture=is_capture)

            # If reduced search failed high but didn't exceed beta, re-search with full depth
            if score > alpha and score < beta:
                # If reduction was applied and search returned a better score than alpha
                if reduction > 0:
                    # Re-search with full depth, narrow window
                    score = -negamax_search(search_depth, ply + 1, -alpha - 1, -alpha,
                                            is_pv_node=False, allow_null_move=True,
                                            num_extensions=num_extensions + extension,
                                            prev_move=move, prev_was_capture=is_capture)

                # If score still exceeds alpha after potential re-search, do a full-window PV search
                if score > alpha and score < beta:
                    score = -negamax_search(search_depth, ply + 1, -beta, -alpha,
                                            is_pv_node=True, allow_null_move=True,
                                            num_extensions=num_extensions + extension,
                                            prev_move=move, prev_was_capture=is_capture)
        pop_move()

        # Fail-High (Beta Cutoff)
        if score >= beta:
            # Store TT entry with LOWER bound flag (fail-high: score is >= beta)
            record_tt(depth, ply, beta, HASH_FLAG_LOWERBOUND, move)  # Store beta as score, Pass ply

            # --- Update Heuristics on Beta Cutoff (Phase 4.2 Enhancement) ---
            if not is_capture and move.promotion is None:
                update_killer_moves(move, ply)
                update_history_heuristic(move, depth)  # Use original depth as weight

            return beta  # Pruning occurs

        # Update Alpha (New Best Move Found)
        if score > alpha:
            alpha = score
            best_score_for_node = score  # Store the actual best score found
            best_move_for_node = move
            hash_flag = HASH_FLAG_EXACT  # We found a score within the (alpha, beta) window
            is_pv_node = True  # Found a new best move, this path segment is PV

    # --- Record to Transposition Table ---
    if best_score_for_node <= alpha:  # Initial alpha wasn't improved, so alpha is an upper bound
        # This condition corresponds to the initial hash_flag = HASH_FLAG_UPPERBOUND
        record_tt(depth, ply, alpha, HASH_FLAG_UPPERBOUND, best_move_for_node)
    elif hash_flag == HASH_FLAG_EXACT:  # Alpha was improved, exact score found
        record_tt(depth, ply, best_score_for_node, HASH_FLAG_EXACT, best_move_for_node)
    # (Beta cutoff case handled inside loop)

    return alpha  # Return the best score found (alpha)


# --- Root Search with Aspiration Windows ---
def root_search(depth, prev_score):
    """Performs a search with aspiration windows for better efficiency"""
    # Local aspiration window state for this search
    aspiration_window = ASPIRATION_WINDOW_SIZE
    # Adjust window size based on depth - deeper searches benefit from
    # slightly wider initial windows to reduce research frequency
    if depth > 6:
        aspiration_window = ASPIRATION_WINDOW_SIZE + (depth - 6) * 5

    if depth <= 3 or prev_score is None or abs(prev_score) > MATE_THRESHOLD:
        # For early depths or unknown scores, use full window
        return negamax_search(depth, 0, -CHECKMATE_SCORE, CHECKMATE_SCORE,
                              num_extensions=0, prev_move=None, prev_was_capture=False)

    # Start with a narrow window around previous score
    alpha = max(-CHECKMATE_SCORE, prev_score - aspiration_window)
    beta = min(CHECKMATE_SCORE, prev_score + aspiration_window)

    # Track window failures to avoid infinite loops in pathological positions
    window_failures = 0
    max_failures = 4

    while window_failures < max_failures:
        score = negamax_search(depth, 0, alpha, beta,
                               num_extensions=0, prev_move=None, prev_was_capture=False)

        # Check if search failed high or low
        if score <= alpha:
            # Failed low - widen alpha
            window_failures += 1
            # More aggressive widening for repeated failures
            widening_factor = 2 * (1 + window_failures // 2)
            alpha = max(-CHECKMATE_SCORE, alpha - aspiration_window * widening_factor)
            # Slight adjustment to beta as well for stability
            beta = min(CHECKMATE_SCORE, prev_score + aspiration_window)
        elif score >= beta:
            # Failed high - widen beta
            window_failures += 1
            # More aggressive widening for repeated failures
            widening_factor = 2 * (1 + window_failures // 2)
            beta = min(CHECKMATE_SCORE, beta + aspiration_window * widening_factor)
            # Slight adjustment to alpha as well for stability
            alpha = max(-CHECKMATE_SCORE, prev_score - aspiration_window)
        else:
            # Search succeeded within window
            return score

    # If we reach here, we've hit the maximum failures
    # Fall back to a full window search
    return negamax_search(depth, 0, -CHECKMATE_SCORE, CHECKMATE_SCORE,
                          num_extensions=0, prev_move=None, prev_was_capture=False)

# --- Position Evaluation Helpers ---
def is_terminal_position():
    """Check if the current position is terminal (checkmate, stalemate, or draw)"""
    # Check for draws first (faster checks)
    if is_repetition(2) or bitboards.HALF_MOVE_COUNTER >= 100 or is_insufficient_material_draw():
        return True, DRAW_SCORE

    # Check for checkmate/stalemate
    legal_moves = generate_legal_moves()
    if not legal_moves:
        # No legal moves - either checkmate or stalemate
        if is_check():
            # Checkmate (use negative mate score since this is from perspective of side to move)
            return True, -CHECKMATE_SCORE
        else:
            # Stalemate
            return True, DRAW_SCORE

    return False, 0  # Not terminal

# --- Opening Book Integration ---
def try_opening_book(board):
    """Attempt to find a move from the opening book"""
    # List of book files to try in order of preference
    # book_files = [
    #     "books/human.bin",
    #     "books/performance.bin",
    #     "books/gm2600.bin",
    #     "books/komodo.bin",
    # ]
    #
    # for book_file in book_files:
    #     try:
    #         with chess.polyglot.open_reader(book_file) as reader:
    #             # Try weighted choice first for variety
    #             try:
    #                 entry = reader.weighted_choice(board)
    #                 if entry:
    #                     return entry.move
    #             except IndexError:
    #                 # No entries or error in weighted choice
    #                 pass
    #
    #             # Fall back to finding the best move by weight
    #             try:
    #                 entries = list(reader.find_all(board))
    #                 if entries:
    #                     return max(entries, key=lambda e: e.weight).move
    #             except IndexError:
    #                 # No entries
    #                 pass
    #     except FileNotFoundError:
    #         # Book file not found, try next one
    #         continue
    #     except Exception:
    #         continue
    #
    # return None  # No book move found
    book_path = "books/Book.txt"

    book = OpeningBook(book_path)
    success, move = book.try_get_book_move(board.fen(), 0.5)
    if success:
        return move
    return chess.Move.null()

# --- Principal Variation Extraction ---
def extract_pv(depth, best_move):
    """Extract the principal variation from the transposition table"""
    if not best_move:
        return ""

    pv_string = best_move.uci()
    pv_length = 1
    max_pv_length = min(depth, 10)  # Limit PV length to avoid excessive computation

    # Create temporary board to follow the PV
    temp_board = _create_board_from_bitboards()
    current_move = best_move

    # Follow the PV by making moves and checking TT
    while pv_length < max_pv_length:
        try:
            # Make the move on our temporary board
            temp_board.push(current_move)

            # Get TT entry for the resulting position
            new_hash = chess.polyglot.zobrist_hash(temp_board)
            tt_entry = transposition_table.get(new_hash)

            if tt_entry and tt_entry.best_move:
                # Check if the move is legal in this position
                if tt_entry.best_move in temp_board.legal_moves:
                    pv_string += " " + tt_entry.best_move.uci()
                    current_move = tt_entry.best_move
                    pv_length += 1
                else:
                    break
            else:
                break
        except Exception:
            break

    return pv_string

# --- Time Management ---
def manage_search_time(depth, elapsed_time, time_limit, prev_time, score_stability):
    """
    Advanced time management for search.
    Returns True if search should continue, False if it should stop.
    """
    if elapsed_time > time_limit * 0.8:
        return False

    if depth < 4:
        return True

    # Basic time prediction (how long will next depth take)
    if prev_time > 0:
        predicted_next_time = prev_time * (4.0 + depth * 0.2)

        # If next iteration would exceed time limit, stop 
        if elapsed_time + predicted_next_time > time_limit * 0.9:
            return False

    # If score is stable and used significant time, exit
    if score_stability > 3 and elapsed_time > time_limit * 0.5:
        return False

    return True

# --- Iterative Deepening Framework ---
def iterative_deepening_search(max_depth, time_limit_seconds):
    """
    Performs IDDFS, calling negamax_search for increasing depths.
    Manages time limits and stores the best move found.
    """
    global nodes_searched, search_start_time_global, time_limit_global, stop_search

    search_start_time_global = time.time()
    time_limit_global = time_limit_seconds
    stop_search = False
    best_move_overall = None
    best_score_overall = -CHECKMATE_SCORE
    nodes_this_iteration_prev = 0
    prev_score = None
    prev_time = 0
    score_stability = 0  # Counter for stable scores between iterations

    print(f"Starting IDDFS search up to depth {max_depth} or {time_limit_seconds}s")
    clear_tt()  

    try:
        for current_depth in range(1, max_depth + 1):
            search_start_time_iter = time.time()

            # --- Call Root Search with Aspiration Windows ---
            score = root_search(current_depth, prev_score)

            # --- Track score stability for time management ---
            if prev_score is not None:
                if abs(score - prev_score) < 20:  # Score didn't change much
                    score_stability += 1
                else:
                    score_stability = 0  # Reset stability counter

            prev_score = score  # Save for next iteration's window

            # --- Process Results ---
            search_time = time.time() - search_start_time_iter
            total_time = time.time() - search_start_time_global

            # --- Get Best Move from TT ---
            root_hash = board_state.CURRENT_ZOBRIST_HASH
            root_entry = transposition_table.get(root_hash)
            best_move_this_iter = None

            if root_entry and root_entry.best_move:
                legal_moves = generate_legal_moves()
                if root_entry.best_move in legal_moves:
                    best_move_this_iter = root_entry.best_move
                else:
                    print(f"Warning: TT best move {root_entry.best_move.uci()} is not legal at root!")

            # Fallback if TT didn't provide a valid move
            if not best_move_this_iter:
                legal_moves = generate_legal_moves()
                if legal_moves:
                    ordered_moves = order_moves(legal_moves, 0, None)
                    if ordered_moves:
                        best_move_this_iter = ordered_moves[0]
                    else:
                        print("Error: No ordered moves found despite legal moves existing.")
                else:
                    print("Error: No legal moves at root in IDDFS loop.")

            # Update overall best move
            if best_move_this_iter:
                best_move_overall = best_move_this_iter
                best_score_overall = score
            elif not best_move_overall and generate_legal_moves():
                best_move_overall = order_moves(generate_legal_moves(), 0, None)[0]

            # --- Display Search Info ---
            tt_stats = get_tt_stats()
            iter_nodes = nodes_searched - nodes_this_iteration_prev
            nps = int(iter_nodes / max(search_time, 0.001))
            nodes_this_iteration_prev = nodes_searched

            tt_hit_rate = (tt_stats['hits'] / max(tt_stats['probes'], 1) * 100)

            # Prepare score string
            score_str = str(score)
            if score > MATE_THRESHOLD:
                mate_in = (MATE_SCORE - score + 1) // 2
                score_str = f"mate {mate_in}"
            elif score < -MATE_THRESHOLD:
                mate_in = (MATE_SCORE + score + 1) // 2
                score_str = f"mate -{mate_in}"

            # Extract PV
            pv_string = extract_pv(current_depth, best_move_this_iter)

            # Print UCI info
            print(f"info depth {current_depth} score {score_str} time {int(search_time*1000)} " +
                    f"nodes {iter_nodes} nps {nps} hashfull {min(1000, int(tt_stats['size']/1000))} " +
                    f"tthit {tt_hit_rate:.1f}% pv {pv_string}")

            # Check if we should continue searching
            if not manage_search_time(current_depth, total_time, time_limit_seconds,
                                        prev_time, score_stability):
                print("info string Breaking search early - time management")
                break

            prev_time = search_time  # Store for next iteration

    except TimeUpException:
        print("info string Search interrupted by time limit.")
    except Exception as e:
        print(f"info string Search error: {e}")
    finally:
        return best_move_overall, best_score_overall

# --- Public Interface ---
def find_best_move(board=None, max_depth=4, time_limit_seconds=10.0):
    """
    Public function to find the best move in the current position.
    If board is provided, initializes internal state from it first.
    Returns the best move found within the time/depth constraints.
    """
    global stop_search, nodes_searched

    # If a board object is provided, initialize our state from it
    if board is not None:
        initialize_board_state(board)

    # Start with a fresh node count
    nodes_searched = 0
    stop_search = False

    # Check for terminal positions immediately (saves time for obvious positions)
    is_terminal, terminal_score = is_terminal_position()
    if is_terminal:
        legal_moves = generate_legal_moves()
        if legal_moves:
            # If terminal but moves exist (e.g., draw by repetition but can avoid)
            # Just pick the first ordered move
            return order_moves(legal_moves, 0, None)[0]
        return board.legal_move()[0]  # No legal moves

    # Check for opening book move
    if board:
        book_move = try_opening_book(board)
        if book_move:
            print(f"info string Found opening book move: {book_move}")
            return book_move

    # Launch the iterative deepening search
    best_move, _ = iterative_deepening_search(max_depth, time_limit_seconds)
    return best_move

def stop_calculation():
    """Signal the search to stop at the next opportunity"""
    global stop_search
    stop_search = True

# Testing
if __name__ == "__main__":
    import traceback
    def main():
        global nodes_searched  # Proper global declaration at the beginning
        print("=== Chess Search Module Test Suite ===")
        # 1. Test search from standard starting position
        test_board = chess.Board()
        print("\nTest 1: Search from starting position:")
        print(test_board)
        initialize_board_state(test_board)
        start_time = time.time()
        best_move = find_best_move(max_depth=5, time_limit_seconds=3.0)
        search_time = time.time() - start_time
        print(f"Best move found: {best_move.uci() if best_move else 'None'}")
        print(f"Time taken: {search_time:.2f} seconds")
        print(f"Nodes searched: {nodes_searched:,}")
        # 2. Test a tactical position (Mate in 2)
        mate_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
        mate_board = chess.Board(mate_fen)
        print("\nTest 2: Tactical position (should find Qxf7#):")
        print(mate_board)
        initialize_board_state(mate_board)
        nodes_searched = 0  # Reset node counter
        start_time = time.time()
        mate_move = find_best_move(max_depth=4, time_limit_seconds=3.0)
        search_time = time.time() - start_time
        expected_move = chess.Move.from_uci("h5f7")
        is_correct = mate_move == expected_move if mate_move else False
        print(f"Best move found: {mate_move.uci() if mate_move else 'None'}")
        print("Expected move: Qxf7# (h5f7)")  # No need for f-string here
        print(f"Correct solution: {'Yes' if is_correct else 'No'}")
        print(f"Time taken: {search_time:.2f} seconds")
        print(f"Nodes searched: {nodes_searched:,}")
        # 3. Depth comparison test
        print("\nTest 3: Depth comparison on middlegame position:")
        middlegame_fen = "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9"
        middlegame_board = chess.Board(middlegame_fen)
        print(middlegame_board)
        initialize_board_state(middlegame_board)
        for depth in [1, 2, 3, 4, 5]:
            nodes_searched = 0  # Reset node counter
            start_time = time.time()
            move = find_best_move(max_depth=depth, time_limit_seconds=2.0)
            search_time = time.time() - start_time
            print(f"Depth {depth}: Move {move.uci() if move else 'None'} in {search_time:.2f}s ({nodes_searched:,} nodes)")
        # 4. Opening book test (if available)
        print("\nTest 4: Opening book lookup:")
        opening_board = chess.Board()  # Starting position should be in books
        book_move = try_opening_book(opening_board)
        if book_move:
            print(f"Found book move: {book_move.uci()}")
        else:
            print("No book move found - books may not be available")
        # 5. Performance benchmark
        print("\nTest 5: Performance benchmark (depth 6, 5 seconds):")
        initialize_board_state(test_board)
        nodes_searched = 0  # Reset node counter
        perf_start = time.time()
        find_best_move(max_depth=6, time_limit_seconds=5.0)
        perf_time = time.time() - perf_start
        print(f"Nodes searched: {nodes_searched:,}")
        print(f"Time taken: {perf_time:.2f} seconds")
        print(f"Nodes per second: {int(nodes_searched/perf_time):,}")
        # 6. Endgame tablebase test (if available)
        if SYZYGY_TABLEBASE is not None:
            print("\nTest 6: Endgame tablebase probe:")
            egtb_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"  # King and pawn vs king
            egtb_board = chess.Board(egtb_fen)
            print(egtb_board)
            initialize_board_state(egtb_board)
            egtb_move = find_best_move(max_depth=5, time_limit_seconds=1.0)
            print(f"Best move from tablebase: {egtb_move.uci() if egtb_move else 'None'}")
        print("\n=== All tests completed successfully ===")
        return True
    try:
        success = main()
        if not success:
            print("Tests completed with warnings.")
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nError during tests: {e}")
        traceback.print_exc()
        print("\nSearch module test failed!")
