import sys
import chess
import time
import threading # For handling 'stop' command during search

# Import engine components
from .board_state import initialize_board_state, push_move
from .search import find_best_move
from .transposition_table import clear_tt
from .move_ordering import KILLER_MOVES, HISTORY_HEURISTIC # For clearing on new game

# --- UCI Engine State ---
current_board = chess.Board()
engine_options = {
    "Threads": 1,
    "Hash": 16
}
is_searching = False
search_thread = None

# --- UCI Command Handling ---

def handle_uci():
    """ Responds to the 'uci' command. """
    print("id name Chess")
    print("id author Chess")
    # TODO: Add options (like Hash, Threads)
    # print(f"option name Hash type spin default 16 min 1 max 1024")
    # print(f"option name Threads type spin default 1 min 1 max 1")
    print("uciok")

def handle_isready():
    """ Responds to the 'isready' command. """
    # If engine needs time to initialize (e.g., loading large files), do it here.
    # For now, assume ready immediately.
    print("readyok")

def handle_ucinewgame():
    """ Responds to the 'ucinewgame' command. """
    # Reset engine state for a new game
    clear_tt()
    # Clear killer moves and history heuristic
    for ply in range(len(KILLER_MOVES)):
        KILLER_MOVES[ply][0] = None
        KILLER_MOVES[ply][1] = None
    for piece_type in range(len(HISTORY_HEURISTIC)):
        for square in range(len(HISTORY_HEURISTIC[piece_type])):
            HISTORY_HEURISTIC[piece_type][square] = 0
    # print("info string Cleared TT, Killers, History") # Optional debug

def handle_position(parts):
    """ Handles the 'position' command. """
    global current_board
    if len(parts) < 2:
        return # Invalid command

    if parts[1] == "startpos":
        current_board = chess.Board()
        moves_start_index = 2
    elif parts[1] == "fen":
        # Try to parse FEN string, might span multiple parts
        try:
            fen_parts = []
            idx = 2
            # FEN has 6 parts: board, turn, castling, ep, halfmove, fullmove
            while idx < len(parts) and len(fen_parts) < 6:
                fen_parts.append(parts[idx])
                idx += 1
            fen_string = " ".join(fen_parts)
            current_board = chess.Board(fen=fen_string)
            moves_start_index = idx
        except (ValueError, IndexError):
            print("info string Error parsing FEN")
            return
    else:
        print("info string Error: Unknown position format")
        return

    # Parse moves if any
    if moves_start_index < len(parts) and parts[moves_start_index] == "moves":
        for move_uci in parts[moves_start_index + 1:]:
            try:
                move = current_board.parse_uci(move_uci)
                # Check if move is legal before pushing for robustness
                if move in current_board.legal_moves:
                    current_board.push(move)
                else:
                     print(f"info string Warning: Illegal move '{move_uci}' received in position command.")
                     # Decide how to handle: ignore? stop parsing? For now, ignore.
                     # break # Option: Stop processing moves on first illegal one
            except ValueError:
                print(f"info string Error parsing move: {move_uci}")
                # Decide how to handle: ignore? stop parsing? For now, ignore.
                # break # Option: Stop processing moves on first invalid one

    # Initialize the internal bitboard state AFTER setting up the python-chess board
    initialize_board_state(current_board)
    # print(f"info string Board state initialized for FEN: {current_board.fen()}")

def run_search(board_to_search, depth_limit, time_limit_ms):
    """ Function to run the search in a separate thread. """
    global is_searching, search_thread
    is_searching = True
    # Convert time limit from ms to seconds for find_best_move
    time_limit_sec = time_limit_ms / 1000.0 if time_limit_ms else float('inf')
    depth = depth_limit if depth_limit else 64 # Use a large number if no depth limit

    # Note: find_best_move will print its own 'info' lines if modified later
    best_move = find_best_move(board_to_search, max_depth=depth, time_limit_seconds=time_limit_sec)

    if best_move:
        print(f"bestmove {best_move.uci()}")
    else:
        # Should not happen if legal moves exist, but handle defensively
        print("bestmove 0000") # Null move indicates error or no move

    is_searching = False
    search_thread = None # Clear thread reference when done


def handle_go(parts):
    """ Handles the 'go' command. """
    global current_board, is_searching, search_thread
    if is_searching:
        print("info string Already searching")
        return

    # Parse parameters (very basic parsing for now)
    depth_limit = 0
    movetime_limit = 0
    wtime, btime, winc, binc = 0, 0, 0, 0

    i = 1
    while i < len(parts):
        token = parts[i]
        if token == "depth" and i + 1 < len(parts):
            depth_limit = int(parts[i+1])
            i += 1
        elif token == "movetime" and i + 1 < len(parts):
            movetime_limit = int(parts[i+1])
            i += 1
        elif token == "wtime" and i + 1 < len(parts):
            wtime = int(parts[i+1])
            i += 1
        elif token == "btime" and i + 1 < len(parts):
            btime = int(parts[i+1])
            i += 1
        elif token == "winc" and i + 1 < len(parts):
            winc = int(parts[i+1])
            i += 1
        elif token == "binc" and i + 1 < len(parts):
            binc = int(parts[i+1])
            i += 1
        # TODO: Add parsing for 'infinite', 'nodes', 'mate'
        i += 1

    # --- Time Management (Basic Example) ---
    time_for_move_ms = 0
    if movetime_limit > 0:
        time_for_move_ms = movetime_limit - 50 # Small buffer
    elif wtime > 0 and current_board.turn == chess.WHITE:
        # Basic: Allocate ~1/30th of remaining time + increment
        time_for_move_ms = (wtime // 30) + winc - 50
    elif btime > 0 and current_board.turn == chess.BLACK:
        time_for_move_ms = (btime // 30) + binc - 50

    time_for_move_ms = max(50, time_for_move_ms) # Ensure at least 50ms

    # Use depth limit if specified, otherwise rely on time
    if depth_limit > 0:
        time_for_move_ms = float('inf') # Effectively disable time limit if depth is set
    elif time_for_move_ms == 0:
         # Default if no time controls or depth given - e.g., 5 seconds
         time_for_move_ms = 5000

    # Make a copy of the board state for the search thread
    board_copy = current_board.copy()

    # Start search in a new thread
    search_thread = threading.Thread(target=run_search, args=(board_copy, depth_limit, time_for_move_ms))
    search_thread.start()


def handle_stop():
    """ Handles the 'stop' command. """
    global is_searching
    if is_searching:
        # Signal the search thread to stop
        # This requires the search function (find_best_move/negamax_search)
        # to periodically check a global flag or use the TimeUpException more proactively.
        # For now, we rely on the time limit check inside the search.
        # A more robust solution involves setting a global 'stop_flag' = True
        # and having the search check it frequently.
        print("info string Stop command received (search will stop based on internal time checks)")
        # We don't forcefully kill the thread, let it finish its current work up to the next time check.
        # The best move found *so far* will be printed when run_search finishes.
        pass # The search thread will eventually finish and print bestmove
    else:
        print("info string Not searching")


# --- Main Loop ---
def uci_loop():
    """ Main UCI loop reading commands from stdin. """
    global search_thread
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue

        command = parts[0]

        if command == "uci":
            handle_uci()
        elif command == "isready":
            handle_isready()
        elif command == "ucinewgame":
            handle_ucinewgame()
        elif command == "position":
            handle_position(parts)
        elif command == "go":
            handle_go(parts)
        elif command == "stop":
            handle_stop()
        elif command == "quit":
            handle_stop() # Ensure search stops if running
            if search_thread and search_thread.is_alive():
                 search_thread.join(timeout=1.0) # Wait briefly for thread
            break # Exit loop
        elif command == "print": # Debug command
            print(current_board)
            initialize_board_state(current_board) # Ensure bitboards match
            # Add prints for internal bitboard state if needed
        else:
            print(f"info string Unknown command: {command}")

        # Ensure output is flushed
        sys.stdout.flush()

if __name__ == "__main__":
    uci_loop()
