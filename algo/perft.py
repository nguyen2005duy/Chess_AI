import chess
import time
import cProfile # Import profiler
import pstats   # Import stats analysis
import io       # To capture profiler output
from .board_state import push_move, pop_move, initialize_board_state, BOARD_STATE_HISTORY, CURRENT_ZOBRIST_HASH # Import hash (Removed CAPTURED_PIECE_HISTORY)
from . import bitboards # Import the module itself
from .move_generation import generate_legal_moves # Import the correct function
from .zobrist import calculate_zobrist_hash as calc_hash # Import hash calculation if needed for debug

def perft(depth):
    """
    Counts the number of legal leaf nodes at a given depth using the
    generate_legal_moves function and push/pop_move for state changes.
    """
    if depth == 0:
        return 1

    nodes = 0
    # Zobrist hash checking for debugging state restoration
    initial_hash_at_depth = CURRENT_ZOBRIST_HASH

    legal_moves = generate_legal_moves() # Use the function that filters for legality

    for move in legal_moves:
        push_move(move)
        nodes += perft(depth - 1)
        pop_move()

    # --- DEBUG: Verify state restoration ---
    final_hash_at_depth = CURRENT_ZOBRIST_HASH
    if initial_hash_at_depth != final_hash_at_depth:
         print(f"\n**** HASH MISMATCH AT DEPTH {depth} ****")
         # print(f"Start Node FEN: {chess.Board().fen()}") # Requires reconstructing FEN if needed
         print(f"Initial hash: {initial_hash_at_depth:016x}")
         print(f"Final hash:   {final_hash_at_depth:016x}")
         # This indicates an issue in push_move or pop_move
         # For more detailed debugging, you might need to compare full board states
         # raise Exception("Zobrist hash mismatch during perft!") # Optionally halt on error
    # --- END DEBUG ---

    return nodes

def run_perft_tests(profile_depth=0):
    """
    Runs perft tests on standard positions using the bitboard engine.
    If profile_depth > 0, it profiles the calculation for that depth on the first position.
    """
    test_positions = {
        # Initial Position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": {
            1: 20,
            2: 400,
            3: 8902,
            4: 197281,
            5: 4865609, # Takes longer
            # 6: 119060324, # Takes much longer
        },
        # Position 2 (Kiwipete)
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1": {
            1: 48,
            2: 2039,
            3: 97862,
            4: 4085603, # Takes longer
            # 5: 193690690, # Takes much longer
        },
        # Position 3
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1": {
            1: 14,
            2: 191,
            3: 2812,
            4: 43238,
            5: 674624, # Takes longer
            # 6: 11030083, # Takes much longer
        },
         # Position 4
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1": {
             1: 6,
             2: 264,
             3: 9467,
             4: 422333,
             # 5: 15833292, # Takes longer
        },
        # Position 5
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8": {
             1: 44,
             2: 1486,
             3: 62379,
             4: 2103487,
             # 5: 89941194, # Takes longer
        },
        # Position 6
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10": {
             1: 46,
             2: 2079,
             3: 89890,
             4: 3894594,
             # 5: 164075551, # Takes longer
        }
    }

    print("Running Perft Tests (Bitboard Implementation)...")
    total_start_time = time.time()
    passed_all = True
    first_position = True

    for fen, expected_nodes in test_positions.items():
        board = chess.Board(fen)
        print(f"\nTesting position: {fen}")
        # Initialize bitboards from the FEN string using the correct function
        initialize_board_state(board)
        # Clear history before starting a new test
        BOARD_STATE_HISTORY.clear()
        # CAPTURED_PIECE_HISTORY.clear() # Removed as the list no longer exists

        position_passed = True
        for depth, expected in expected_nodes.items():
            # --- Profiling Start ---
            profiler = None
            if profile_depth > 0 and depth == profile_depth and first_position:
                print(f"\n--- PROFILING Perft({depth}) for FEN: {fen} ---")
                profiler = cProfile.Profile()
                profiler.enable()
            # --- Profiling End ---

            start_time = time.time()
            nodes = perft(depth)
            end_time = time.time()
            duration = end_time - start_time
            nps = nodes / duration if duration > 0 else 0

            # --- Profiling Start ---
            if profiler:
                profiler.disable()
                s = io.StringIO()
                sortby = pstats.SortKey.CUMULATIVE # Can also use 'tottime'
                ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
                ps.print_stats(30) # Print top 30 functions
                print(s.getvalue())
                print(f"--- END PROFILING Perft({depth}) ---\n")
                # Reset profile_depth to avoid profiling other positions/depths
                profile_depth = 0
            # --- Profiling End ---

            result_str = f"Depth {depth}: Nodes = {nodes:>10}, Expected = {expected:>10}, Time = {duration:>6.2f}s, NPS = {nps:,.0f}"
            if nodes != expected:
                print(f"  ERROR: {result_str}")
                position_passed = False
                passed_all = False
            else:
                print(f"  PASS:  {result_str}")

        if position_passed:
             print("  Position PASSED.")
        else:
             print("  Position FAILED.")
        first_position = False # Only profile the first position

    total_end_time = time.time()
    print(f"\nTotal Perft Test Time: {total_end_time - total_start_time:.2f}s")
    if passed_all:
        print("All Perft tests PASSED.")
    else:
        print("Some Perft tests FAILED.")

# Example Usage
if __name__ == "__main__":
    # Set profile_depth to the depth you want to profile (e.g., 4)
    # Set to 0 to disable profiling and run all tests normally
    run_perft_tests(profile_depth=5)
