import chess
import chess.engine
import time
from algo.search import find_best_move

# --- Configuration ---
STOCKFISH_PATH = "./stockfish.exe"
STOCKFISH_ELO = 1800
ST1_ELO = 1800  # ELO for our reference Stockfish (ST1)
ENGINE_TIME_LIMIT_S = 10.0
ENGINE_DEPTH_LIMIT = 4
LIMIT_STRENGTH = True
NUM_THREADS = 8


def report_game_result(board):
    """Report the result of the game with detailed reason."""
    print("\n" + "=" * 50)
    print("Game Over!")
    print(board)
    result = board.result(claim_draw=True)
    print(f"Result: {result}")

    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"Checkmate! {winner} wins.")
    elif board.is_stalemate():
        print("Stalemate.")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    elif board.is_seventyfive_moves():
        print("Draw by 75-move rule.")
    elif board.is_fivefold_repetition():
        print("Draw by fivefold repetition.")
    elif board.can_claim_draw():
        print("Draw by claim (threefold repetition or 50-move rule).")


def evaluate_position(engine, board, time_limit):
    """Get evaluation of current position from engine."""
    info = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score = info.get("score", None)
    if score:
        # Convert score to a value white's perspective
        cp_score = score.white().score(mate_score=10000)
        return cp_score / 100.0  # Convert centipawns to pawns
    return None


def get_stockfish_move(engine, board, time_limit):
    """Get a move from Stockfish engine with analysis."""
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    best_move = result.move

    # Get position evaluation
    eval_score = evaluate_position(engine, board, time_limit / 5)  # Use shorter time for eval

    return best_move, eval_score


def is_opening_book_move(move_count):
    """
    Determine if a move is considered to be in the opening book.
    For simplicity, we consider the first 10 moves (5 per side) as opening book moves.
    """
    return move_count < 10  # First 5 full moves (10 half-moves)


def play_game(engine_color=chess.WHITE):
    """
    Plays a game between engine and Stockfish at a fixed ELO,
    while also showing what another Stockfish instance (ST1) would play.

    Args:
        engine_color: The color engine plays (chess.WHITE or chess.BLACK)
    """
    board = chess.Board()
    stockfish_engine = None
    st1_engine = None

    # Stats tracking
    total_engine_moves = 0
    matching_moves = 0
    non_book_engine_moves = 0
    non_book_matching_moves = 0

    try:
        # --- Initialize Stockfish engines ---
        print(f"Initializing Stockfish from: {STOCKFISH_PATH}")

        # Main Stockfish for playing against
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        stockfish_engine.configure({
            "UCI_LimitStrength": LIMIT_STRENGTH,
            "Threads": NUM_THREADS,
            "UCI_Elo": STOCKFISH_ELO
        })
        print(f"Main Stockfish ELO set to: {STOCKFISH_ELO}")

        # ST1 - Reference Stockfish for move comparison
        st1_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        st1_engine.configure({
            "UCI_LimitStrength": LIMIT_STRENGTH,
            "Threads": NUM_THREADS,
            "UCI_Elo": ST1_ELO
        })
        print(f"ST1 Stockfish ELO set to: {ST1_ELO}")

        # --- Game Loop ---
        move_count = 0
        while not board.is_game_over(claim_draw=True):
            print("\n" + "=" * 50)
            print(f"Move {move_count // 2 + 1}, Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
            print(board)
            print(f"FEN: {board.fen()}")

            is_book_move = is_opening_book_move(move_count)
            if is_book_move:
                print("Note: This is considered an opening book move")

            best_move = None
            start_time = time.time()

            # Always get ST1's move suggestion for comparison
            st1_move, st1_eval = get_stockfish_move(st1_engine, board, ENGINE_TIME_LIMIT_S / 2)
            st1_move_uci = st1_move.uci() if st1_move else "none"

            if board.turn == engine_color:
                print("Our Engine's Turn...")
                # Get our engine's move
                best_move = find_best_move(board, max_depth=ENGINE_DEPTH_LIMIT,
                                           time_limit_seconds=ENGINE_TIME_LIMIT_S)

                if best_move is None:
                    print("Engine returned no move!")
                    # Fallback to first legal move
                    if board.legal_moves:
                        best_move = list(board.legal_moves)[0]
                        print(f"Falling back to first legal move")
                    else:
                        print("No legal moves available for engine.")
                        break

                if best_move:
                    # Compare with ST1's move
                    print(f"Our Engine chose: {best_move.uci()}")
                    print(f"ST1 would play: {st1_move_uci}")

                    # Update statistics
                    total_engine_moves += 1
                    if not is_book_move:
                        non_book_engine_moves += 1

                    # Check if moves match
                    if best_move.uci() == st1_move_uci:
                        matching_moves += 1
                        if not is_book_move:
                            non_book_matching_moves += 1
                        print("✓ Moves match!")
                    else:
                        print("✗ Moves differ!")

                    print(f"ST1 position evaluation: {st1_eval:.2f} pawns")

                    # Print current match percentage
                    if total_engine_moves > 0:
                        match_percentage = (matching_moves / total_engine_moves) * 100
                        print(f"Overall match rate: {matching_moves}/{total_engine_moves} ({match_percentage:.1f}%)")

                    if non_book_engine_moves > 0:
                        non_book_match_percentage = (non_book_matching_moves / non_book_engine_moves) * 100
                        print(
                            f"Non-book match rate: {non_book_matching_moves}/{non_book_engine_moves} ({non_book_match_percentage:.1f}%)")
            else:
                print("Stockfish's Turn...")
                result = stockfish_engine.play(board, chess.engine.Limit(time=ENGINE_TIME_LIMIT_S))
                best_move = result.move

                if best_move:
                    print(f"Stockfish ({STOCKFISH_ELO} ELO) chose: {best_move.uci()}")
                    # Don't need to compare with ST1 for opponent's moves, but we can show what ST1 would play
                    if best_move.uci() != st1_move_uci:
                        print(f"Note: ST1 would play differently: {st1_move_uci}")

            end_time = time.time()
            print(f"Move time: {end_time - start_time:.2f}s")

            if best_move:
                board.push(best_move)
            else:
                print("Error: No move generated.")
                break

            move_count += 1

        # Report game result
        report_game_result(board)

        # Final statistics report
        print("\n" + "=" * 50)
        print("Engine Performance Statistics:")
        if total_engine_moves > 0:
            match_percentage = (matching_moves / total_engine_moves) * 100
            print(f"Overall match rate with ST1: {matching_moves}/{total_engine_moves} ({match_percentage:.1f}%)")
        else:
            print("No engine moves were made.")

        if non_book_engine_moves > 0:
            non_book_match_percentage = (non_book_matching_moves / non_book_engine_moves) * 100
            print(
                f"Non-book match rate with ST1: {non_book_matching_moves}/{non_book_engine_moves} ({non_book_match_percentage:.1f}%)")
        else:
            print("No non-book engine moves were made.")

    except chess.engine.EngineTerminatedError:
        print("Error: Stockfish engine terminated unexpectedly.")
    except FileNotFoundError:
        print(f"Error: Stockfish executable not found at '{STOCKFISH_PATH}'. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if stockfish_engine:
            stockfish_engine.quit()
            print("Main Stockfish engine closed.")
        if st1_engine:
            st1_engine.quit()
            print("ST1 Stockfish engine closed.")

        # Print final stats even if there was an error
        if total_engine_moves > 0:
            match_percentage = (matching_moves / total_engine_moves) * 100
            print(f"\nFinal overall match rate: {matching_moves}/{total_engine_moves} ({match_percentage:.1f}%)")
        if non_book_engine_moves > 0:
            non_book_match_percentage = (non_book_matching_moves / non_book_engine_moves) * 100
            print(
                f"Final non-book match rate: {non_book_matching_moves}/{non_book_engine_moves} ({non_book_match_percentage:.1f}%)")


if __name__ == "__main__":
    print("Playing as White (Our Engine vs Stockfish)")
    play_game(engine_color=chess.WHITE)

    # Uncomment to play as Black
    # print("\nPlaying as Black (Stockfish vs Our Engine)")
    # play_game(engine_color=chess.BLACK)