import chess
import chess.engine
import time
from algo.search import find_best_move

# --- Configuration ---
STOCKFISH_PATH = "./stockfish-linux"
STOCKFISH_ELO = 2500
ENGINE_TIME_LIMIT_S = 8.0
ENGINE_DEPTH_LIMIT = 5
LIMIT_STRENGTH = True
NUM_THREADS = 8

def report_game_result(board):
    """Report the result of the game with detailed reason."""
    print("\n" + "="*20)
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

def play_game(engine_color=chess.WHITE):
    """
    Plays a game between engine and Stockfish at a fixed ELO.

    Args:
        engine_color: The color engine plays (chess.WHITE or chess.BLACK)
    """
    board = chess.Board()
    stockfish_engine = None

    try:
        # --- Initialize Stockfish ---
        print(f"Initializing Stockfish from: {STOCKFISH_PATH}")
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

        # Configure Stockfish strength
        stockfish_engine.configure({
            "UCI_LimitStrength": LIMIT_STRENGTH,
            "Threads": NUM_THREADS,
            "UCI_Elo": STOCKFISH_ELO
        })
        print(f"Stockfish ELO set to: {STOCKFISH_ELO}")

        # --- Game Loop ---
        move_count = 0
        while not board.is_game_over(claim_draw=True):
            print("\n" + "="*20)
            print(f"Move {move_count // 2 + 1}, Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
            print(board)
            print(f"FEN: {board.fen()}")

            best_move = None
            start_time = time.time()

            if board.turn == engine_color:
                print("Engine's Turn...")
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
                    print(f"Engine chose: {best_move.uci()}")
            else:
                print("Stockfish's Turn...")
                result = stockfish_engine.play(board, chess.engine.Limit(time=ENGINE_TIME_LIMIT_S))
                best_move = result.move
                if best_move: 
                    print(f"Stockfish ({STOCKFISH_ELO} ELO) chose: {best_move.uci()}")

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
            print("Stockfish engine closed.")

if __name__ == "__main__":
    play_game(engine_color=chess.WHITE)
    # play_game(engine_color=chess.BLACK)
