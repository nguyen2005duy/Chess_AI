import chess
# No longer needs ui, run, state, timer, board, game_logic imports here.

# Import engine components from algo package root
try:
    # Ensure the path is correct relative to the project root
    from algo.search import find_best_move
    ENGINE_AVAILABLE = True
    print("Engine search function 'find_best_move' loaded successfully.")
except ImportError as e:
    print(f"Warning: Could not import 'find_best_move' from 'algo.search': {e}. AI opponent will not work.")
    ENGINE_AVAILABLE = False
    # Define dummy function if engine not found
    def find_best_move(board: chess.Board, max_depth: int, time_limit_seconds: float) -> chess.Move | None:
        print("Warning: Using dummy 'find_best_move' function.")
        return None
except Exception as e:
    print(f"An unexpected error occurred importing 'find_best_move': {e}")
    ENGINE_AVAILABLE = False
    def find_best_move(board: chess.Board, max_depth: int, time_limit_seconds: float) -> chess.Move | None:
        print("Warning: Using dummy 'find_best_move' function due to unexpected import error.")
        return None
