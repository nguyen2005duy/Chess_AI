import chess
import chess.engine
import time
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Import your search function for testing
try:
    from algo.search import find_best_move
except ImportError:
    print("Warning: Could not import find_best_move from algo.search")


    # Define a placeholder function for testing
    def find_best_move(board, max_depth=4, time_limit_seconds=5.0):
        return list(board.legal_moves)[0] if board.legal_moves else None

# --- Configuration ---
STOCKFISH_PATH = "./stockfish.exe"  # Update this for your system
DEBUG_POSITIONS = 300  # Number of positions to analyze
STOCKFISH_DEPTH = 6  # Depth for "ground truth" analysis
ENGINE_DEPTH_LIMIT = 6
ENGINE_TIME_LIMIT_S = 5
OUTPUT_DIR = "engine_comparison"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


def configure_stockfish_engine(engine, elo=None):
    """Configure Stockfish engine with consistent settings."""
    config = {
        "Threads": 1,  # Single thread for consistent results
        "Hash": 128,  # Hash table size in MB
    }

    # Add ELO limiting if specified
    if elo is not None:
        config.update({
            "UCI_LimitStrength": True,
            "UCI_Elo": elo
        })

    try:
        engine.configure(config)
        print(f"Stockfish configured" + (f" for {elo} ELO" if elo else ""))
    except chess.engine.EngineError as e:
        print(f"Warning: Some engine options not supported. {e}")

    return engine


def generate_test_positions(num_positions=50, ply_range=(5, 30)):
    """Generate a set of test positions by playing random games."""
    positions = []
    print(f"Generating {num_positions} test positions...")

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        configure_stockfish_engine(engine)

        while len(positions) < num_positions:
            board = chess.Board()
            target_ply = np.random.randint(ply_range[0], ply_range[1])

            try:
                # Play a partial game to reach desired position
                for _ in range(target_ply):
                    if board.is_game_over():
                        break

                    # Use Stockfish for some moves, random for others to get variety
                    if np.random.random() < 0.7:  # 70% Stockfish moves
                        result = engine.play(board, chess.engine.Limit(time=0.1))
                        move = result.move
                    else:  # 30% random but legal moves
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            move = np.random.choice(legal_moves)
                        else:
                            break

                    board.push(move)

                # Only add non-terminal positions
                if not board.is_game_over() and board.fen() not in [pos["fen"] for pos in positions]:
                    positions.append({"fen": board.fen(), "ply": board.fullmove_number})

            except Exception as e:
                # Just continue to next position on error
                continue

    print(f"Generated {len(positions)} unique positions")
    return positions


def analyze_positions(positions_batch, stockfish_path, stockfish_depth, engine_depth_limit, engine_time_limit):
    """Analyze a batch of positions in a separate process."""
    import chess
    import chess.engine
    import time
    from collections import defaultdict
    import sys
    import os

    # Import the search function for this process
    try:
        from algo.search import find_best_move
    except ImportError:
        print("Warning: Could not import find_best_move from algo.search in worker process")

        # Define a placeholder function for testing
        def find_best_move(board, max_depth=4, time_limit_seconds=5.0):
            return list(board.legal_moves)[0] if board.legal_moves else None

    # Helper functions needed within this process
    def configure_stockfish_engine(engine, elo=None):
        """Configure Stockfish engine with consistent settings."""
        config = {
            "Threads": 1,  # Single thread for consistent results
            "Hash": 128,  # Hash table size in MB
        }

        # Add ELO limiting if specified
        if elo is not None:
            config.update({
                "UCI_LimitStrength": True,
                "UCI_Elo": elo
            })

        try:
            engine.configure(config)
        except chess.engine.EngineError as e:
            print(f"Warning: Some engine options not supported. {e}")

        return engine

    def analyze_position(board, engine, depth=6):
        """Analyze a position with the given engine."""
        try:
            limit = chess.engine.Limit(depth=depth)

            start_time = time.time()
            info = engine.analyse(board, limit)
            end_time = time.time()

            # Extract results
            result = {
                "fen": board.fen(),
                "time_taken": end_time - start_time,
            }

            # Extract score
            if "score" in info:
                cp_score = info["score"].white().score(mate_score=10000)
                result["score"] = cp_score / 100.0  # Convert to pawns
            else:
                result["score"] = None

            # Extract best move
            if "pv" in info and len(info["pv"]) > 0:
                result["best_move"] = info["pv"][0].uci()
            else:
                result["best_move"] = None

            return result
        except Exception as e:
            # Return empty result on error
            return {
                "fen": board.fen() if board else None,
                "time_taken": 0,
                "score": None,
                "best_move": None
            }

    def analyze_with_your_engine(board, max_depth, time_limit):
        """Analyze a position with your custom engine."""
        start_time = time.time()
        result = {
            "fen": board.fen(),
            "time_taken": 0,
            "best_move": None,
        }

        try:
            # Call your engine's find_best_move function
            best_move = find_best_move(board, max_depth=max_depth, time_limit_seconds=time_limit)

            # Ensure best_move is a proper Move object
            if best_move and not isinstance(best_move, chess.Move):
                # Try to convert string notation to a move if that's what was returned
                if isinstance(best_move, str):
                    try:
                        best_move = chess.Move.from_uci(best_move)
                    except ValueError:
                        best_move = None
                else:
                    best_move = None

            end_time = time.time()
            result["time_taken"] = end_time - start_time
            result["best_move"] = best_move.uci() if best_move else None

        except Exception:
            end_time = time.time()
            result["time_taken"] = end_time - start_time

        return result

    def compare_engine_moves(your_move, stockfish_move, legal_moves, board):
        """Compare the engine moves and classify the difference."""
        if your_move == stockfish_move:
            return "exact_match"

        # Handle None cases first
        if your_move is None:
            return "no_move"

        try:
            your_move_obj = chess.Move.from_uci(your_move)

            # Check if move is legal
            if your_move_obj not in legal_moves:
                return "illegal_move"

            # Only proceed if stockfish move exists
            if stockfish_move:
                stockfish_move_obj = chess.Move.from_uci(stockfish_move)

                # Check if same piece is moving
                from_square_same = your_move_obj.from_square == stockfish_move_obj.from_square

                # Check if it's a capture vs non-capture
                your_is_capture = board.is_capture(your_move_obj)
                stockfish_is_capture = board.is_capture(stockfish_move_obj)

                if from_square_same and your_is_capture != stockfish_is_capture:
                    return "capture_decision_difference"
                elif from_square_same:
                    return "same_piece_different_target"
                else:
                    # Get the pieces being moved
                    your_piece = board.piece_at(your_move_obj.from_square)
                    stockfish_piece = board.piece_at(stockfish_move_obj.from_square)

                    if your_piece and stockfish_piece and your_piece.piece_type == stockfish_piece.piece_type:
                        return "same_piece_type_different_location"
                    else:
                        return "different_piece_move"
        except Exception:
            return "error_comparing"

        return "different_move"

    # Process the batch
    batch_results = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as stockfish:
        configure_stockfish_engine(stockfish)

        for position in positions_batch:
            try:
                fen = position["fen"]
                board = chess.Board(fen)
                legal_moves = list(board.legal_moves)

                # Skip positions with very few options
                if len(legal_moves) < 3:
                    continue

                # Analyze with Stockfish
                stockfish_analysis = analyze_position(board, stockfish, depth=stockfish_depth)

                # Analyze with your engine
                your_analysis = analyze_with_your_engine(board, max_depth=engine_depth_limit,
                                                         time_limit=engine_time_limit)

                # Compare results
                comparison = {
                    "position_id": position.get("position_id", 0),
                    "fen": fen,
                    "stockfish_move": stockfish_analysis.get("best_move"),
                    "your_move": your_analysis.get("best_move"),
                    "legal_move_count": len(legal_moves),
                    "fullmove_number": board.fullmove_number
                }

                # Classify move comparison
                move_comparison_type = compare_engine_moves(
                    your_analysis.get("best_move"),
                    stockfish_analysis.get("best_move"),
                    legal_moves,
                    board
                )
                comparison["move_comparison"] = move_comparison_type

                # Calculate time ratio (your engine / stockfish)
                if stockfish_analysis.get("time_taken", 0) > 0:
                    time_ratio = your_analysis.get("time_taken", 0) / stockfish_analysis.get("time_taken", 0)
                    comparison["time_ratio"] = time_ratio

                batch_results.append(comparison)

            except Exception as e:
                # Skip problematic positions
                continue

    return batch_results


def analyze_with_your_engine(board, max_depth=ENGINE_DEPTH_LIMIT, time_limit=ENGINE_TIME_LIMIT_S):
    """Analyze a position with your custom engine."""
    start_time = time.time()
    result = {
        "fen": board.fen(),
        "time_taken": 0,
        "best_move": None,
    }

    try:
        # Call your engine's find_best_move function
        best_move = find_best_move(board, max_depth=max_depth, time_limit_seconds=time_limit)

        # Ensure best_move is a proper Move object
        if best_move and not isinstance(best_move, chess.Move):
            # Try to convert string notation to a move if that's what was returned
            if isinstance(best_move, str):
                try:
                    best_move = chess.Move.from_uci(best_move)
                except ValueError:
                    best_move = None
            else:
                best_move = None

        end_time = time.time()
        result["time_taken"] = end_time - start_time
        result["best_move"] = best_move.uci() if best_move else None

    except Exception:
        end_time = time.time()
        result["time_taken"] = end_time - start_time

    return result


def compare_engine_moves(your_move, stockfish_move, legal_moves, board):
    """Compare the engine moves and classify the difference."""
    if your_move == stockfish_move:
        return "exact_match"

    # Handle None cases first
    if your_move is None:
        return "no_move"

    try:
        your_move_obj = chess.Move.from_uci(your_move)

        # Check if move is legal
        if your_move_obj not in legal_moves:
            return "illegal_move"

        # Only proceed if stockfish move exists
        if stockfish_move:
            stockfish_move_obj = chess.Move.from_uci(stockfish_move)

            # Check if same piece is moving
            from_square_same = your_move_obj.from_square == stockfish_move_obj.from_square

            # Check if it's a capture vs non-capture
            your_is_capture = board.is_capture(your_move_obj)
            stockfish_is_capture = board.is_capture(stockfish_move_obj)

            if from_square_same and your_is_capture != stockfish_is_capture:
                return "capture_decision_difference"
            elif from_square_same:
                return "same_piece_different_target"
            else:
                # Get the pieces being moved
                your_piece = board.piece_at(your_move_obj.from_square)
                stockfish_piece = board.piece_at(stockfish_move_obj.from_square)

                if your_piece and stockfish_piece and your_piece.piece_type == stockfish_piece.piece_type:
                    return "same_piece_type_different_location"
                else:
                    return "different_piece_move"
    except Exception:
        return "error_comparing"

    return "different_move"


def analyze_move_differences_batch(positions_batch, stockfish_path):
    """Analyze why your engine's moves differ from Stockfish's for a batch of positions."""
    import chess
    import chess.engine
    import time

    # Process the batch with one stockfish instance
    detailed_analyses = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as stockfish:
        # Configure stockfish
        config = {
            "Threads": 1,
            "Hash": 128,
        }
        try:
            stockfish.configure(config)
        except chess.engine.EngineError:
            pass

        for pos in positions_batch:
            try:
                fen = pos.get("fen")
                your_move = pos.get("your_move")
                stockfish_move = pos.get("stockfish_move")

                if not fen or not your_move or not stockfish_move:
                    continue

                board = chess.Board(fen)
                analysis = analyze_move_difference(board, your_move, stockfish_move, stockfish)

                interesting_position = {
                    "fen": fen,
                    "stockfish_move": stockfish_move,
                    "your_move": your_move,
                    "analysis": analysis,
                    "fullmove_number": board.fullmove_number
                }
                detailed_analyses.append(interesting_position)

            except Exception:
                continue

    return detailed_analyses


def analyze_move_difference(board, your_move, stockfish_move, stockfish):
    """Analyze why your engine's move differs from Stockfish's."""
    if your_move == stockfish_move:
        return "Moves match"

    if not your_move:
        return "Your engine didn't return a move"

    if not stockfish_move:
        return "Stockfish didn't return a move"

    analysis = {}

    try:
        # Analyze a position with the given engine
        def analyze_position(board, engine, depth=6):
            try:
                limit = chess.engine.Limit(depth=depth)
                start_time = time.time()
                info = engine.analyse(board, limit)
                end_time = time.time()

                # Extract results
                result = {
                    "fen": board.fen(),
                    "time_taken": end_time - start_time,
                }

                # Extract score
                if "score" in info:
                    cp_score = info["score"].white().score(mate_score=10000)
                    result["score"] = cp_score / 100.0  # Convert to pawns
                else:
                    result["score"] = None

                # Extract best move
                if "pv" in info and len(info["pv"]) > 0:
                    result["best_move"] = info["pv"][0].uci()
                else:
                    result["best_move"] = None

                return result
            except Exception:
                # Return empty result on error
                return {
                    "fen": board.fen() if board else None,
                    "time_taken": 0,
                    "score": None,
                    "best_move": None
                }

        # Evaluate position after your move
        board_after_your_move = board.copy()
        your_move_obj = chess.Move.from_uci(your_move)
        board_after_your_move.push(your_move_obj)
        your_result = analyze_position(board_after_your_move, stockfish, depth=8)

        # Evaluate position after Stockfish move
        board_after_stockfish = board.copy()
        stockfish_move_obj = chess.Move.from_uci(stockfish_move)
        board_after_stockfish.push(stockfish_move_obj)
        stockfish_result = analyze_position(board_after_stockfish, stockfish, depth=8)

        # Calculate evaluation difference
        if your_result["score"] is not None and stockfish_result["score"] is not None:
            eval_diff = your_result["score"] - stockfish_result["score"]
            if board.turn == chess.BLACK:
                eval_diff = -eval_diff
            analysis["eval_diff"] = eval_diff
        else:
            analysis["eval_diff"] = 0

        # Check material changes
        def count_material(b):
            material = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0}
            for square in chess.SQUARES:
                piece = b.piece_at(square)
                if piece:
                    material[piece.symbol()] += 1
            return material

        material_before = count_material(board)
        material_after_your = count_material(board_after_your_move)
        material_after_stockfish = count_material(board_after_stockfish)

        # Calculate material value changes
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
                        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0}

        your_material_change = sum((material_after_your[p] - material_before[p]) * piece_values[p.upper()]
                                   for p in material_before)
        stockfish_material_change = sum((material_after_stockfish[p] - material_before[p]) * piece_values[p.upper()]
                                        for p in material_before)

        analysis["your_material_change"] = your_material_change
        analysis["stockfish_material_change"] = stockfish_material_change

        # Check if moves are captures
        your_captures = any(material_before[p] > material_after_your[p] for p in material_before)
        stockfish_captures = any(material_before[p] > material_after_stockfish[p] for p in material_before)

        analysis["your_captures"] = your_captures
        analysis["stockfish_captures"] = stockfish_captures

        # Check if captured piece values differ
        if your_captures and stockfish_captures:
            # Calculate value of piece captured by your move
            your_captured_value = 0
            for p in material_before:
                if material_before[p] > material_after_your[p]:
                    your_captured_value += (material_before[p] - material_after_your[p]) * abs(piece_values[p])

            # Calculate value of piece captured by stockfish move
            stockfish_captured_value = 0
            for p in material_before:
                if material_before[p] > material_after_stockfish[p]:
                    stockfish_captured_value += (material_before[p] - material_after_stockfish[p]) * abs(
                        piece_values[p])

            analysis["your_captured_value"] = your_captured_value
            analysis["stockfish_captured_value"] = stockfish_captured_value

        # Check checks and threats
        analysis["your_gives_check"] = board_after_your_move.is_check()
        analysis["stockfish_gives_check"] = board_after_stockfish.is_check()

        # Examine piece development (for opening)
        if board.fullmove_number <= 10:
            # Check if the move develops a new piece
            piece_at_your_from = board.piece_at(your_move_obj.from_square)
            piece_at_stockfish_from = board.piece_at(stockfish_move_obj.from_square)

            your_develops_piece = (piece_at_your_from and
                                   piece_at_your_from.piece_type in [chess.KNIGHT, chess.BISHOP] and
                                   your_move_obj.from_square in [chess.B1, chess.G1, chess.C1, chess.F1,
                                                                 chess.B8, chess.G8, chess.C8, chess.F8])

            stockfish_develops_piece = (piece_at_stockfish_from and
                                        piece_at_stockfish_from.piece_type in [chess.KNIGHT, chess.BISHOP] and
                                        stockfish_move_obj.from_square in [chess.B1, chess.G1, chess.C1, chess.F1,
                                                                           chess.B8, chess.G8, chess.C8, chess.F8])

            analysis["your_develops_piece"] = your_develops_piece
            analysis["stockfish_develops_piece"] = stockfish_develops_piece

        # Check center control (for opening/middlegame)
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]

        # Count how many center squares are attacked before and after moves
        def count_center_attacks(b, color):
            return sum(1 for sq in center_squares if b.is_attacked_by(color, sq))

        your_center_control = count_center_attacks(board_after_your_move, board.turn)
        stockfish_center_control = count_center_attacks(board_after_stockfish, board.turn)

        analysis["your_center_control"] = your_center_control
        analysis["stockfish_center_control"] = stockfish_center_control

        # Generate insights
        insights = []

        if abs(analysis.get("eval_diff", 0)) < 0.3:
            insights.append("Moves are similar in evaluation")
        elif analysis.get("eval_diff", 0) < 0:
            insights.append(f"Stockfish's move is better by {abs(analysis.get('eval_diff', 0)):.2f} pawns")
        else:
            insights.append(f"Your move is surprisingly better by {analysis.get('eval_diff', 0):.2f} pawns")

        if your_captures and not stockfish_captures:
            insights.append("Your engine captured material but Stockfish chose a positional move")
        elif stockfish_captures and not your_captures:
            insights.append("Stockfish captured material but your engine chose a different approach")

        if your_captures and stockfish_captures and "your_captured_value" in analysis and "stockfish_captured_value" in analysis:
            if analysis["your_captured_value"] < analysis["stockfish_captured_value"]:
                insights.append(
                    f"Stockfish captured higher value material (worth {analysis['stockfish_captured_value']} vs your {analysis['your_captured_value']})")
            elif analysis["your_captured_value"] > analysis["stockfish_captured_value"]:
                insights.append(
                    f"Your engine captured higher value material (worth {analysis['your_captured_value']} vs Stockfish's {analysis['stockfish_captured_value']})")

        if analysis.get("stockfish_gives_check", False) and not analysis.get("your_gives_check", False):
            insights.append("Stockfish found a check that your engine missed")
        elif analysis.get("your_gives_check", False) and not analysis.get("stockfish_gives_check", False):
            insights.append("Your engine found a check that Stockfish deemed suboptimal")

        if board.fullmove_number <= 10:
            if analysis.get("stockfish_develops_piece", False) and not analysis.get("your_develops_piece", False):
                insights.append("Stockfish developed a piece in the opening, while your move did not")
            elif analysis.get("your_develops_piece", False) and not analysis.get("stockfish_develops_piece", False):
                insights.append(
                    "Your engine developed a piece in the opening, while Stockfish chose a different priority")

        if analysis.get("stockfish_center_control", 0) > analysis.get("your_center_control", 0):
            insights.append(
                f"Stockfish's move improves center control more than yours ({analysis.get('stockfish_center_control', 0)} vs {analysis.get('your_center_control', 0)} center squares attacked)")
        elif analysis.get("your_center_control", 0) > analysis.get("stockfish_center_control", 0):
            insights.append(
                f"Your move improves center control more than Stockfish's ({analysis.get('your_center_control', 0)} vs {analysis.get('stockfish_center_control', 0)} center squares attacked)")

        analysis["insights"] = insights
        return analysis

    except Exception:
        # Return basic analysis if detailed analysis fails
        return {
            "eval_diff": 0,
            "insights": ["Analysis unavailable due to error"]
        }


def collect_move_patterns(results):
    """Analyze patterns in move selection differences."""
    patterns = {
        "opening_patterns": defaultdict(int),
        "middlegame_patterns": defaultdict(int),
        "endgame_patterns": defaultdict(int),
        "material_imbalance": defaultdict(int),
        "piece_preference": {
            "your_engine": defaultdict(int),
            "stockfish": defaultdict(int)
        }
    }

    for pos in results.get("positions", []):
        try:
            fen = pos.get("fen")
            if not fen:
                continue

            board = chess.Board(fen)
            your_move = pos.get("your_move")
            stockfish_move = pos.get("stockfish_move")

            if not your_move or not stockfish_move:
                continue

            # Determine game phase
            pieces = len(board.piece_map())
            phase = "opening_patterns" if board.fullmove_number <= 10 else \
                "endgame_patterns" if pieces <= 12 else "middlegame_patterns"

            # Record the move difference type
            move_comparison = pos.get("move_comparison", "unknown")
            patterns[phase][move_comparison] += 1

            # Record piece types being moved
            try:
                your_move_obj = chess.Move.from_uci(your_move)
                stockfish_move_obj = chess.Move.from_uci(stockfish_move)

                your_piece = board.piece_at(your_move_obj.from_square)
                stockfish_piece = board.piece_at(stockfish_move_obj.from_square)

                if your_piece:
                    patterns["piece_preference"]["your_engine"][chess.piece_name(your_piece.piece_type)] += 1
                if stockfish_piece:
                    patterns["piece_preference"]["stockfish"][chess.piece_name(stockfish_piece.piece_type)] += 1
            except:
                pass

            # Check material imbalance
            material_diff = calculate_material_difference(board)
            imbalance = "equal" if abs(material_diff) <= 1 else \
                "white_advantage" if material_diff > 1 else "black_advantage"
            patterns["material_imbalance"][move_comparison + "_" + imbalance] += 1

        except Exception:
            continue

    return patterns


def calculate_material_difference(board):
    """Calculate material difference in a position (positive = white advantage)."""
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

    white_material = sum(piece_values[p.piece_type] for p in board.piece_map().values()
                         if p.color == chess.WHITE)
    black_material = sum(piece_values[p.piece_type] for p in board.piece_map().values()
                         if p.color == chess.BLACK)

    return white_material - black_material


def run_engine_comparison():
    """Compare your chess engine with Stockfish using multiprocessing."""
    import multiprocessing as mp
    import time
    import os
    import json
    import numpy as np
    from collections import defaultdict
    from tqdm import tqdm

    try:
        from generate_test_positions import generate_test_positions
        from collect_move_patterns import collect_move_patterns
        from generate_comparison_graph import generate_comparison_graph
        from print_comparison_insights import print_comparison_insights
    except ImportError:
        # Import these functions from the main module if they're not separate
        from __main__ import (generate_test_positions, collect_move_patterns,
                              generate_comparison_graph, print_comparison_insights)

    results = {
        "positions": [],
        "summary": {},
        "detailed_analysis": [],
        "move_patterns": {}
    }

    try:
        print("Starting chess engine comparison with multiprocessing...")

        # Configuration
        STOCKFISH_PATH = "./stockfish.exe"  # Update this for your system
        DEBUG_POSITIONS = 100  # Number of positions to analyze
        STOCKFISH_DEPTH = 6  # Depth for "ground truth" analysis
        ENGINE_DEPTH_LIMIT = 6
        ENGINE_TIME_LIMIT_S = 5
        OUTPUT_DIR = "engine_comparison"

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

        # Generate test positions
        test_positions = generate_test_positions(DEBUG_POSITIONS)
        print(f"Generated {len(test_positions)} test positions")

        # Determine the number of processes to use
        num_cores = mp.cpu_count()
        num_processes = max(1, min(num_cores - 1, 4))  # Use at most 4 processes, leave 1 core free
        print(f"Using {num_processes} parallel processes for analysis")

        # Split positions into batches for parallel processing
        batch_size = len(test_positions) // num_processes
        if batch_size < 1:
            batch_size = 1

        position_batches = []
        for i in range(0, len(test_positions), batch_size):
            # Add position_id to each position for tracking
            batch = []
            for idx, position in enumerate(test_positions[i:i + batch_size]):
                position_with_id = position.copy()
                position_with_id["position_id"] = i + idx
                batch.append(position_with_id)
            position_batches.append(batch)

        # Analyze positions in parallel
        move_comparisons = defaultdict(int)
        time_ratios = []
        different_move_positions = []

        start_time = time.time()
        print(f"Analyzing {len(test_positions)} positions using {num_processes} processes...")

        # Create a pool of worker processes
        with mp.Pool(processes=num_processes) as pool:
            # Create argument tuples for each batch
            args = [(batch, STOCKFISH_PATH, STOCKFISH_DEPTH, ENGINE_DEPTH_LIMIT, ENGINE_TIME_LIMIT_S)
                    for batch in position_batches]

            # Run the analysis in parallel
            batch_results = list(tqdm(pool.starmap(analyze_positions, args),
                                      total=len(position_batches)))

        # Flatten results and process them
        for batch in batch_results:
            for pos in batch:
                results["positions"].append(pos)
                move_comparisons[pos.get("move_comparison", "unknown")] += 1

                if pos.get("time_ratio") is not None:
                    time_ratios.append(pos.get("time_ratio"))

                # Collect positions with different moves for detailed analysis
                if pos.get("move_comparison") not in ["exact_match", "no_move", "illegal_move"]:
                    different_move_positions.append(pos)

        print(f"Position analysis completed in {time.time() - start_time:.1f} seconds")

        # Analyze move differences in parallel for interesting positions
        if different_move_positions:
            print(f"Analyzing {len(different_move_positions)} positions with move differences...")

            # Split the different move positions into batches
            diff_batch_size = len(different_move_positions) // num_processes
            if diff_batch_size < 1:
                diff_batch_size = 1

            diff_batches = []
            for i in range(0, len(different_move_positions), diff_batch_size):
                diff_batches.append(different_move_positions[i:i + diff_batch_size])

            # Analyze the differences in parallel
            with mp.Pool(processes=num_processes) as pool:
                args = [(batch, STOCKFISH_PATH) for batch in diff_batches]
                detailed_batches = list(tqdm(pool.starmap(analyze_move_differences_batch, args),
                                             total=len(diff_batches)))

            # Flatten the detailed analysis results
            detailed_analyses = []
            for batch in detailed_batches:
                detailed_analyses.extend(batch)

            # Sort by evaluation difference
            if detailed_analyses:
                detailed_analyses.sort(
                    key=lambda x: abs(x["analysis"].get("eval_diff", 0)),
                    reverse=True
                )
                results["detailed_analysis"] = detailed_analyses[:min(5, len(detailed_analyses))]

        # Calculate summary statistics
        total_positions = len(results["positions"])
        if total_positions > 0:
            results["summary"] = {
                "total_positions": total_positions,
                "move_comparison": dict(move_comparisons),
                "exact_match_percentage": move_comparisons[
                                              "exact_match"] / total_positions * 100 if total_positions > 0 else 0,
                "average_time_ratio": np.mean(time_ratios) if time_ratios else None,
            }

            # Collect move patterns
            results["move_patterns"] = collect_move_patterns(results)

        # Save results to file
        with open(os.path.join(OUTPUT_DIR, "comparison_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"Comparison complete. Results saved to {os.path.join(OUTPUT_DIR, 'comparison_results.json')}")

        # Generate circle graph
        generate_comparison_graph(results)

        # Print key insights
        print_comparison_insights(results)

        return results

    except Exception as e:
        print(f"Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        return results


def generate_comparison_graph(results):
    """Generate a circle graph showing move comparison distribution."""
    try:
        # Move comparison pie chart
        if "move_comparison" in results["summary"]:
            move_comp = results["summary"]["move_comparison"]

            # Clean and simplify labels
            labels = []
            sizes = []

            label_mapping = {
                "exact_match": "Exact Match",
                "different_move": "Different Move",
                "different_piece_move": "Different Piece",
                "same_piece_different_target": "Same Piece, Different Target",
                "capture_decision_difference": "Capture Decision Diff",
                "no_move": "No Move Returned",
                "illegal_move": "Illegal Move",
                "same_piece_type_different_location": "Same Piece Type, Diff Location",
                "error_comparing": "Error Comparing"
            }

            for key, value in move_comp.items():
                if value > 0:  # Only include non-zero values
                    labels.append(label_mapping.get(key, key))
                    sizes.append(value)

            # Create pie chart
            plt.figure(figsize=(10, 8))
            colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))

            # Plot with a slight explosion for the largest segment
            explode = [0.1 if i == np.argmax(sizes) else 0 for i in range(len(sizes))]

            wedges, texts, autotexts = plt.pie(
                sizes,
                labels=None,  # We'll add labels manually for better formatting
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                colors=colors,
                shadow=True
            )

            # Enhance the chart appearance
            plt.title('Move Comparison Distribution', fontsize=16, pad=20)

            # Add a legend with percentages
            legend_labels = [f"{l} ({s}/{sum(sizes)}, {s / sum(sizes) * 100:.1f}%)" for l, s in zip(labels, sizes)]
            plt.legend(wedges, legend_labels, title="Move Types",
                       loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))

            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "plots", "move_comparison_pie.png"), dpi=100, bbox_inches='tight')
            plt.close()

            print(f"Chart saved to {os.path.join(OUTPUT_DIR, 'plots', 'move_comparison_pie.png')}")

    except Exception as e:
        print(f"Error generating chart: {e}")


def print_comparison_insights(results):
    """Print detailed insights from the engine comparison."""
    print("\n=== ENGINE COMPARISON INSIGHTS ===")

    summary = results.get("summary", {})
    total = summary.get("total_positions", 0)

    if total == 0:
        print("No positions were analyzed.")
        return

    # Print overall match rate
    match_rate = summary.get("exact_match_percentage", 0)
    print(
        f"Overall match rate: {match_rate:.1f}% ({summary.get('move_comparison', {}).get('exact_match', 0)}/{total} positions)")

    # Speed comparison
    time_ratio = summary.get("average_time_ratio")
    if time_ratio:
        if time_ratio < 0.8:
            print(f"Your engine is {1 / time_ratio:.1f}x faster than Stockfish on average")
        elif time_ratio > 1.2:
            print(f"Your engine is {time_ratio:.1f}x slower than Stockfish on average")
        else:
            print(f"Your engine speed is comparable to Stockfish (ratio: {time_ratio:.2f})")

    # Print game phase analysis
    move_patterns = results.get("move_patterns", {})
    print("\n=== MOVE PATTERN ANALYSIS ===")

    # Opening analysis
    opening_patterns = move_patterns.get("opening_patterns", {})
    if opening_patterns:
        opening_total = sum(opening_patterns.values())
        print("\nOpening phase patterns:")
        for pattern, count in sorted(opening_patterns.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  • {pattern}: {count}/{opening_total} ({count / opening_total * 100:.1f}%)")

    # Middlegame analysis
    middlegame_patterns = move_patterns.get("middlegame_patterns", {})
    if middlegame_patterns:
        middlegame_total = sum(middlegame_patterns.values())
        print("\nMiddlegame phase patterns:")
        for pattern, count in sorted(middlegame_patterns.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  • {pattern}: {count}/{middlegame_total} ({count / middlegame_total * 100:.1f}%)")

    # Piece preference analysis
    piece_prefs = move_patterns.get("piece_preference", {})
    if piece_prefs:
        print("\nPiece preference analysis:")

        your_prefs = piece_prefs.get("your_engine", {})
        stockfish_prefs = piece_prefs.get("stockfish", {})

        if your_prefs and stockfish_prefs:
            your_total = sum(your_prefs.values())
            stockfish_total = sum(stockfish_prefs.values())

            if your_total > 0 and stockfish_total > 0:
                print("  Piece usage frequency:")
                pieces = sorted(set(list(your_prefs.keys()) + list(stockfish_prefs.keys())))

                for piece in pieces:
                    your_pct = your_prefs.get(piece, 0) / your_total * 100
                    stockfish_pct = stockfish_prefs.get(piece, 0) / stockfish_total * 100
                    diff = your_pct - stockfish_pct

                    print(
                        f"  • {piece.capitalize()}: Your engine: {your_pct:.1f}%, Stockfish: {stockfish_pct:.1f}% (diff: {diff:+.1f}%)")

                    # Add interpretations for significant differences
                    if abs(diff) >= 10:
                        if diff > 0:
                            print(f"    - Your engine prefers using {piece}s more than Stockfish")
                        else:
                            print(f"    - Your engine uses {piece}s less frequently than Stockfish")

    # Print detailed analysis of interesting positions
    detailed = results.get("detailed_analysis", [])
    if detailed:
        print("\n=== MOST INTERESTING POSITION DIFFERENCES ===")
        for i, pos in enumerate(detailed[:min(5, len(detailed))]):
            print(f"\nPosition {i + 1}:")
            print(f"  FEN: {pos['fen']}")
            print(f"  Move number: {pos.get('fullmove_number', 'unknown')}")
            print(f"  Your move: {pos['your_move']}, Stockfish move: {pos['stockfish_move']}")

            analysis = pos.get("analysis", {})
            eval_diff = analysis.get("eval_diff", 0)
            print(f"  Evaluation difference: {eval_diff:.2f} pawns")

            # Print insights
            for insight in analysis.get("insights", []):
                print(f"  • {insight}")

    print("\n=== DETAILED SUGGESTIONS FOR IMPROVEMENT ===")
    move_comp = summary.get("move_comparison", {})

    # Collect all issues for categorized suggestions
    issues = {
        "evaluation": [],
        "piece_selection": [],
        "targeting": [],
        "captures": [],
        "technical": [],
        "game_phase": []
    }

    # Evaluation function suggestions
    if match_rate < 40:
        issues["evaluation"].append(
            "Your evaluation function likely needs significant improvement as the overall match rate is low")

    # Check if there are consistent differences in certain phases
    phases = ["opening_patterns", "middlegame_patterns", "endgame_patterns"]
    for phase in phases:
        phase_patterns = move_patterns.get(phase, {})
        if phase_patterns:
            phase_total = sum(phase_patterns.values())
            exact_matches = phase_patterns.get("exact_match", 0)
            if phase_total > 0 and exact_matches / phase_total < 0.3:
                phase_name = phase.split("_")[0].capitalize()
                issues["game_phase"].append(
                    f"{phase_name} play needs improvement (only {exact_matches}/{phase_total} matches)")

    # Piece selection suggestions
    if move_comp.get("different_piece_move", 0) > 0:
        percentage = move_comp.get("different_piece_move", 0) / total * 100
        issues["piece_selection"].append(
            f"Your engine selected completely different pieces in {percentage:.1f}% of positions")

    if move_comp.get("same_piece_type_different_location", 0) > 0:
        percentage = move_comp.get("same_piece_type_different_location", 0) / total * 100
        issues["piece_selection"].append(
            f"Your engine selected same piece types but from different locations in {percentage:.1f}% of positions")

    # Targeting suggestions
    if move_comp.get("same_piece_different_target", 0) > 0:
        percentage = move_comp.get("same_piece_different_target", 0) / total * 100
        issues["targeting"].append(
            f"Your engine selected correct pieces but different targets in {percentage:.1f}% of positions")

    # Capture logic suggestions
    if move_comp.get("capture_decision_difference", 0) > 0:
        percentage = move_comp.get("capture_decision_difference", 0) / total * 100
        issues["captures"].append(f"Your engine made different capture decisions in {percentage:.1f}% of positions")

    # Check piece preference differences
    piece_prefs = move_patterns.get("piece_preference", {})
    if piece_prefs:
        your_prefs = piece_prefs.get("your_engine", {})
        stockfish_prefs = piece_prefs.get("stockfish", {})

        if your_prefs and stockfish_prefs:
            your_total = sum(your_prefs.values())
            stockfish_total = sum(stockfish_prefs.values())

            if your_total > 0 and stockfish_total > 0:
                for piece in set(list(your_prefs.keys()) + list(stockfish_prefs.keys())):
                    your_pct = your_prefs.get(piece, 0) / your_total * 100
                    stockfish_pct = stockfish_prefs.get(piece, 0) / stockfish_total * 100
                    diff = your_pct - stockfish_pct

                    if abs(diff) >= 15:  # Only significant differences
                        if diff > 0:
                            issues["piece_selection"].append(
                                f"Your engine overuses {piece}s by {diff:.1f}% compared to Stockfish")
                        else:
                            issues["piece_selection"].append(
                                f"Your engine underuses {piece}s by {abs(diff):.1f}% compared to Stockfish")

    # Technical issues
    if move_comp.get("no_move", 0) > 0:
        issues["technical"].append(
            f"Fix cases where your engine doesn't return a move ({move_comp.get('no_move', 0)} occurrences)")

    if move_comp.get("illegal_move", 0) > 0:
        issues["technical"].append(
            f"Fix cases where your engine returns illegal moves ({move_comp.get('illegal_move', 0)} occurrences)")

    if move_comp.get("error_comparing", 0) > 0:
        issues["technical"].append(
            f"Fix error cases in move comparison ({move_comp.get('error_comparing', 0)} occurrences)")

    # Print all suggestions by category
    categories = {
        "technical": "Technical Issues",
        "evaluation": "Evaluation Function",
        "piece_selection": "Piece Selection Strategy",
        "targeting": "Move Targeting",
        "captures": "Capture Logic",
        "game_phase": "Game Phase Strategy"
    }

    for cat, title in categories.items():
        if issues[cat]:
            print(f"\n{title} Suggestions:")
            for suggestion in issues[cat]:
                print(f"- {suggestion}")

    # Specific improvement advice based on analysis
    detailed_analysis = results.get("detailed_analysis", [])
    if detailed_analysis:
        print("\nSpecific Improvements Based on Position Analysis:")

        # Collect common patterns from position analysis
        patterns = defaultdict(int)
        for pos in detailed_analysis:
            analysis = pos.get("analysis", {})
            for insight in analysis.get("insights", []):
                patterns[insight] += 1

        # Report top patterns
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            if count > 1:  # Only show recurring patterns
                print(f"- {pattern} (observed in {count} positions)")


def main():
    """Main function to run the comparison."""
    print("Chess Engine Comparison Tool")
    print("===========================")

    run_engine_comparison()


if __name__ == "__main__":
    main()
