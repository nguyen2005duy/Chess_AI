import chess
import chess.engine
import time
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
STOCKFISH_ELO = 1800
DEBUG_POSITIONS = 100  # Number of positions to analyze
STOCKFISH_DEPTH = 15  # Depth for "ground truth" analysis
ENGINE_DEPTH_LIMIT = 6
ENGINE_TIME_LIMIT_S = 5.0
OUTPUT_DIR = "engine_debug"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        # Fallback to minimal configuration
        if elo is not None:
            try:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
                print(f"Fallback configuration: Stockfish set to {elo} ELO")
            except chess.engine.EngineError:
                print("Warning: Could not set ELO rating. Engine may play at full strength.")

    return engine


def get_position_evaluation(engine, board, depth=15):
    """Get a detailed position evaluation from the engine."""
    result = {}

    # Get the principal variation and score
    info = engine.analyse(board, chess.engine.Limit(depth=depth))

    # Extract the score
    if "score" in info:
        score = info["score"].white().score(mate_score=10000)
        result["score"] = score / 100.0  # Convert centipawns to pawns

    # Extract the best move
    if "pv" in info and len(info["pv"]) > 0:
        result["best_move"] = info["pv"][0].uci()

    # Extract the principal variation
    if "pv" in info:
        result["pv"] = [move.uci() for move in info["pv"]]

    # Extract depth reached
    if "depth" in info:
        result["depth"] = info["depth"]

    # Extract nodes searched
    if "nodes" in info:
        result["nodes"] = info["nodes"]

    # Extract time
    if "time" in info:
        result["time"] = info["time"]

    return result


def analyze_position(board, engine, engine_name="Stockfish", depth=15, time_limit=None):
    """Analyze a position with the given engine."""
    limit = chess.engine.Limit(depth=depth) if time_limit is None else chess.engine.Limit(time=time_limit)

    start_time = time.time()
    info = engine.analyse(board, limit)
    end_time = time.time()

    # Extract results
    result = {
        "engine": engine_name,
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
        result["pv"] = [move.uci() for move in info["pv"]]
    else:
        result["best_move"] = None
        result["pv"] = []

    # Extract depth
    result["depth"] = info.get("depth", 0)

    # Extract nodes
    result["nodes"] = info.get("nodes", 0)

    return result


def analyze_with_your_engine(board, max_depth=ENGINE_DEPTH_LIMIT, time_limit=ENGINE_TIME_LIMIT_S):
    """Analyze a position with your custom engine."""
    start_time = time.time()

    # Call your engine's find_best_move function
    best_move = find_best_move(board, max_depth=max_depth, time_limit_seconds=time_limit)

    end_time = time.time()

    result = {
        "engine": "YourEngine",
        "fen": board.fen(),
        "time_taken": end_time - start_time,
        "best_move": best_move.uci() if best_move else None,
    }

    return result


def generate_test_positions(num_positions=100, ply_range=(5, 40)):
    """Generate a set of test positions by playing random games."""
    positions = []

    print(f"Generating {num_positions} test positions...")

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        configure_stockfish_engine(engine, STOCKFISH_ELO)

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
                print(f"Error generating position: {e}")

    print(f"Generated {len(positions)} unique positions")
    return positions


def compare_engine_moves(your_move, stockfish_move, legal_moves):
    """Compare the engine moves and classify the difference."""
    if your_move == stockfish_move:
        return "exact_match"

    # Convert moves to chess.Move objects for easier comparison
    try:
        your_move_obj = chess.Move.from_uci(your_move) if your_move else None
        stockfish_move_obj = chess.Move.from_uci(stockfish_move) if stockfish_move else None

        # Check if either move is None or illegal
        if your_move is None:
            return "no_move"
        if your_move_obj not in legal_moves:
            return "illegal_move"

        # TODO: Add more sophisticated move comparison logic here
        # For example, you might want to check if the moves attack/defend similar squares,
        # capture the same piece, etc.

        return "different_move"
    except Exception as e:
        print(f"Error comparing moves: {e}")
        return "error"


def run_engine_diagnostics():
    """Run comprehensive diagnostics on your chess engine."""
    results = {
        "positions": [],
        "summary": {},
        "errors": []
    }

    try:
        print("Starting chess engine diagnostics...")

        # Generate test positions
        test_positions = generate_test_positions(DEBUG_POSITIONS)

        # Initialize Stockfish
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
            configure_stockfish_engine(stockfish)  # Full strength for ground truth

            print(f"Analyzing {len(test_positions)} positions...")

            move_comparisons = defaultdict(int)
            eval_diffs = []
            time_ratios = []

            for idx, position in enumerate(tqdm(test_positions)):
                try:
                    fen = position["fen"]
                    board = chess.Board(fen)
                    legal_moves = list(board.legal_moves)

                    # Skip positions with very few options
                    if len(legal_moves) < 3:
                        continue

                    # Analyze with Stockfish (ground truth)
                    stockfish_analysis = analyze_position(
                        board, stockfish, engine_name="Stockfish", depth=STOCKFISH_DEPTH
                    )

                    # Analyze with your engine
                    your_analysis = analyze_with_your_engine(
                        board, max_depth=ENGINE_DEPTH_LIMIT, time_limit=ENGINE_TIME_LIMIT_S
                    )

                    # Compare results
                    comparison = {
                        "position_id": idx,
                        "fen": fen,
                        "stockfish": stockfish_analysis,
                        "your_engine": your_analysis,
                        "legal_move_count": len(legal_moves)
                    }

                    # Classify move comparison
                    move_comparison_type = compare_engine_moves(
                        your_analysis.get("best_move"),
                        stockfish_analysis.get("best_move"),
                        legal_moves
                    )
                    comparison["move_comparison"] = move_comparison_type
                    move_comparisons[move_comparison_type] += 1

                    # Calculate time ratio (your engine / stockfish)
                    if stockfish_analysis.get("time_taken", 0) > 0:
                        time_ratio = your_analysis.get("time_taken", 0) / stockfish_analysis.get("time_taken", 0)
                        time_ratios.append(time_ratio)
                        comparison["time_ratio"] = time_ratio

                    results["positions"].append(comparison)

                except Exception as e:
                    error_msg = f"Error analyzing position {idx}: {e}"
                    print(error_msg)
                    results["errors"].append(error_msg)

            # Calculate summary statistics
            total_positions = len(results["positions"])
            if total_positions > 0:
                results["summary"] = {
                    "total_positions": total_positions,
                    "move_comparison": dict(move_comparisons),
                    "exact_match_percentage": move_comparisons[
                                                  "exact_match"] / total_positions * 100 if total_positions > 0 else 0,
                    "average_time_ratio": np.mean(time_ratios) if time_ratios else None,
                    "median_time_ratio": np.median(time_ratios) if time_ratios else None,
                }

        # Save results to file
        with open(os.path.join(OUTPUT_DIR, "diagnostic_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"Diagnostics complete. Results saved to {os.path.join(OUTPUT_DIR, 'diagnostic_results.json')}")

        # Generate plots
        generate_diagnostic_plots(results)

        return results

    except Exception as e:
        print(f"Error in diagnostics: {e}")
        import traceback
        traceback.print_exc()
        results["errors"].append(str(e))
        return results


def generate_diagnostic_plots(results):
    """Generate diagnostic plots from the results."""
    try:
        os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

        # Move comparison pie chart
        if "move_comparison" in results["summary"]:
            move_comp = results["summary"]["move_comparison"]
            labels = list(move_comp.keys())
            sizes = list(move_comp.values())

            plt.figure(figsize=(10, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Move Comparison Distribution')
            plt.savefig(os.path.join(OUTPUT_DIR, "plots", "move_comparison_pie.png"))
            plt.close()

        # Time ratio histogram
        time_ratios = [pos.get("time_ratio", 0) for pos in results["positions"] if "time_ratio" in pos]
        if time_ratios:
            plt.figure(figsize=(10, 6))
            plt.hist(time_ratios, bins=20, alpha=0.7)
            plt.axvline(np.median(time_ratios), color='r', linestyle='dashed', linewidth=1)
            plt.axvline(np.mean(time_ratios), color='g', linestyle='dashed', linewidth=1)
            plt.text(np.median(time_ratios), 0, f'Median: {np.median(time_ratios):.2f}', color='r')
            plt.text(np.mean(time_ratios), 0, f'Mean: {np.mean(time_ratios):.2f}', color='g')
            plt.title('Time Ratio (Your Engine / Stockfish)')
            plt.xlabel('Time Ratio')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(OUTPUT_DIR, "plots", "time_ratio_hist.png"))
            plt.close()

        print(f"Plots saved to {os.path.join(OUTPUT_DIR, 'plots')}")

    except Exception as e:
        print(f"Error generating plots: {e}")


def extract_patterns_from_results(results):
    """Extract patterns from the diagnostic results to understand engine weaknesses."""
    patterns = {
        "tactical_misses": [],
        "positional_errors": [],
        "endgame_errors": [],
        "opening_errors": [],
        "time_issues": []
    }

    # Helper function to estimate game phase
    def estimate_game_phase(fen):
        parts = fen.split()
        piece_count = sum(c.isalpha() for c in parts[0])
        if piece_count > 26:  # More than 10 pieces per side on average
            return "opening"
        elif piece_count > 16:  # More than 8 pieces per side
            return "middlegame"
        else:
            return "endgame"

    for position in results["positions"]:
        fen = position["fen"]
        board = chess.Board(fen)
        phase = estimate_game_phase(fen)

        # Skip exact matches
        if position.get("move_comparison") == "exact_match":
            continue

        # Check for time issues
        if position.get("time_ratio", 0) > 5:  # Your engine took 5x longer than Stockfish
            patterns["time_issues"].append({
                "fen": fen,
                "time_ratio": position.get("time_ratio"),
                "legal_moves": position.get("legal_move_count")
            })

        # Record phase-specific errors
        if phase == "opening":
            patterns["opening_errors"].append(fen)
        elif phase == "endgame":
            patterns["endgame_errors"].append(fen)

        # TODO: Add more sophisticated pattern detection
        # For example, you could check for tactical errors by seeing if your engine
        # missed a capture, check, or mate in N that Stockfish found

    return patterns


def analyze_search_behavior(board, max_depth=6, time_limit=5.0, stockfish_path=STOCKFISH_PATH):
    """
    Analyze your search algorithm's behavior compared to Stockfish.
    This function requires your search module to have some instrumentation.
    """
    # This is a placeholder - you'll need to add appropriate instrumentation to your search
    # algorithm to collect data about visited nodes, pruning, etc.
    search_stats = {
        "nodes_visited": 0,
        "quiescence_nodes": 0,
        "alpha_beta_cutoffs": 0,
        "transposition_hits": 0,
        "max_depth_reached": 0,
        "evaluation_calls": 0,
    }

    # Here you would call your engine's find_best_move with additional parameters to
    # collect the search statistics

    # Compare with Stockfish (if it can provide this info)

    return search_stats


def run_position_specific_test(fen, max_depth=8):
    """Run a detailed test on a specific position to debug issues."""
    print(f"Running detailed test on position: {fen}")

    board = chess.Board(fen)
    print(board)

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        configure_stockfish_engine(stockfish)

        # Get stockfish analysis at different depths
        stockfish_by_depth = {}
        for depth in range(1, max_depth + 1):
            info = stockfish.analyse(board, chess.engine.Limit(depth=depth))
            move = info["pv"][0].uci() if "pv" in info and info["pv"] else None
            score = info["score"].white().score(mate_score=10000) / 100.0 if "score" in info else None
            stockfish_by_depth[depth] = {"move": move, "score": score}

        print("\nStockfish analysis by depth:")
        for depth, data in stockfish_by_depth.items():
            print(f"Depth {depth}: {data['move']} (Score: {data['score']})")

        # Get your engine's analysis
        start_time = time.time()
        your_move = find_best_move(board, max_depth=max_depth, time_limit_seconds=30.0)
        end_time = time.time()

        print(f"\nYour engine's move: {your_move.uci() if your_move else 'None'}")
        print(f"Time taken: {end_time - start_time:.2f}s")

        # Compare with Stockfish
        stockfish_move = stockfish_by_depth[max_depth]["move"]
        print(f"Stockfish's move at depth {max_depth}: {stockfish_move}")

        if your_move and your_move.uci() == stockfish_move:
            print("✓ Moves match!")
        else:
            print("✗ Moves differ!")

            # Get detailed analysis of both moves
            if your_move:
                # Make your move
                board.push(your_move)
                # Get Stockfish's evaluation of the position after your move
                info = stockfish.analyse(board, chess.engine.Limit(depth=max_depth))
                your_move_eval = info["score"].white().score(mate_score=10000) / 100.0 if "score" in info else None
                # Undo your move
                board.pop()

                print(f"Evaluation after your move: {your_move_eval}")

            # Make Stockfish's move
            stockfish_move_obj = chess.Move.from_uci(stockfish_move) if stockfish_move else None
            if stockfish_move_obj:
                board.push(stockfish_move_obj)
                # Get Stockfish's evaluation of the position after its move
                info = stockfish.analyse(board, chess.engine.Limit(depth=max_depth))
                stockfish_move_eval = info["score"].white().score(mate_score=10000) / 100.0 if "score" in info else None
                # Undo Stockfish's move
                board.pop()

                print(f"Evaluation after Stockfish's move: {stockfish_move_eval}")

                if your_move and your_move_eval is not None and stockfish_move_eval is not None:
                    eval_diff = stockfish_move_eval - your_move_eval
                    # Account for perspective (positive values are good for the side to move)
                    if board.turn == chess.BLACK:
                        eval_diff = -eval_diff
                    print(f"Evaluation difference: {eval_diff:.2f}")

                    if abs(eval_diff) < 0.5:
                        print("The moves are quite close in evaluation.")
                    else:
                        print(f"Stockfish's move is significantly better by {abs(eval_diff):.2f} pawns.")


def test_evaluation_function(num_positions=50):
    """
    Test your evaluation function against Stockfish's evaluations.
    This requires your evaluation function to be exposed separately.
    """
    try:
        # Try to import your evaluation function
        from algo.evaluation import calculate_heuristic_score_from_board as your_evaluate
        print("Successfully imported your evaluation function.")
    except ImportError as e:
        print(f"Could not import your evaluation function: {e}")
        print("Make sure you have created the algo/evaluation.py file with the correct function.")
        return

    positions = generate_test_positions(num_positions)

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        configure_stockfish_engine(stockfish)

        results = []
        for position in tqdm(positions):
            fen = position["fen"]
            board = chess.Board(fen)

            # Get Stockfish's evaluation
            info = stockfish.analyse(board, chess.engine.Limit(depth=12))
            stockfish_eval = info["score"].white().score(mate_score=10000) / 100.0 if "score" in info else None

            # Get your evaluation
            try:
                your_eval = your_evaluate(board)

                # Compare
                results.append({
                    "fen": fen,
                    "stockfish_eval": stockfish_eval,
                    "your_eval": your_eval,
                    "diff": abs(
                        stockfish_eval - your_eval) if stockfish_eval is not None and your_eval is not None else None
                })
            except Exception as e:
                print(f"Error evaluating position {fen}: {e}")

        # Calculate statistics
        if results:
            diffs = [r["diff"] for r in results if r["diff"] is not None]
            if diffs:
                avg_diff = np.mean(diffs)
                median_diff = np.median(diffs)
                max_diff = np.max(diffs)

                print(f"Evaluation function test results:")
                print(f"Average difference: {avg_diff:.2f} pawns")
                print(f"Median difference: {median_diff:.2f} pawns")
                print(f"Maximum difference: {max_diff:.2f} pawns")

                # Plot correlation
                plt.figure(figsize=(10, 6))
                x = [r["stockfish_eval"] for r in results if
                     r["stockfish_eval"] is not None and r["your_eval"] is not None]
                y = [r["your_eval"] for r in results if r["stockfish_eval"] is not None and r["your_eval"] is not None]
                plt.scatter(x, y, alpha=0.7)
                # Linear regression line
                if x and y:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(x), "r--")
                plt.plot([-5, 5], [-5, 5], "k--")  # Perfect correlation line
                plt.xlabel("Stockfish Evaluation (pawns)")
                plt.ylabel("Your Evaluation (pawns)")
                plt.title("Evaluation Function Correlation")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(OUTPUT_DIR, "plots", "eval_correlation.png"))
                plt.close()

                # Add plotting of largest differences
                largest_diffs = sorted(results, key=lambda r: r["diff"] if r["diff"] is not None else 0, reverse=True)[
                                :5]
                print("\nPositions with largest evaluation differences:")
                for i, pos in enumerate(largest_diffs):
                    print(f"{i + 1}. FEN: {pos['fen']}")
                    print(
                        f"   Stockfish: {pos['stockfish_eval']:.2f}, Your Engine: {pos['your_eval']:.2f}, Diff: {pos['diff']:.2f}")

                    # Visualize these positions if matplotlib is available
                    try:
                        board = chess.Board(pos['fen'])
                        fig = plt.figure(figsize=(5, 5))
                        ax = fig.add_subplot(111)

                        # Draw the board
                        for square in chess.SQUARES:
                            x = square % 8
                            y = 7 - (square // 8)
                            color = 'white' if (x + y) % 2 == 0 else 'gray'
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))

                            piece = board.piece_at(square)
                            if piece:
                                piece_symbol = piece.symbol()
                                color = 'black' if piece.color == chess.WHITE else 'red'
                                ax.text(x + 0.5, y + 0.5, piece_symbol, fontsize=20,
                                        ha='center', va='center', color=color)

                        ax.set_xlim(0, 8)
                        ax.set_ylim(0, 8)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_aspect('equal')
                        plt.title(
                            f"Diff: {pos['diff']:.2f} pawns\nStock: {pos['stockfish_eval']:.2f}, Yours: {pos['your_eval']:.2f}")
                        plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"diff_position_{i + 1}.png"))
                        plt.close()
                    except Exception as e:
                        print(f"Could not visualize position: {e}")


def check_search_issues(max_depth=4):
    """
    Check for common issues in your search algorithm by analyzing simple test positions.
    This helps identify problems like horizon effect, quiescence search issues, etc.
    """
    # Test positions that might reveal specific issues
    test_positions = [
        # Horizon effect test (check if your engine can see beyond horizon)
        {"fen": "8/8/8/8/8/k7/2r5/K7 w - - 0 1", "name": "Horizon effect test"},

        # Quiescence search test (check if your engine handles captures properly)
        {"fen": "r1bqkbnr/ppp2ppp/2n5/3pp3/2B5/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4", "name": "Quiescence test"},

        # Material imbalance test
        {"fen": "r1bqk2r/ppp2ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 6", "name": "Material balance"},

        # King safety test
        {"fen": "r1bqk2r/ppp2ppp/2n2n2/4p3/2BpP3/2N2N2/PPP2PPP/R1BQK2R w KQkq - 0 7", "name": "King safety"},

        # Zugzwang position
        {"fen": "8/8/p7/1p6/1P6/P7/8/8 w - - 0 1", "name": "Zugzwang"}
    ]

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
        configure_stockfish_engine(stockfish)

        for test in test_positions:
            board = chess.Board(test["fen"])
            print(f"\nTesting {test['name']}: {test['fen']}")
            print(board)

            # Get Stockfish's evaluation and move
            stockfish_analysis = analyze_position(
                board, stockfish, engine_name="Stockfish", depth=max_depth * 2
            )

            # Get your engine's move
            start_time = time.time()
            your_move = find_best_move(board, max_depth=max_depth, time_limit_seconds=10.0)
            end_time = time.time()

            print(
                f"Stockfish move: {stockfish_analysis.get('best_move')} (eval: {stockfish_analysis.get('score', 'N/A')})")
            print(f"Your move: {your_move.uci() if your_move else 'None'} (time: {end_time - start_time:.2f}s)")

            # Make both moves and compare resulting positions
            if your_move and stockfish_analysis.get('best_move'):
                # Make your move
                board_after_your_move = board.copy()
                board_after_your_move.push(your_move)

                # Make Stockfish's move
                board_after_stockfish = board.copy()
                stockfish_move = chess.Move.from_uci(stockfish_analysis.get('best_move'))
                board_after_stockfish.push(stockfish_move)

                # Compare resulting positions
                your_result_analysis = analyze_position(
                    board_after_your_move, stockfish, engine_name="Stockfish", depth=max_depth
                )
                stockfish_result_analysis = analyze_position(
                    board_after_stockfish, stockfish, engine_name="Stockfish", depth=max_depth
                )

                eval_diff = (your_result_analysis.get('score', 0) or 0) - (
                            stockfish_result_analysis.get('score', 0) or 0)
                # Adjust for perspective
                if board.turn == chess.BLACK:
                    eval_diff = -eval_diff

                print(f"Position evaluation after your move: {your_result_analysis.get('score')}")
                print(f"Position evaluation after Stockfish move: {stockfish_result_analysis.get('score')}")
                print(f"Difference: {eval_diff:.2f} pawns")

                if abs(eval_diff) > 1.0:
                    print("⚠️ Significant evaluation difference detected!")


def main():
    """Main function to run all diagnostics."""
    print("Chess Engine Diagnostic Tool")
    print("==========================")

    while True:
        print("\nAvailable tests:")
        print("1. Run comprehensive engine diagnostics")
        print("2. Test a specific position")
        print("3. Check for common search issues")
        print("4. Run evaluation function test")
        print("5. Exit")

        choice = input("\nSelect an option (1-5): ")

        if choice == '1':
            run_engine_diagnostics()
        elif choice == '2':
            fen = input("Enter FEN position (or press Enter for default): ")
            if not fen:
                fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
            depth = input("Enter max depth (or press Enter for default 8): ")
            depth = int(depth) if depth.isdigit() else 8
            run_position_specific_test(fen, max_depth=depth)
        elif choice == '3':
            depth = input("Enter max depth (or press Enter for default 4): ")
            depth = int(depth) if depth.isdigit() else 4
            check_search_issues(max_depth=depth)
        elif choice == '4':
            num = input("Enter number of positions to test (or press Enter for default 50): ")
            num = int(num) if num.isdigit() else 50
            test_evaluation_function(num_positions=num)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a valid option (1-5).")

if __name__ == "__main__":
    main()