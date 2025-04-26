import chess
import chess.engine
import chess.pgn
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from model import OptimizedChessNet  # Import the ChessConvNet model


# Keep board_to_tensor function consistent with the training code
def board_to_tensor(board):
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    tensor = torch.zeros(19, 8, 8)

    # 12 channels for piece positions (6 piece types * 2 colors)
    for i, piece in enumerate(pieces):
        for square in chess.SQUARES:
            if board.piece_at(square) and board.piece_at(square).symbol() == piece:
                tensor[i, 7 - chess.square_rank(square), chess.square_file(square)] = 1

    # 1 channel for en passant
    if board.ep_square:
        tensor[12, 7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1

    # 4 channels for castling rights
    tensor[13, 0, 0] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[14, 0, 0] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[15, 0, 0] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[16, 0, 0] = board.has_queenside_castling_rights(chess.BLACK)

    # 1 channel for side to move
    tensor[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # 1 channel for move count (normalized)
    tensor[18, :, :] = min(board.fullmove_number / 100.0, 1.0)

    return tensor


def move_to_index(move):
    """Convert a chess.Move to an index between 0 and 1967"""
    from_sq = move.from_square
    to_sq = move.to_square

    # Regular moves: 64 * 64 = 4096 possible combinations
    # But we'll optimize by recognizing most moves are to nearby squares
    # We'll use a more compact encoding with promotion info

    # Calculate move index based on starting square, target square, and promotion
    promotion_offset = 0
    if move.promotion:
        # Chess pieces for promotion: knight=2, bishop=3, rook=4, queen=5
        # Subtract 2 to get 0-3 range
        promotion_offset = move.promotion - 1

    # Base index from squares
    move_index = from_sq * 64 + to_sq

    # Add promotion offset
    if promotion_offset > 0:
        # Adjust index for promotions (4 promotion types * 56 possible promotion moves)
        move_index = 4096 + (promotion_offset - 1) * 56 + (to_sq - from_sq - 7)

    return min(move_index, 1967)  # Ensure we don't exceed our output size


def index_to_move(index, board):
    """Convert an index to a legal chess.Move"""
    # Get all legal moves
    legal_moves = list(board.legal_moves)

    # Get all move indices
    move_indices = [move_to_index(move) for move in legal_moves]

    # Find the closest match (in case our encoding isn't perfect)
    if index in move_indices:
        return legal_moves[move_indices.index(index)]
    else:
        # Return the move with index closest to the target
        closest_idx = min(range(len(move_indices)), key=lambda i: abs(move_indices[i] - index))
        return legal_moves[closest_idx]


class ChessMinimaxEngine:
    def __init__(self, model_path, depth=3, device=None):
        """
        Chess engine using minimax with neural network evaluation

        Args:
            model_path: Path to saved ChessConvNet model
            depth: Maximum depth for minimax search
            device: Torch device to use (cuda or cpu)
        """
        self.depth = depth
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the ChessConvNet model
        self.model = ChessConvNet(input_channels=19, filters=128, num_blocks=10).to(self.device)

        # Load saved weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is the direct state dict
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Cache for evaluated positions
        self.eval_cache = {}

        print(f"Loaded ChessConvNet model from {model_path} on {self.device}")

    def evaluate_position(self, board):
        """Evaluate a position using the neural network"""
        board_hash = hash(board.fen())

        # Check cache first
        if board_hash in self.eval_cache:
            return self.eval_cache[board_hash]

        # Convert board to tensor
        state_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get policy and value from the network
            _, value = self.model(state_tensor)

            # Scale value to be from the perspective of the current player
            value = value.item() if board.turn == chess.WHITE else -value.item()

        # Cache the evaluation
        self.eval_cache[board_hash] = value

        return value

    def get_policy_scores(self, board):
        """Get policy scores for all legal moves"""
        # Convert board to tensor
        state_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get policy logits from the network
            policy_logits, _ = self.model(state_tensor)

            # Get legal moves and their indices
            legal_moves = list(board.legal_moves)
            legal_indices = [move_to_index(move) for move in legal_moves]

            # Create a mask of legal moves
            mask = torch.zeros(1968).to(self.device)
            mask[legal_indices] = 1.0

            # Apply mask to policy logits
            masked_logits = policy_logits.clone()
            masked_logits[0][mask == 0] = float('-inf')

            # Convert to probabilities
            probabilities = F.softmax(masked_logits, dim=1)

            # Get scores for each legal move
            move_scores = {}
            for i, move in enumerate(legal_moves):
                move_idx = legal_indices[i]
                move_scores[move] = probabilities[0][move_idx].item()

        return move_scores

    def minimax(self, board, depth, alpha=-float('inf'), beta=float('inf'), maximizing=True):
        """Minimax algorithm with alpha-beta pruning"""
        # Check if board is in terminal state
        if board.is_game_over():
            if board.is_checkmate():
                return -1000 if maximizing else 1000
            else:
                return 0  # Draw

        # If we've reached the maximum depth, evaluate position
        if depth == 0:
            return self.evaluate_position(board)

        # Get all legal moves
        legal_moves = list(board.legal_moves)

        # If maximizing (white)
        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        # If minimizing (black)
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def get_best_move(self, board, depth=None):
        """Get the best move using minimax with the neural network policy as move ordering"""
        depth = depth or self.depth

        # Get policy scores for move ordering
        move_scores = self.get_policy_scores(board)
        legal_moves = list(move_scores.keys())

        # Sort moves by policy scores for better pruning
        legal_moves.sort(key=lambda move: move_scores[move], reverse=True)

        # Track best move and its evaluation
        best_move = None
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')

        # Alpha-beta bounds
        alpha = -float('inf')
        beta = float('inf')

        # Evaluate each move
        for move in legal_moves:
            board.push(move)

            # Get evaluation from minimax
            if board.turn == chess.WHITE:  # White's turn before the move, so now it's Black's
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
            else:  # Black's turn before the move
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                if eval < best_eval:
                    best_eval = eval
                    best_move = move
                beta = min(beta, eval)

            board.pop()

            # Check for alpha-beta cutoff
            if beta <= alpha:
                break

        return best_move

    def iterative_deepening(self, board, max_depth=None, time_limit=None):
        """
        Perform iterative deepening search to find the best move

        Args:
            board: Chess board position
            max_depth: Maximum depth to search
            time_limit: Time limit in seconds

        Returns:
            best_move: Best move found within constraints
        """
        max_depth = max_depth or self.depth
        start_time = time.time()
        best_move = None

        # Start with depth 1 and increase until max_depth or time limit
        for current_depth in range(1, max_depth + 1):
            if time_limit and time.time() - start_time > time_limit * 0.8:
                break  # Stop if we've used 80% of our time limit

            move = self.get_best_move(board, current_depth)
            if move:
                best_move = move

            # Check time again to see if we should stop
            if time_limit and time.time() - start_time > time_limit * 0.95:
                break

        return best_move or (list(board.legal_moves)[0] if board.legal_moves else None)


class SimpleStockfishBenchmark:
    def __init__(self, stockfish_path="stockfish", nn_model_path="chess_model.pth",
                 time_limit=0.1, nn_depth=3, stockfish_elo=500):
        """
        Initialize a simple benchmark between neural network minimax engine and Stockfish

        Args:
            stockfish_path: Path to the Stockfish executable
            nn_model_path: Path to the trained neural network model
            time_limit: Time limit for each move in seconds
            nn_depth: Depth for the neural network minimax search
            stockfish_elo: Target Stockfish ELO rating (supports low ELO)
        """
        self.stockfish_path = stockfish_path
        self.nn_model_path = nn_model_path
        self.time_limit = time_limit
        self.nn_depth = nn_depth
        self.stockfish_elo = stockfish_elo

        # Load the neural network engine (updated to use our defined ChessMinimaxEngine)
        self.nn_engine = ChessMinimaxEngine(nn_model_path, depth=nn_depth)

    def init_stockfish(self):
        """Initialize and configure Stockfish engine with support for very low ELO"""
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

        # Configure Stockfish's strength to low ELO
        engine.configure({"UCI_LimitStrength": True})

        try:
            # Your Stockfish version has a minimum ELO of 1320
            # Setting to minimum allowed value
            engine.configure({"UCI_Elo": max(1320, min(self.stockfish_elo, 3000))})
        except Exception as e:
            print(f"Warning when setting UCI_Elo: {e}")
            print("Trying alternative methods to weaken engine...")

        # Use skill level to make Stockfish play poorly
        try:
            engine.configure({"Skill Level": 0})  # Lowest skill level
        except:
            pass

        # Apply multiple handicaps to make Stockfish very weak
        try:
            engine.configure({"Hash": 1})  # Minimum hash size (1MB)
            engine.configure({"MultiPV": 1})  # Consider fewer variations
            engine.configure({"Move Overhead": 500})  # Add significant move overhead
            engine.configure({"Contempt": 0})  # Neutral contempt
            engine.configure({"Threads": 1})  # Single thread
        except:
            pass

        # For newer Stockfish versions that support these options
        try:
            # Limit search depth drastically
            engine.configure({"Maximum Depth": 1})
        except:
            pass

        try:
            # Some versions support this option to add randomness
            engine.configure({"Nodestime": 1})
        except:
            pass

        return engine

    def get_stockfish_move(self, engine, board):
        """Get a move from Stockfish with the given time limit"""
        # Use very limited time and depth to weaken Stockfish
        result = engine.play(board, chess.engine.Limit(time=self.time_limit, depth=1))
        return result.move

    def get_nn_move(self, board):
        """Get a move from the neural network engine"""
        move = self.nn_engine.iterative_deepening(board, max_depth=self.nn_depth, time_limit=self.time_limit)
        return move

    def play_game(self, stockfish_engine, nn_plays_white=True, display=False):
        """
        Play a full game between neural network and Stockfish

        Args:
            stockfish_engine: Initialized Stockfish engine
            nn_plays_white: Whether NN engine plays as white
            display: Whether to display the game progress

        Returns:
            game: Chess PGN game object
        """
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "NN vs Stockfish"
        game.headers["White"] = "NN" if nn_plays_white else "Stockfish"
        game.headers["Black"] = "Stockfish" if nn_plays_white else "NN"

        node = game

        # Play until the game is over
        while not board.is_game_over(claim_draw=True):
            if display:
                print(board)
                print()

            # Determine which engine's turn it is
            nn_turn = (board.turn == chess.WHITE and nn_plays_white) or \
                      (board.turn == chess.BLACK and not nn_plays_white)

            if nn_turn:
                try:
                    move = self.get_nn_move(board)
                    engine_name = "NN"
                except Exception as e:
                    print(f"Error getting move from NN engine: {e}")
                    break
            else:
                try:
                    move = self.get_stockfish_move(stockfish_engine, board)
                    engine_name = "Stockfish"
                except Exception as e:
                    print(f"Error getting move from Stockfish: {e}")
                    break

            if display:
                print(f"{engine_name} plays: {board.san(move)}")

            # Record the move in the game
            node = node.add_variation(move)

            # Make the move on the board
            board.push(move)

        # Game over - determine the result
        if board.is_checkmate():
            result = "1-0" if board.turn == chess.BLACK else "0-1"
        elif board.is_stalemate() or board.is_insufficient_material() or \
                board.is_fifty_moves() or board.is_repetition():
            result = "1/2-1/2"
        else:
            result = "1/2-1/2"  # Default to draw for any other termination

        game.headers["Result"] = result

        if display:
            print(board)
            print(f"Game over: {result}")

        return game

    def run_benchmark(self, num_games=10, display_games=False):
        """
        Run a simple benchmark of multiple games

        Args:
            num_games: Number of games to play
            display_games: Whether to display each game's moves

        Returns:
            games: List of completed games
        """
        # Initialize Stockfish
        try:
            stockfish = self.init_stockfish()
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            print("Make sure the path to Stockfish executable is correct.")
            return []

        games = []

        try:
            for i in range(num_games):
                # Alternate which engine plays white
                nn_plays_white = (i % 2 == 0)

                print(f"Playing game {i + 1}/{num_games} - NN plays {'white' if nn_plays_white else 'black'}")

                # Play the game
                try:
                    game = self.play_game(stockfish, nn_plays_white, display=display_games)
                    games.append(game)
                    print(f"Game {i + 1} result: {game.headers['Result']}")
                except Exception as e:
                    print(f"Error in game {i + 1}: {e}")
                    continue

        finally:
            # Clean up Stockfish process
            try:
                stockfish.quit()
            except:
                pass

        print(f"Completed {len(games)} games")
        return games

    def save_pgn(self, games, output_file):
        """Save all games in PGN format"""
        if not games:
            print("No games to save")
            return

        with open(output_file, "w") as f:
            for game in games:
                print(game, file=f, end="\n\n")
        print(f"Saved {len(games)} games to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Simple benchmark for NN Chess Engine against low-ELO Stockfish")
    parser.add_argument("--stockfish", type=str, default="stockfish_ai/stockfish.exe", help="Path to Stockfish executable")
    parser.add_argument("--model", type=str, default="chess_rl_model/final_model.pth",
                        help="Path to neural network model")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--time", type=float, default=0.1, help="Time limit per move in seconds")
    parser.add_argument("--depth", type=int, default=3, help="Depth for neural network engine")
    parser.add_argument("--elo", type=int, default=2000, help="Target Stockfish ELO rating (minimum 1320)")
    parser.add_argument("--display", action="store_true", help="Display each game")
    parser.add_argument("--pgn", type=str, default="games.pgn", help="Output PGN file")

    args = parser.parse_args()

    benchmark = SimpleStockfishBenchmark(
        stockfish_path=args.stockfish,
        nn_model_path=args.model,
        time_limit=args.time,
        nn_depth=args.depth,
        stockfish_elo=args.elo
    )

    print(f"Starting benchmark: {args.games} games against Stockfish (target ELO: {args.elo})")
    games = benchmark.run_benchmark(num_games=args.games, display_games=args.display)

    # Save games
    benchmark.save_pgn(games, args.pgn)


if __name__ == "__main__":
    main()