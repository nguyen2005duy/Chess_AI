import chess
import chess.pgn
import numpy as np
import os
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import time
import json
import re
from stockfish import Stockfish

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to Stockfish relative to the script
STOCKFISH_PATH = os.path.join(SCRIPT_DIR, "stockfish_ai/stockfish.exe")

# Configuration variables
CONFIG = {
    'output_base_dir': 'data',
    'eval_depth': 14,
    'min_ply': 12,
    'max_ply': 200,
    'max_eval': 600,
    'num_workers': max(1, multiprocessing.cpu_count() - 2),
    'games_per_batch': 500,
    'sample_rate': 0.4,
    'engine_hash': 64,
    'engine_threads': 1,
    'engine_timeout': 5.0,
    'pgn_paths': [
        "filter/filtered_pgns/2000-2199/2016_09_2000-2199_30000games.pgn",
        "filter/filtered_pgns/2200-2399/2016_09_2200-2399_20000games.pgn",
        "filter/filtered_pgns/2400-2599/2016_09_2400-2599_10000games.pgn",
        "filter/filtered_pgns/2400-2599/2016_08_2400-2599_10000games.pgn",
        "filter/filtered_pgns/2600-3000/2016_07_2600-3000_1782games.pgns"
    ],
    'min_elo': 0,
    'batch_size': 1000,
    'output_format': 'pkl',
    'progress_update_frequency': 10,
    'endgame_sample_boost': 1.5
}


def load_config_from_file(config_path='pgn_converter_config.json'):
    """Load configuration from a JSON file if it exists."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                CONFIG.update(user_config)
        except Exception as e:
            print(f"Error loading config: {e}")


def ensure_dir_exists(directory):
    """Ensures that the directory structure exists."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")


class PGNConverter:
    """Converts PGN files to NNUE training data using Stockfish evaluations."""

    def __init__(self, config):
        """Initialize the converter with the provided configuration."""
        self.config = config
        self.output_base_dir = config['output_base_dir']
        self.eval_depth = config['eval_depth']
        self.min_ply = config['min_ply']
        self.max_ply = config['max_ply']
        self.max_eval = config['max_eval']
        self.num_workers = config['num_workers']
        self.games_per_batch = config['games_per_batch']
        self.batch_size = config.get('batch_size', 1000)
        self.sample_rate = config['sample_rate']
        self.engine_hash = config['engine_hash']
        self.engine_threads = config['engine_threads']
        self.engine_timeout = config['engine_timeout']
        self.pgn_paths = config['pgn_paths']
        self.min_elo = config.get('min_elo', 0)
        self.output_format = config.get('output_format', 'pkl')
        self.progress_update_frequency = config.get('progress_update_frequency', 10)
        self.endgame_sample_boost = config.get('endgame_sample_boost', 1.5)

        # Create output directory if it doesn't exist
        ensure_dir_exists(self.output_base_dir)

        # Test if Stockfish library is working
        try:
            self._test_stockfish()
        except Exception as e:
            raise RuntimeError(f"Error testing Stockfish library: {e}")

    def _test_stockfish(self):
        """Test if Stockfish library is working properly."""
        try:
            # Verify that Stockfish exists at the specified path
            if not os.path.exists(STOCKFISH_PATH):
                raise FileNotFoundError(f"Stockfish executable not found at {STOCKFISH_PATH}")

            stockfish_instance = Stockfish(path=STOCKFISH_PATH)
            stockfish_instance.set_depth(3)
            stockfish_instance.set_position()
            result = stockfish_instance.get_evaluation()
            return result
        except Exception as e:
            raise RuntimeError(
                f"Error testing Stockfish library: {e}. Make sure Stockfish is installed at {STOCKFISH_PATH}")

    def _extract_dir_name_from_pgn(self, pgn_path):
        """Extract a clean directory name from the PGN path."""
        basename = os.path.basename(pgn_path)
        match = re.match(r'(\d{4}_\d{2}_\d{4}-\d{4}).*\.pgn', basename)
        if match:
            return match.group(1)
        else:
            return re.sub(r'_\d+games$', '', os.path.splitext(basename)[0])

    def _fen_to_features(self, fen):
        """Convert a FEN string to NNUE input features."""
        board = chess.Board(fen)
        white_features = np.zeros(768, dtype=np.float32)
        black_features = np.zeros(768, dtype=np.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = (piece.piece_type - 1) * 2 + (0 if piece.color else 1)

                # White's perspective
                white_index = square * 12 + piece_idx
                if 0 <= white_index < 768:  # Added bounds checking
                    white_features[white_index] = 1.0

                # Black's perspective
                black_square = square ^ 56  # Equivalent to chess.square_mirror(square)
                black_index = black_square * 12 + piece_idx
                if 0 <= black_index < 768:  # Added bounds checking
                    black_features[black_index] = 1.0

        side_to_move = board.turn
        return white_features, black_features, side_to_move

    def process_pgn_files(self):
        """Process all PGN files from specified paths."""
        if not self.pgn_paths:
            print("No PGN paths specified in configuration.")
            return

        pgn_files = [path for path in self.pgn_paths if os.path.isfile(path)]
        if not pgn_files:
            print("No valid PGN files found in the specified paths.")
            return

        for idx, pgn_path in enumerate(pgn_files, 1):
            print(f"Processing PGN file {idx}/{len(pgn_files)}: {pgn_path}")
            try:
                self._process_pgn_in_batches(pgn_path)
            except Exception as e:
                print(f"Error processing {pgn_path}: {e}")

    def _count_games_in_pgn(self, pgn_file):
        """Count or estimate the total number of games in a PGN file."""
        cache_dir = os.path.join(self.output_base_dir, "_cache")
        os.makedirs(cache_dir, exist_ok=True)

        file_hash = str(os.path.getsize(pgn_file)) + "_" + os.path.basename(pgn_file)
        cache_file = os.path.join(cache_dir, f"{file_hash}.count")

        # Check for game count in filename
        basename = os.path.basename(pgn_file)
        match = re.search(r'(\d+)games', basename)
        if match:
            games_count = int(match.group(1))
            print(f"Found game count in filename: {games_count}")
            with open(cache_file, 'w') as f:
                f.write(str(games_count))
            return games_count

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return int(f.read().strip())
            except:
                pass

        # Fast estimate based on file size
        file_size = os.path.getsize(pgn_file)
        estimated_games = max(1000, int(file_size / 2000))
        print(f"Estimated games count from file size: {estimated_games}")

        with open(cache_file, 'w') as f:
            f.write(str(estimated_games))

        return estimated_games

    def _process_pgn_in_batches(self, pgn_path):
        """Process a PGN file in sequential batches of specified size."""
        total_games = self._count_games_in_pgn(pgn_path)
        num_batches = (total_games + self.batch_size - 1) // self.batch_size
        dir_name = self._extract_dir_name_from_pgn(pgn_path)

        # Create pgn-specific output directory
        pgn_output_dir = os.path.join(self.output_base_dir, dir_name)
        os.makedirs(pgn_output_dir, exist_ok=True)

        # Create batch progress tracking file
        batch_progress_file = os.path.join(pgn_output_dir, f"batch_progress.json")

        # Load batch progress if it exists
        if os.path.exists(batch_progress_file):
            try:
                with open(batch_progress_file, 'r') as f:
                    batch_progress = json.load(f)
                    current_batch = batch_progress.get('current_batch', 1)
                    games_processed = batch_progress.get('games_processed', 0)
                    positions_evaluated = batch_progress.get('positions_evaluated', 0)
            except Exception:
                current_batch = 1
                games_processed = 0
                positions_evaluated = 0
        else:
            current_batch = 1
            games_processed = 0
            positions_evaluated = 0

        # Process each batch sequentially
        while current_batch <= num_batches:
            batch_file = os.path.join(pgn_output_dir, f"batch_{current_batch}.{self.output_format}")

            # Skip if batch file already exists
            if os.path.exists(batch_file):
                print(f"Batch {current_batch} already exists, skipping...")
                current_batch += 1
                games_processed += self.batch_size
                continue

            # Process current batch
            try:
                print(f"Processing batch {current_batch}/{num_batches}...")
                batch_result = self._process_batch(
                    pgn_path,
                    pgn_output_dir,
                    batch_num=current_batch,
                    start_game=games_processed,
                    games_limit=self.batch_size
                )
                games_in_batch = batch_result['games_processed']
                positions_in_batch = batch_result['positions_evaluated']

                # Update counts
                games_processed += games_in_batch
                positions_evaluated += positions_in_batch

                print(f"Completed batch {current_batch}: {games_in_batch} games, {positions_in_batch} positions")

                # Save batch progress
                with open(batch_progress_file, 'w') as f:
                    json.dump({
                        'current_batch': current_batch + 1,
                        'games_processed': games_processed,
                        'positions_evaluated': positions_evaluated,
                        'total_games': total_games,
                        'last_updated': time.time()
                    }, f)

                current_batch += 1

                # Exit if we've processed all games
                if games_processed >= total_games:
                    print(f"Processed all {total_games} games in {pgn_path}")
                    break

            except Exception as e:
                print(f"Error processing batch {current_batch}: {e}")
                current_batch += 1
                games_processed += self.batch_size  # Approximate, just to move forward

    def _process_batch(self, pgn_file, output_dir, batch_num, start_game=0, games_limit=1000):
        """Process a batch of games from the PGN file."""
        # Extract positions from the specified range of games
        positions_by_game = self._extract_positions_from_pgn(
            pgn_file,
            start_game,
            games_limit=games_limit
        )

        if not positions_by_game:
            print(f"No positions extracted from games starting at {start_game}")
            return {"games_processed": 0, "positions_evaluated": 0}

        # Count total positions
        total_positions = sum(len(pos) for pos in positions_by_game.values())
        print(f"Extracted {total_positions} positions from {len(positions_by_game)} games")

        # Process positions
        processed_data_by_game = self._process_positions_by_game(positions_by_game)

        # Count evaluated positions
        total_evaluated = sum(len(positions) for positions in processed_data_by_game.values())
        print(f"Successfully evaluated {total_evaluated}/{total_positions} positions")

        # Get clean name for output file prefix
        output_prefix = self._extract_dir_name_from_pgn(pgn_file)
        self._save_batch_data(processed_data_by_game, output_prefix, output_dir, batch_num, start_game)

        return {"games_processed": len(positions_by_game), "positions_evaluated": total_evaluated}

    def _is_endgame(self, board):
        """Determine if a position is an endgame based on material count."""
        piece_map = board.piece_map()
        piece_count = len(piece_map)

        # Count major pieces (queens and rooks)
        major_piece_count = sum(1 for piece in piece_map.values()
                                if piece.piece_type == chess.QUEEN or
                                piece.piece_type == chess.ROOK)

        # Consider it an endgame if:
        # 1. Total pieces <= 12, or
        # 2. No queens and <= 2 rooks
        if piece_count <= 12:
            return True

        queens = sum(1 for piece in piece_map.values() if piece.piece_type == chess.QUEEN)
        if queens == 0 and major_piece_count <= 2:
            return True

        return False

    def _extract_positions_from_pgn(self, pgn_file, start_game=0, games_limit=None):
        """Extract chess positions from PGN file with game-level tracking."""
        positions_by_game = {}
        games_processed = 0
        positions_extracted = 0

        # Create progress tracking file
        progress_dir = os.path.join(self.output_base_dir, "_progress")
        ensure_dir_exists(progress_dir)
        progress_file = os.path.join(progress_dir, f"extraction_progress_{start_game}.json")

        try:
            with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                game_count = 0

                # Skip to the starting game
                while game_count < start_game:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        return positions_by_game
                    game_count += 1

                # Calculate the max games to process
                max_games = games_limit if games_limit else float('inf')

                # Create progress bar
                pbar = tqdm(total=max_games if max_games < float('inf') else None,
                            desc=f"Extracting from games {start_game + 1}-{start_game + max_games if max_games < float('inf') else 'end'}")

                while games_processed < max_games:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    game_count += 1
                    games_processed += 1
                    pbar.update(1)

                    # Update progress tracking
                    if games_processed % self.progress_update_frequency == 0:
                        pbar.set_description(
                            f"Games: {games_processed}/{max_games} | Positions: {positions_extracted}"
                        )

                        # Save progress to file
                        with open(progress_file, 'w') as pf:
                            json.dump({
                                'timestamp': time.time(),
                                'games_processed': games_processed,
                                'positions_extracted': positions_extracted,
                                'batch_target': max_games,
                                'start_game': start_game
                            }, pf)

                    # Extract positions from game
                    board = game.board()
                    positions_this_game = []

                    for move in game.mainline_moves():
                        current_ply = board.ply()

                        # Apply ply filtering
                        if self.min_ply <= current_ply <= self.max_ply:
                            # Create a board copy for endgame detection
                            board_copy = chess.Board(board.fen())

                            # Base sampling rate
                            sample_rate = self.sample_rate

                            # Boost sampling rate for endgames
                            if self._is_endgame(board_copy):
                                sample_rate *= self.endgame_sample_boost

                            # Sample based on adjusted sampling rate
                            if random.random() < sample_rate:
                                positions_this_game.append(board.fen())
                                positions_extracted += 1

                        # Make the move
                        board.push(move)

                    # Store positions for this game if any
                    if positions_this_game:
                        # Use actual game index as key
                        game_idx = start_game + games_processed
                        positions_by_game[game_idx] = positions_this_game

                pbar.close()

                # Final update of the progress file
                with open(progress_file, 'w') as pf:
                    json.dump({
                        'timestamp': time.time(),
                        'games_processed': games_processed,
                        'positions_extracted': positions_extracted,
                        'batch_target': max_games,
                        'start_game': start_game,
                        'completed': True
                    }, pf)

            return positions_by_game

        except Exception as e:
            print(f"Error extracting positions from PGN: {e}")
            return {}

    def _process_positions_by_game(self, positions_by_game):
        """Process positions with game-level tracking."""
        # Flatten positions for processing but keep track of game_id
        position_list = []
        game_mapping = []

        for game_id, positions in positions_by_game.items():
            for pos in positions:
                position_list.append(pos)
                game_mapping.append(game_id)

        # Progress tracking
        positions_evaluated = 0
        progress_dir = os.path.join(self.output_base_dir, "_progress")
        ensure_dir_exists(progress_dir)
        stockfish_progress_file = os.path.join(progress_dir,
                                               f"stockfish_progress_{min(positions_by_game.keys()) if positions_by_game else 0}.json")

        # Process positions
        processed_data = []

        # Split positions into chunks for parallel processing
        chunk_size = max(10, (len(position_list) + self.num_workers - 1) // self.num_workers)
        position_chunks = [position_list[i:i + chunk_size] for i in range(0, len(position_list), chunk_size)]

        print(f"Processing {len(position_list)} positions in {len(position_chunks)} chunks...")
        print(f"Each worker will process approximately {chunk_size} positions")

        # Track progress
        completed_chunks = 0
        total_chunks = len(position_chunks)

        # Display progress bar
        with tqdm(total=len(position_list), desc="Stockfish evaluation progress") as pbar:
            # Track worker progress
            worker_progress = {i: 0 for i in range(len(position_chunks))}
            last_total_progress = 0

            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all chunks with explicit worker IDs
                futures = {}
                for i, chunk in enumerate(position_chunks):
                    futures[executor.submit(self._process_chunk, chunk, i)] = i

                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        worker_id = futures[future]
                        chunk_data = future.result(timeout=120)  # Extended timeout
                        chunk_size = len(chunk_data)
                        processed_data.extend(chunk_data)

                        # Update worker progress
                        worker_progress[worker_id] = chunk_size

                        # Calculate total progress
                        total_progress = sum(worker_progress.values())
                        progress_delta = total_progress - last_total_progress

                        # Update positions evaluated count
                        positions_evaluated += progress_delta
                        last_total_progress = total_progress

                        # Update progress bar
                        pbar.update(progress_delta)

                        # Update completed chunks counter
                        completed_chunks += 1

                        # Progress updates
                        pbar.set_description(
                            f"Stockfish eval: {positions_evaluated}/{len(position_list)} pos ({positions_evaluated * 100 // len(position_list)}%) - {completed_chunks}/{total_chunks} chunks"
                        )

                        # Save progress to file
                        with open(stockfish_progress_file, 'w') as pf:
                            json.dump({
                                'timestamp': time.time(),
                                'positions_evaluated': positions_evaluated,
                                'total_positions': len(position_list),
                                'chunks_completed': completed_chunks,
                                'total_chunks': total_chunks,
                                'percent_complete': positions_evaluated * 100 // len(position_list) if len(
                                    position_list) > 0 else 0
                            }, pf)

                        print(
                            f"Chunk {worker_id} complete: {chunk_size} positions processed. Total: {positions_evaluated}/{len(position_list)}")

                    except Exception as e:
                        print(f"Error processing chunk {futures[future]}: {e}")
                        completed_chunks += 1
                        pbar.update(chunk_size)

            print(f"Stockfish evaluation complete: {positions_evaluated}/{len(position_list)} positions processed")

        # Reorganize processed data by game
        processed_by_game = {}
        for i, data in enumerate(processed_data):
            if i < len(game_mapping):
                game_id = game_mapping[i]
                if game_id not in processed_by_game:
                    processed_by_game[game_id] = []
                processed_by_game[game_id].append(data)

        return processed_by_game

    def _process_chunk(self, positions, worker_id):
        """Process a chunk of positions using the Stockfish library."""
        print(f"Worker {worker_id} starting on {len(positions)} positions")

        try:
            # Initialize Stockfish with explicit path and parameters
            stockfish_instance = Stockfish(path=STOCKFISH_PATH, parameters={
                "Threads": self.engine_threads,
                "Hash": self.engine_hash
            })
            stockfish_instance.set_depth(self.eval_depth)
        except Exception as e:
            print(f"Error initializing Stockfish for worker {worker_id}: {e}")
            return []

        results = []
        update_interval = max(1, min(100, len(positions) // 20))  # More frequent updates

        for i, fen in enumerate(positions):
            # Print progress updates
            if i % update_interval == 0 or i == len(positions) - 1:
                print(
                    f"Worker {worker_id}: {i + 1}/{len(positions)} positions evaluated ({(i + 1) * 100 // len(positions)}%)")

            try:
                # Set position and get evaluation
                stockfish_instance.set_fen_position(fen)
                evaluation = stockfish_instance.get_evaluation()

                # Extract score in centipawns
                if evaluation['type'] == 'cp':
                    score = evaluation['value']
                elif evaluation['type'] == 'mate':
                    # Convert mate score to large cp value with proper sign
                    mate_score = evaluation['value']
                    score = 10000 if mate_score > 0 else -10000
                else:
                    continue  # Skip unknown evaluation types

                # Skip positions with extreme evaluations
                if abs(score) > self.max_eval:
                    continue

                # Extract features
                white_features, black_features, side_to_move = self._fen_to_features(fen)

                # Create data point
                data_point = {
                    'fen': fen,
                    'score': score,
                    'white_features': white_features,
                    'black_features': black_features,
                    'side_to_move': side_to_move
                }
                results.append(data_point)

            except Exception as e:
                # Skip problematic positions
                if i % update_interval == 0:
                    print(f"Worker {worker_id}: Error with position {i}: {e}")
                continue

        print(f"Worker {worker_id} finished: {len(results)}/{len(positions)} positions successfully evaluated")
        return results

    def _save_batch_data(self, processed_data_by_game, output_prefix, output_dir, batch_num, start_game):
        """Save processed data to a single batch file."""
        if not processed_data_by_game:
            print(f"No data to save for batch {batch_num}")
            return

        # Get all game IDs
        game_ids = sorted(list(processed_data_by_game.keys()))

        # Combine all games into one batch file
        all_batch_data = []
        for game_id in game_ids:
            all_batch_data.extend(processed_data_by_game[game_id])

        # Create batch file path
        batch_file = os.path.join(output_dir, f"batch_{batch_num}.{self.output_format}")

        # Save as binary file
        with open(batch_file, 'wb') as f:
            pickle.dump(all_batch_data, f)

        print(f"Saved {len(all_batch_data)} positions to {batch_file}")

        # Save batch metadata
        first_game = game_ids[0] if game_ids else start_game
        last_game = game_ids[-1] if game_ids else start_game
        meta_file = os.path.join(output_dir, f"batch_{batch_num}_meta.json")
        with open(meta_file, 'w') as f:
            json.dump({
                'batch_number': batch_num,
                'first_game': first_game,
                'last_game': last_game,
                'total_games': len(game_ids),
                'total_positions': len(all_batch_data),
                'created_time': time.time(),
                'pgn_source': output_prefix,
                'file_format': self.output_format,
                'min_elo': self.min_elo
            }, f)


def main():
    """Main function to run the converter."""
    print("Starting PGN Converter...")

    # Load configuration from file if it exists
    load_config_from_file()

    try:
        # Initialize and run converter
        converter = PGNConverter(CONFIG)
        converter.process_pgn_files()
        print("PGN Converter completed successfully!")
    except Exception as e:
        print(f"Error running PGN Converter: {e}")


if __name__ == "__main__":
    main()