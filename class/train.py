# File 3: train.py (modified version)
import numpy as np
import os
import chess
import chess.pgn
from model import create_chess_model, custom_cce, custom_mse
from dataset import board_to_input, move_to_policy
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint


def load_games(data_dir="lichess"):
    """Load all games from all PGN files in a directory"""
    games = []
    # Get list of all PGN files in directory
    pgn_files = [f for f in os.listdir(data_dir) if f.endswith('.pgn')]

    for pgn_file in pgn_files:
        file_path = os.path.join(data_dir, pgn_file)
        print(f"Processing {file_path}...")

        with open(file_path) as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:  # End of file
                        break
                    if (len(games) < 500):
                        games.append(game)
                    else :
                        print(f"Total games loaded: {len(games)}")
                        return games
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Error reading game: {e}")
                    continue

        print(f"Loaded {len(games)} games from {pgn_file}")

    print(f"Total games loaded: {len(games)}")
    return games


def generate_training_data(games, max_samples=None):
    """Generate training samples from games with memory management"""
    X, policies, values = [], [], []
    sample_count = 0

    for game_idx, game in enumerate(games):
        try:
            board = game.board()
            result = game.headers.get("Result", "1/2-1/2")

            # Convert result to numerical value
            if result == '1-0':
                result_value = 1
            elif result == '0-1':
                result_value = -1
            else:  # Draw or unknown
                result_value = 0

            for move in game.mainline_moves():
                # Convert board state to input tensor
                input_tensor = board_to_input(board)

                # Convert move to policy index
                policy_index = move_to_policy(move, board)

                if policy_index is not None:
                    # Create policy vector
                    policy_vector = np.zeros(4672)
                    policy_vector[policy_index] = 1.0

                    X.append(input_tensor)
                    policies.append(policy_vector)
                    values.append(result_value)

                    sample_count += 1
                    if max_samples and sample_count >= max_samples:
                        return np.array(X), np.array(policies), np.array(values)

                # Push move to board for next position
                board.push(move)

        except Exception as e:
            print(f"Error processing game {game_idx}: {e}")
            continue

        # Garbage collection for large datasets
        if game_idx % 1000 == 0:
            print(f"Processed {game_idx} games, collected {sample_count} samples")

    return np.array(X), np.array(policies), np.array(values)


def train_model():
    # Initialize model
    model = create_chess_model()
    model.compile(
        optimizer=Adam(0.001),
        loss={'policy': custom_cce, 'value': custom_mse},
        loss_weights=[0.7, 0.3],
        metrics={'policy': 'accuracy', 'value': 'mae'}
    )

    # Setup model checkpointing
    checkpoint = ModelCheckpoint('chess_model.h5',
                                 save_best_only=True,
                                 monitor='val_loss',
                                 mode='min')

    # Load and process data
    print("Loading games...")
    games = load_games("lichess")

    print("Generating training data...")
    X, policies, values = generate_training_data(games)

    print(f"Training data shape: {X.shape}")
    print(f"Policy targets shape: {policies.shape}")
    print(f"Value targets shape: {values.shape}")

    # Train model with validation split
    print("Starting training...")
    history = model.fit(X, [policies, values],
                        batch_size=256,
                        epochs=20,
                        validation_split=0.1,
                        callbacks=[checkpoint])

    print("Training complete. Saving final model...")
    model.save('chess_model_final.h5')


if __name__ == "__main__":
    train_model()