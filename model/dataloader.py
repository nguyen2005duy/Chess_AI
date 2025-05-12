import os
import torch
import numpy as np
import pickle
import glob
from torch.utils.data import Dataset, DataLoader, random_split
import random


class NNUEDataset(Dataset):
    """Dataset class for NNUE training data."""

    def __init__(self, data_dir, batch_pattern="batch_*.pkl", cache_size=5000):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Directory containing batch files
            batch_pattern (str): Pattern to match batch files
            cache_size (int): Number of positions to cache in memory
        """
        self.data_dir = data_dir
        self.batch_files = glob.glob(os.path.join(data_dir, batch_pattern))
        self.batch_files.sort()  # Sort for consistent ordering

        self.position_counts = []
        self.total_positions = 0

        # Count positions in each file to build an index
        print(f"Indexing {len(self.batch_files)} batch files...")
        for batch_file in self.batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    positions_count = len(batch_data)
                    self.position_counts.append(positions_count)
                    self.total_positions += positions_count
            except Exception as e:
                print(f"Error reading {batch_file}: {e}")
                self.position_counts.append(0)

        print(f"Found {self.total_positions} total positions across {len(self.batch_files)} files")

        # Set up position index mapping
        self.position_index_map = []
        for batch_idx, count in enumerate(self.position_counts):
            self.position_index_map.extend([(batch_idx, pos_idx) for pos_idx in range(count)])

        # LRU cache for recently accessed batches
        self.cache = {}
        self.cache_size = cache_size
        self.cache_access_order = []

    def __len__(self):
        """Return the total number of positions."""
        return self.total_positions

    def _get_batch_data(self, batch_idx):
        """Load batch data from file or cache."""
        if batch_idx in self.cache:
            # Update access order
            self.cache_access_order.remove(batch_idx)
            self.cache_access_order.append(batch_idx)
            return self.cache[batch_idx]

        # Load from file
        batch_file = self.batch_files[batch_idx]
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)

            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove least recently used
                lru_batch = self.cache_access_order.pop(0)
                del self.cache[lru_batch]

            self.cache[batch_idx] = batch_data
            self.cache_access_order.append(batch_idx)

            return batch_data
        except Exception as e:
            print(f"Error loading batch {batch_file}: {e}")
            return []

    def __getitem__(self, idx):
        """Get a position by index."""
        if idx >= len(self.position_index_map):
            raise IndexError("Position index out of range")

        batch_idx, pos_idx = self.position_index_map[idx]
        batch_data = self._get_batch_data(batch_idx)

        if not batch_data or pos_idx >= len(batch_data):
            # Return a zero sample if data can't be loaded
            return {
                'white_features': torch.zeros(768, dtype=torch.float32),
                'black_features': torch.zeros(768, dtype=torch.float32),
                'side_to_move': torch.tensor(0, dtype=torch.float32),
                'score': torch.tensor(0, dtype=torch.float32),
                'fen': ''
            }

        position = batch_data[pos_idx]

        # Convert numpy arrays to PyTorch tensors
        white_features = torch.tensor(position['white_features'], dtype=torch.float32)
        black_features = torch.tensor(position['black_features'], dtype=torch.float32)

        # Normalize score from centipawns to -1 to 1 range
        # Sigmoid transformation for smoother training
        score = float(position['score']) / 1000.0  # Scale down, using 1000 as max value
        score = torch.tensor(score, dtype=torch.float32)

        # Convert side_to_move to binary tensor (0: black, 1: white)
        side_to_move = torch.tensor(float(position['side_to_move']), dtype=torch.float32)

        return {
            'white_features': white_features,
            'black_features': black_features,
            'side_to_move': side_to_move,
            'score': score,
            'fen': position.get('fen', '')  # For debugging
        }


def create_data_loaders(data_dir, batch_size=1024, val_split=0.1, test_split=0.05,
                        num_workers=4, cache_size=2000, seed=42):
    """
    Create train, validation, and test data loaders.

    Args:
        data_dir (str): Directory containing batch files
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
        test_split (float): Fraction of data to use for testing
        num_workers (int): Number of worker processes for data loading
        cache_size (int): Number of positions to cache in memory
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Create full dataset
    dataset = NNUEDataset(data_dir, cache_size=cache_size)

    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - val_size - test_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test positions")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Example usage:
if __name__ == "__main__":
    # Example usage
    data_dir = "../data/2016_07_2000-2199"  # Path to directory with batch files

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir,
        batch_size=128,
        val_split=0.1,
        num_workers=4
    )

    # Display sample batch
    for batch in train_loader:
        print(f"Batch size: {len(batch['score'])}")
        print(f"White features shape: {batch['white_features'].shape}")
        print(f"Black features shape: {batch['black_features'].shape}")
        print(f"Side to move shape: {batch['side_to_move'].shape}")
        print(f"Score mean: {batch['score'].mean().item()}")
        break