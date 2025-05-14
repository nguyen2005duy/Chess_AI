import random
import re
import chess


class OpeningBook:
    """
    Python implementation of an opening book for chess.
    Reads moves from a text file and provides weighted random selection of moves.
    """

    class BookMove:
        """Inner class representing a move in the opening book with its frequency"""

        def __init__(self, move_string, num_times_played):
            self.move_string = move_string
            self.num_times_played = num_times_played

        def __repr__(self):
            return f"{self.move_string} ({self.num_times_played})"

    def __init__(self, file_path):
        """
        Initialize the opening book from a file path.

        Args:
            file_path (str): Path to the opening book file
        """
        self.rng = random.Random()
        self.moves_by_position = {}

        # Read the book file
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Opening book file not found: {file_path}")
            return
        except Exception as e:
            print(f"Error reading opening book: {e}")
            return

        # Split by "pos" entries
        entries = content.strip().split("pos")[1:]

        for entry in entries:
            lines = entry.strip().split('\n')
            position_fen = lines[0].strip()

            book_moves = []
            for i in range(1, len(lines)):
                if not lines[i].strip():
                    continue

                move_data = lines[i].split()
                if len(move_data) == 2:
                    try:
                        move_string = move_data[0]
                        num_times_played = int(move_data[1])
                        book_moves.append(self.BookMove(move_string, num_times_played))
                    except ValueError:
                        # Skip if count is not a valid integer
                        continue

            if book_moves:
                self.moves_by_position[position_fen] = book_moves

    def has_book_move(self, position_fen):
        """
        Check if the given position has any book moves.

        Args:
            position_fen (str): FEN string of the position

        Returns:
            bool: True if position has book moves, False otherwise
        """
        clean_fen = self._remove_move_counters_from_fen(position_fen)
        return clean_fen in self.moves_by_position

    def try_get_book_move(self, position_fen, weight_pow=0.5):
        """
        Try to get a book move for the given position.

        Args:
            position_fen (str): FEN string of the position
            weight_pow (float): Power to apply to weights (0 = equal probability, 1 = proportional to frequency)

        Returns:
            tuple: (bool, str) - Success flag and move string
        """
        weight_pow = max(0, min(weight_pow, 1))  # Clamp between 0 and 1
        clean_fen = self._remove_move_counters_from_fen(position_fen)

        if clean_fen in self.moves_by_position:
            moves = self.moves_by_position[clean_fen]

            # Calculate weighted play counts
            def weighted_play_count(play_count):
                return int(play_count ** weight_pow)

            total_play_count = sum(weighted_play_count(move.num_times_played) for move in moves)

            # Calculate probabilities and cumulative probabilities
            weights = []
            weight_sum = 0

            for move in moves:
                weight = weighted_play_count(move.num_times_played) / total_play_count
                weight_sum += weight
                weights.append(weight)

            prob_cumul = []
            cumul = 0

            for i, weight in enumerate(weights):
                prob = weight / weight_sum
                cumul += prob
                prob_cumul.append(cumul)
                # Debug line if needed:
                # print(f"{moves[i].move_string}: {prob * 100:.2f}% (cumul = {cumul})")

            # Select move based on weighted probability
            random_val = self.rng.random()
            for i, threshold in enumerate(prob_cumul):
                if random_val <= threshold:
                    return True, chess.Move.from_uci(moves[i].move_string)

        return False, chess.Move.null()

    def _remove_move_counters_from_fen(self, fen):
        """
        Remove move counters from FEN string.

        Args:
            fen (str): Full FEN string

        Returns:
            str: FEN string without move counters
        """
        # Remove the halfmove and fullmove counters
        parts = fen.split()
        if len(parts) >= 4:
            return " ".join(parts[:4])
        return fen