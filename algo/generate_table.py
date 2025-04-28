# generate_tables.py
import sys
import os
# Add project root to path if needed, depending on how you run it
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Importing magic_bitboards to populate tables...")
from algo import magic_bitboards # Adjust import based on your structure

if __name__ == "__main__":
    magic_bitboards.populate_magic_tables()
    print("\nPopulation function finished.")
    # Recommend adding code here to save the populated
    # magic_bitboards.ROOK_ATTACKS and magic_bitboards.BISHOP_ATTACKS
    # tables to disk (e.g., using pickle).
    # See commented-out example in populate_magic_tables().