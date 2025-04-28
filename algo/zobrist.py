import chess
import random

# Zobrist keys: 64 squares * 12 piece types (6 white, 6 black) + 1 black to move + 4 castling rights + 8 en passant files
NUM_PIECE_TYPES = 12 # P, N, B, R, Q, K (white), p, n, b, r, q, k (black)
# Flatten ZOBRIST_TABLE to 2D: [piece_index][square]
ZOBRIST_TABLE = [[0] * 64 for _ in range(NUM_PIECE_TYPES)] # Piece keys (2D)
ZOBRIST_BLACK_TO_MOVE = 0
ZOBRIST_CASTLING_RIGHTS = [0] * 16 # Indexed by a 4-bit castling rights index (KQkq)
ZOBRIST_EN_PASSANT_FILE = [0] * 8 # Indexed by file (0-7)

# Piece type mapping for Zobrist table index
PIECE_TO_ZOBRIST_INDEX = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

def init_zobrist():
    """Initializes the Zobrist key table with random 64-bit numbers."""
    global ZOBRIST_TABLE, ZOBRIST_BLACK_TO_MOVE, ZOBRIST_CASTLING_RIGHTS, ZOBRIST_EN_PASSANT_FILE

    # Use random.getrandbits(64) for clarity and potential performance
    for i in range(NUM_PIECE_TYPES):
        for j in range(64):
            # Use 2D indexing for flattened table
            ZOBRIST_TABLE[i][j] = random.getrandbits(64)

    ZOBRIST_BLACK_TO_MOVE = random.getrandbits(64)

    for i in range(16):
        ZOBRIST_CASTLING_RIGHTS[i] = random.getrandbits(64)

    for i in range(8):
        ZOBRIST_EN_PASSANT_FILE[i] = random.getrandbits(64)

def calculate_zobrist_hash(board: chess.Board):
    """Calculates the Zobrist hash for a given python-chess board state."""
    hash_key = 0

    # Piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_index = PIECE_TO_ZOBRIST_INDEX.get((piece.piece_type, piece.color))
            if piece_index is not None:
                # Use 2D indexing for flattened table
                hash_key ^= ZOBRIST_TABLE[piece_index][square]

    # Side to move
    if board.turn == chess.BLACK:
        hash_key ^= ZOBRIST_BLACK_TO_MOVE

    # Castling rights - Use the corrected 4-bit index calculation
    castling_index = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_index |= 1 << 0 # K bit
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_index |= 1 << 1 # Q bit
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_index |= 1 << 2 # k bit
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_index |= 1 << 3 # q bit
    hash_key ^= ZOBRIST_CASTLING_RIGHTS[castling_index]

    # En passant square
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        hash_key ^= ZOBRIST_EN_PASSANT_FILE[ep_file]

    return hash_key

# Initialize keys when the module is imported
init_zobrist()

# Example Usage
if __name__ == "__main__":
    board1 = chess.Board()
    hash1 = calculate_zobrist_hash(board1)
    print(f"Initial Hash: {hash1:016x}")

    board1.push_san("e4")
    hash2 = calculate_zobrist_hash(board1)
    print(f"Hash after e4: {hash2:016x}")

    board1.push_san("c5")
    hash3 = calculate_zobrist_hash(board1)
    print(f"Hash after c5: {hash3:016x}")

    # Test incremental update (conceptual - needs implementation in board_state.py)
    # Update example to use 2D indexing
    # hash_after_e4_manual = hash1 ^ ZOBRIST_TABLE[PIECE_TO_ZOBRIST_INDEX[(chess.PAWN, chess.WHITE)]][chess.E2] \
    #                        ^ ZOBRIST_TABLE[PIECE_TO_ZOBRIST_INDEX[(chess.PAWN, chess.WHITE)]][chess.E4] \
    #                        ^ ZOBRIST_BLACK_TO_MOVE \
    #                        ^ ZOBRIST_EN_PASSANT_FILE[chess.square_file(chess.E3)] # Assuming e3 is ep square
    # print(f"Manual Hash e4:{hash_after_e4_manual:016x}") # Won't match exactly due to EP/castling changes