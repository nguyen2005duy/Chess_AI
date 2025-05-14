import chess

# Define bitboards for each piece type and color
# 64-bit integers will be used to represent the board state
WHITE_PAWNS = 0
WHITE_KNIGHTS = 0
WHITE_BISHOPS = 0
WHITE_ROOKS = 0
WHITE_QUEENS = 0
WHITE_KINGS = 0

BLACK_PAWNS = 0
BLACK_KNIGHTS = 0
BLACK_BISHOPS = 0
BLACK_ROOKS = 0
BLACK_QUEENS = 0
BLACK_KINGS = 0

# Aggregate bitboards
WHITE_PIECES = 0
BLACK_PIECES = 0
ALL_PIECES = 0
BITBOARDS = [0] * 12
OCCUPATIONS = [0] * 3
# Game state variables
SIDE_TO_MOVE = chess.WHITE
CASTLING_RIGHTS = 0 # Use bitflags for castling rights (e.g., 1 for white kingside, 2 for white queenside, 4 for black kingside, 8 for black queenside)
EN_PASSANT_SQUARE = None # Store the index of the en passant target square, or None
HALF_MOVE_COUNTER = 0 # Number of half-moves since the last pawn advance or capture
FULL_MOVE_COUNTER = 1 # Starts at 1 and is incremented after Black's move

def initialize_bitboards(board: chess.Board):
    """
    Initializes the bitboards based on the current state of a python-chess board.
    """
    global WHITE_PAWNS, WHITE_KNIGHTS, WHITE_BISHOPS, WHITE_ROOKS, WHITE_QUEENS, WHITE_KINGS
    global BLACK_PAWNS, BLACK_KNIGHTS, BLACK_BISHOPS, BLACK_ROOKS, BLACK_QUEENS, BLACK_KINGS
    global WHITE_PIECES, BLACK_PIECES, ALL_PIECES
    global SIDE_TO_MOVE, CASTLING_RIGHTS, EN_PASSANT_SQUARE, HALF_MOVE_COUNTER, FULL_MOVE_COUNTER

    # Use correct python-chess accessors
    white_occupied = board.occupied_co[chess.WHITE]
    black_occupied = board.occupied_co[chess.BLACK]

    WHITE_PAWNS = board.pawns & white_occupied
    WHITE_KNIGHTS = board.knights & white_occupied
    WHITE_BISHOPS = board.bishops & white_occupied
    WHITE_ROOKS = board.rooks & white_occupied
    WHITE_QUEENS = board.queens & white_occupied
    WHITE_KINGS = board.kings & white_occupied

    BLACK_PAWNS = board.pawns & black_occupied
    BLACK_KNIGHTS = board.knights & black_occupied
    BLACK_BISHOPS = board.bishops & black_occupied
    BLACK_ROOKS = board.rooks & black_occupied
    BLACK_QUEENS = board.queens & black_occupied
    BLACK_KINGS = board.kings & black_occupied

    WHITE_PIECES = white_occupied
    BLACK_PIECES = black_occupied
    ALL_PIECES = board.occupied

    SIDE_TO_MOVE = board.turn
    CASTLING_RIGHTS = board.castling_rights
    EN_PASSANT_SQUARE = board.ep_square
    HALF_MOVE_COUNTER = board.halfmove_clock
    FULL_MOVE_COUNTER = board.fullmove_number

def print_bitboard(bitboard):
    """
    Prints a bitboard in a human-readable 8x8 format.
    """
    s = ""
    for i in range(63, -1, -1):
        if (bitboard >> i) & 1:
            s += "1 "
        else:
            s += "0 "
        if i % 8 == 0:
            s += "\n"
    print(s)

def get_king_square(side: chess.Color) -> chess.Square | None:
    """
    Returns the square index of the king for the given side.
    Returns None if the king is not found (which indicates an invalid state).
    """
    king_bb = WHITE_KINGS if side == chess.WHITE else BLACK_KINGS
    if king_bb == 0:
        return None # King not found on the board
    # Assuming there's exactly one king, find its square index
    return chess.lsb(king_bb)
