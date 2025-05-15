import chess

# Define bitboards for each piece type and color
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
CASTLING_RIGHTS = 0
EN_PASSANT_SQUARE = None
HALF_MOVE_COUNTER = 0
FULL_MOVE_COUNTER = 1


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
