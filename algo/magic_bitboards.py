import chess
from typing import List
from . import bitboards

# Constants and Precomputed Data
KING_ATTACKS: List[chess.Bitboard] = [chess.BB_KING_ATTACKS[sq] for sq in chess.SQUARES]
KNIGHT_ATTACKS: List[chess.Bitboard] = [chess.BB_KNIGHT_ATTACKS[sq] for sq in chess.SQUARES]

PAWN_ATTACKS: List[List[chess.Bitboard]] = [[0] * 64 for _ in range(2)]


def _init_pawn_attacks():
    for sq in chess.SQUARES:
        r, f = chess.square_rank(sq), chess.square_file(sq)
        w = 0
        if r < 7:
            if f > 0: w |= chess.BB_SQUARES[sq + 7]
            if f < 7: w |= chess.BB_SQUARES[sq + 9]
        b = 0
        if r > 0:
            if f > 0: b |= chess.BB_SQUARES[sq - 9]
            if f < 7: b |= chess.BB_SQUARES[sq - 7]
        PAWN_ATTACKS[chess.WHITE][sq] = w
        PAWN_ATTACKS[chess.BLACK][sq] = b


_init_pawn_attacks()

ROOK_MAGICS: List[int] = [
    0xa8002c000108020, 0x6c00049b0002001, 0x100200010090040, 0x2480041000800801,
    0x280028004000800, 0x900410008040022, 0x280020001001080, 0x2880002041000080,
    0xa000800080400034, 0x4808020004000, 0x2290802004801000, 0x411000d00100020,
    0x402800800040080, 0xb000401004208, 0x2409000100040200, 0x1002100004082,
    0x22878001e24000, 0x1090810021004010, 0x801030040200012, 0x500808008001000,
    0xa08018014000880, 0x8000808004000200, 0x201008080010200, 0x801020000441091,
    0x800080204005, 0x1040200040100048, 0x120200402082, 0xd14880480100080,
    0x12040280080080, 0x100040080020080, 0x9020010080800200, 0x813241200148449,
    0x491604001800080, 0x100401000402001, 0x4820010021001040, 0x400402202000812,
    0x209009005000802, 0x810800601800400, 0x4301083214000150, 0x204026458e001401,
    0x40204000808000, 0x8001008040010020, 0x8410820820420010, 0x1003001000090020,
    0x804040008008080, 0x12000810020004, 0x1000100200040208, 0x430000a044020001,
    0x280009023410300, 0xe0100040002240, 0x200100401700, 0x2244100408008080,
    0x8000400801980, 0x2000810040200, 0x8010100228810400, 0x2000009044210200,
    0x4080008040102101, 0x40002080411D01, 0x2005524060000901, 0x502001008400422,
    0x489A000810200402, 0x1004400080A13, 0x4000011008020084, 0x26002114058042
]

BISHOP_MAGICS: List[int] = [
    0x89A1121896040240, 0x2004844802002010, 0x2068080051921000, 0x62880A0220200808,
    0x4042004000000, 0x100822020200011, 0xC00444222012000A, 0x28808801216001,
    0x400492088408100, 0x201C401040C0084, 0x840800910A0010, 0x82080240060,
    0x2000840504006000, 0x30010C4108405004, 0x1008005410080802, 0x8144042209100900,
    0x208081020014400, 0x4800201208CA00, 0xF18140408012008, 0x1004002802102001,
    0x841000820080811, 0x40200200A42008, 0x800054042000, 0x88010400410C9000,
    0x520040470104290, 0x1004040051500081, 0x2002081833080021, 0x400C00C010142,
    0x941408200C002000, 0x658810000806011, 0x188071040440A00, 0x4800404002011C00,
    0x104442040404200, 0x511080202091021, 0x4022401120400, 0x80C0040400080120,
    0x8040010040820802, 0x480810700020090, 0x102008E00040242, 0x809005202050100,
    0x8002024220104080, 0x431008804142000, 0x19001802081400, 0x200014208040080,
    0x3308082008200100, 0x41010500040C020, 0x4012020C04210308, 0x208220A202004080,
    0x111040120082000, 0x6803040141280A00, 0x2101004202410000, 0x8200000041108022,
    0x21082088000, 0x2410204010040, 0x40100400809000, 0x822088220820214,
    0x40808090012004, 0x910224040218C9, 0x402814422015008, 0x90014004842410,
    0x1000042304105, 0x10008830412A00, 0x2520081090008908, 0x40102000A0A60140
]

# Shift Counts (64 - relevant bits)
ROOK_SHIFTS = [
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52
]

BISHOP_SHIFTS = [
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
]

# Masks for relevant blocker bits (excludes board edges)
ROOK_MASKS = [0] * 64
BISHOP_MASKS = [0] * 64

# Attack Tables (Initialized later)
ROOK_ATTACKS: List[List[chess.Bitboard]] = []
BISHOP_ATTACKS: List[List[chess.Bitboard]] = []


def _get_rook_mask(sq: chess.Square) -> chess.Bitboard:
    """Generate INNER mask for relevant rook blockers (excludes edges)."""
    r, f = chess.square_rank(sq), chess.square_file(sq)
    mask = 0
    # North ray (excluding rank 7)
    for rank in range(r + 1, 7): mask |= chess.BB_SQUARES[chess.square(f, rank)]
    # South ray (excluding rank 0)
    for rank in range(r - 1, 0, -1): mask |= chess.BB_SQUARES[chess.square(f, rank)]
    # East ray (excluding file 7)
    for file in range(f + 1, 7): mask |= chess.BB_SQUARES[chess.square(file, r)]
    # West ray (excluding file 0)
    for file in range(f - 1, 0, -1): mask |= chess.BB_SQUARES[chess.square(file, r)]
    return mask


def _get_bishop_mask(sq: chess.Square) -> chess.Bitboard:
    """Generate INNER mask for relevant bishop blockers (excludes edges)."""
    r, f = chess.square_rank(sq), chess.square_file(sq)
    mask = 0
    # NE (exclude edges)
    nr, nf = r + 1, f + 1
    while nr < 7 and nf < 7: mask |= chess.BB_SQUARES[chess.square(nf, nr)]; nr += 1; nf += 1
    # NW (exclude edges)
    nr, nf = r + 1, f - 1
    while nr < 7 and nf > 0: mask |= chess.BB_SQUARES[chess.square(nf, nr)]; nr += 1; nf -= 1
    # SE (exclude edges)
    nr, nf = r - 1, f + 1
    while nr > 0 and nf < 7: mask |= chess.BB_SQUARES[chess.square(nf, nr)]; nr -= 1; nf += 1
    # SW (exclude edges)
    nr, nf = r - 1, f - 1
    while nr > 0 and nf > 0: mask |= chess.BB_SQUARES[chess.square(nf, nr)]; nr -= 1; nf -= 1
    return mask


def _calculate_attacks_slow(sq: chess.Square, blockers: chess.Bitboard, is_bishop: bool) -> chess.Bitboard:
    """
    Calculates sliding attacks from sq, stopping AT the first blocker in ANY direction.
    'blockers' is the set of pieces considered obstructions for this specific calculation.
    """
    attacks = 0
    r, f = chess.square_rank(sq), chess.square_file(sq)

    directions = []
    if is_bishop:
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonals
    else:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Orthogonals

    for dr, df in directions:
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:  # Check bounds
            target_sq = chess.square(nf, nr)
            target_bb = chess.BB_SQUARES[target_sq]
            attacks |= target_bb  # Add the square to attacks

            # If this square is a blocker, stop the ray AFTER adding it
            if blockers & target_bb:
                break

            nr += dr
            nf += df
    return attacks


def _populate_attacks(is_bishop: bool):
    """Fill the attack tables for either bishops or rooks."""
    masks = BISHOP_MASKS if is_bishop else ROOK_MASKS
    magics = BISHOP_MAGICS if is_bishop else ROOK_MAGICS
    shifts = BISHOP_SHIFTS if is_bishop else ROOK_SHIFTS
    table = BISHOP_ATTACKS if is_bishop else ROOK_ATTACKS

    for sq in chess.SQUARES:
        mask = masks[sq]
        shift = shifts[sq]
        attack_table_sq = table[sq]

        if mask is None:
            continue

        # Use carry-rippler trick to iterate through all subsets of the INNER mask
        subset = 0
        while True:
            attacks = _calculate_attacks_slow(sq, subset, is_bishop)

            # Compute index using magic multiplication based on the subset
            index = ((subset * magics[sq]) & 0xFFFFFFFFFFFFFFFF) >> shift

            if index >= len(attack_table_sq):
                pass
            elif attack_table_sq[index] == 0:
                attack_table_sq[index] = attacks
            elif attack_table_sq[index] != attacks:
                raise RuntimeError(f"Magic collision detected at square {chess.square_name(sq)}")
            # Next subset
            subset = (subset - mask) & mask
            if subset == 0:
                break


def _init():
    """Initialize the magic bitboard tables."""
    global ROOK_ATTACKS, BISHOP_ATTACKS
    ROOK_ATTACKS = [[0] * (1 << (64 - ROOK_SHIFTS[sq])) for sq in chess.SQUARES]
    BISHOP_ATTACKS = [[0] * (1 << (64 - BISHOP_SHIFTS[sq])) for sq in chess.SQUARES]

    for sq in chess.SQUARES:
        ROOK_MASKS[sq] = _get_rook_mask(sq)
        BISHOP_MASKS[sq] = _get_bishop_mask(sq)

    _populate_attacks(False)  # Rook
    _populate_attacks(True)  # Bishop


# Run initialization on module import
_init()


def get_rook_attack(sq: chess.Square, occupied: chess.Bitboard) -> chess.Bitboard:
    """Gets rook attacks for a square given occupied squares using magic bitboards."""
    mask = ROOK_MASKS[sq]
    if mask is None:
        return 0
    relevant_blockers = occupied & mask  # Use only inner blockers for lookup
    index = ((relevant_blockers * ROOK_MAGICS[sq]) & 0xFFFFFFFFFFFFFFFF) >> ROOK_SHIFTS[sq]
    attack_table_sq = ROOK_ATTACKS[sq]
    if index >= len(attack_table_sq):
        return 0
    return attack_table_sq[index]


def get_bishop_attack(sq: chess.Square, occupied: chess.Bitboard) -> chess.Bitboard:
    """Gets bishop attacks for a square given occupied squares using magic bitboards."""
    mask = BISHOP_MASKS[sq]
    if mask is None:
        return 0
    relevant_blockers = occupied & mask  # Use only inner blockers for lookup
    index = ((relevant_blockers * BISHOP_MAGICS[sq]) & 0xFFFFFFFFFFFFFFFF) >> BISHOP_SHIFTS[sq]
    attack_table_sq = BISHOP_ATTACKS[sq]
    if index >= len(attack_table_sq):
        return 0
    return attack_table_sq[index]


def is_square_attacked(square: chess.Square, attacker_side: chess.Color) -> bool:
    """
    Checks if a square is attacked by the given side using bitboard logic
    and magic lookups. Relies on the global 'bitboards' module state.
    """
    occupied = bitboards.ALL_PIECES

    if attacker_side == chess.WHITE:
        attacker_pawns = bitboards.WHITE_PAWNS
        attacker_knights = bitboards.WHITE_KNIGHTS
        attacker_bishops = bitboards.WHITE_BISHOPS
        attacker_rooks = bitboards.WHITE_ROOKS
        attacker_queens = bitboards.WHITE_QUEENS
        attacker_kings = bitboards.WHITE_KINGS
    else:
        attacker_pawns = bitboards.BLACK_PAWNS
        attacker_knights = bitboards.BLACK_KNIGHTS
        attacker_bishops = bitboards.BLACK_BISHOPS
        attacker_rooks = bitboards.BLACK_ROOKS
        attacker_queens = bitboards.BLACK_QUEENS
        attacker_kings = bitboards.BLACK_KINGS

    defender_side = not attacker_side
    if PAWN_ATTACKS[defender_side][square] & attacker_pawns:
        return True
    if KNIGHT_ATTACKS[square] & attacker_knights:
        return True
    if KING_ATTACKS[square] & attacker_kings:
        return True
    if get_bishop_attack(square, occupied) & (attacker_bishops | attacker_queens):
        return True
    if get_rook_attack(square, occupied) & (attacker_rooks | attacker_queens):
        return True

    return False
