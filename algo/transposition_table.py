# Import only what's needed
from . import board_state # Import the board_state module to access CURRENT_ZOBRIST_HASH
from .evaluation import MATE_THRESHOLD # Need mate score constants
import collections # For OrderedDict

# --- Transposition Table Entry Flags ---
HASH_FLAG_EXACT = 0
HASH_FLAG_LOWERBOUND = 1 # Alpha (Fail-Low)
HASH_FLAG_UPPERBOUND = 2 # Beta (Fail-High)

# --- TT Size and Replacement Strategy Constants ---
TT_MAX_SIZE = 16000000  # Increased to 16 million entries
TT_GENERATION = 0       # Current search generation (incremented each new search)
TT_ALWAYS_REPLACE = 0   # Always replace this slot
TT_GEN_BONUS = 8        # Bonus for entries from current generation
TT_DEPTH_BIAS = 3       # How strongly to prefer deeper searches

class TTEntry:
    """ Represents an entry in the transposition table. """
    def __init__(self, key, score, depth, flags, best_move=None, generation=0):
        self.key = key # Store full Zobrist key for collision detection
        self.score = score # Score relative to the side whose turn it was *at this node*
        self.depth = depth # Remaining depth of the search that stored this entry
        self.flags = flags # HASH_FLAG_EXACT, HASH_FLAG_LOWERBOUND, or HASH_FLAG_UPPERBOUND
        self.best_move = best_move # Best move found from this position
        self.generation = generation # Search generation when this entry was stored

    def __repr__(self):
        flag_str = {0: "EXACT", 1: "ALPHA", 2: "BETA"}.get(self.flags, "?")
        move_str = self.best_move.uci() if self.best_move else "None"
        return f"TTEntry(key={self.key:016x}, score={self.score}, depth={self.depth}, flags={flag_str}, move={move_str}, gen={self.generation})"

# --- Transposition Table ---
transposition_table = collections.OrderedDict()  # Using OrderedDict for better replacement policy
tt_hits = 0
tt_probes = 0
tt_size = 0 # Track size for info

# --- TT Functions ---
def clear_tt():
    """ Clears the transposition table. """
    global transposition_table, tt_hits, tt_probes, tt_size, TT_GENERATION
    transposition_table.clear()
    tt_hits = 0
    tt_probes = 0
    tt_size = 0
    TT_GENERATION = 0

def new_search():
    """ Updates the generation counter for a new search. """
    global TT_GENERATION
    TT_GENERATION += 1

    # Prefer periodic trimming rather than full clearing
    # Keep entries from previous search but age them
    if TT_GENERATION >= 255:
        # Only keep entries from recent generations
        old_keys = [k for k, v in transposition_table.items() if v.generation < TT_GENERATION - 4]
        for key in old_keys:
            del transposition_table[key]
        TT_GENERATION = 5  # Reset after cleanup

def probe_tt(depth, ply, alpha, beta):
    """
    Probes the TT for a potentially useful entry using the current board state hash.
    Returns (score, best_move, found_and_used_flag)
    found_and_used_flag is True if a valid entry directly caused a cutoff or provided an exact score.
    Stored mate scores are adjusted based on the current ply.
    """
    global tt_probes, tt_hits
    tt_probes += 1

    current_hash = board_state.CURRENT_ZOBRIST_HASH
    entry = transposition_table.get(current_hash)

    if not entry:
        return None, None, False

    # Update generation to prevent early replacement
    entry.generation = TT_GENERATION

    # Move to end for LRU behavior in OrderedDict
    transposition_table.move_to_end(current_hash)

    # Allow shallower entries to provide moves, but more cautious about scores
    if entry.key == current_hash:
        # Always count the hit if the hash matches
        tt_hits += 1

        # Adjust stored mate scores based on ply
        stored_score = entry.score
        if abs(stored_score) > MATE_THRESHOLD:
            if stored_score > 0:
                stored_score -= ply  # We're closer to mate now
            else:
                stored_score += ply  # We're closer to being mated

        # If depth is sufficient, use the score according to flag type
        if entry.depth >= depth:
            if entry.flags == HASH_FLAG_EXACT:
                return stored_score, entry.best_move, True

            elif entry.flags == HASH_FLAG_LOWERBOUND and stored_score >= beta:
                return beta, entry.best_move, True

            elif entry.flags == HASH_FLAG_UPPERBOUND and stored_score <= alpha:
                return alpha, entry.best_move, True

        # Even with insufficient depth, try to use bound information
        # if the bounds are strong enough
        elif entry.depth >= depth - 2:  # Allow slightly shallower entries
            if entry.flags == HASH_FLAG_LOWERBOUND and stored_score >= beta + 100:
                # Strong lower bound even with less depth
                return beta, entry.best_move, True
            elif entry.flags == HASH_FLAG_UPPERBOUND and stored_score <= alpha - 100:
                # Strong upper bound even with less depth
                return alpha, entry.best_move, True

        # Return best move but not score
        return None, entry.best_move, False

    return None, None, False

def record_tt(depth, ply, score, flags, best_move=None):
    """
    Stores an entry in the TT using the current board state hash.
    Uses a replacement strategy that considers depth, flag type, and generation.
    """
    global tt_size
    current_hash = board_state.CURRENT_ZOBRIST_HASH

    # Adjust mate scores before storing
    score_to_store = score
    if abs(score) > MATE_THRESHOLD:
        if score > 0:
            score_to_store = score + ply  # We're further from mate
        else:
            score_to_store = score - ply  # We're further from being mated

    # Replacement strategy
    should_store = True
    existing_entry = transposition_table.get(current_hash)

    if existing_entry:
        # Advanced replacement strategy:
        # 1. Always replace if new search is deeper
        # 2. Prefer exact scores over bounds
        # 3. Prefer current generation entries
        # 4. Don't replace exact scores with bounds unless deeper

        current_is_exact = existing_entry.flags == HASH_FLAG_EXACT
        new_is_exact = flags == HASH_FLAG_EXACT

        # Calculate a score for both entries to decide which to keep
        existing_score = existing_entry.depth * TT_DEPTH_BIAS
        if existing_entry.generation == TT_GENERATION:
            existing_score += TT_GEN_BONUS
        if current_is_exact:
            existing_score += 2

        new_score = depth * TT_DEPTH_BIAS
        if new_is_exact:
            new_score += 2

        should_store = (new_score >= existing_score)

    if should_store:
        # Check for table size limit
        if tt_size >= TT_MAX_SIZE and current_hash not in transposition_table:
            # Use LRU replacement policy with OrderedDict
            if transposition_table:
                transposition_table.popitem(last=False)  # Remove oldest entry
                tt_size -= 1

        # Store new entry
        if current_hash not in transposition_table:
            tt_size += 1

        entry = TTEntry(
            current_hash,
            score_to_store,
            depth,
            flags,
            best_move,
            TT_GENERATION)

        transposition_table[current_hash] = entry
        # Move to end (most recently used position)
        if current_hash in transposition_table:
            transposition_table.move_to_end(current_hash)

def get_tt_stats():
    """ Returns TT stats as a dictionary. """
    return {
        "probes": tt_probes,
        "hits": tt_hits,
        "size": tt_size,
        "generation": TT_GENERATION,
        "hit_rate": (tt_hits / max(tt_probes, 1)) * 100
    }
