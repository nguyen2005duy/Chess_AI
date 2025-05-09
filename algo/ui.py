import chess
import chess.svg
import threading
import base64
import sys
import os
from nicegui import run, ui, app # Import run

# Add algo package to sys.path if running ui.py directly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import engine components
try:
    from algo.search import find_best_move
    from algo.board_state import initialize_board_state
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import engine components: {e}. AI opponent will not work.")
    ENGINE_AVAILABLE = False
    def find_best_move(*args, **kwargs): return None
    def initialize_board_state(*args, **kwargs): pass # Dummy

# --- Constants ---
CELL_SIZE = 70
BOARD_BG_COLOR = "#2E2E2E"
HIGHLIGHT_COLOR_CSS = "yellow"
LEGAL_MOVE_HIGHLIGHT_COLOR_CSS = "rgba(31, 119, 180, 0.4)"
LIGHT_SQUARE_COLOR = "#EEEED2"
DARK_SQUARE_COLOR = "#769656"

# --- Pre-render SVGs (Raw strings for ui.html with added size attributes) ---
_piece_svgs = {}
for piece in [chess.Piece(pt, col) for pt in chess.PIECE_TYPES for col in chess.COLORS]:
    svg_string = chess.svg.piece(piece)
    if svg_string.startswith('<svg '):
        svg_string = svg_string.replace('<svg ', '<svg width="100%" height="100%" ', 1)
    _piece_svgs[piece.symbol()] = svg_string

# --- Game State ---
board = chess.Board()
selected_square = None
legal_moves_for_selected = []
is_ai_thinking = False
player_color = chess.WHITE
is_board_flipped = False

# --- UI Element Cache ---
cells = {}
status_label = None
move_history_container = None
ai_thinking_overlay = None

# --- Helper Functions ---
def get_square_from_row_col(r, c):
    """ Convert NiceGUI grid row/col (r=0 top, c=0 left) to chess.Square index. """
    rank = 7 - r if not is_board_flipped else r
    file = c     if not is_board_flipped else 7 - c
    return chess.square(file, rank)

# --- UI Building and Logic ---

def update_board_display():
    """ Updates the NiceGUI board using the cell cache and ui.html for pieces. """
    global cells, selected_square, legal_moves_for_selected
    if not cells or len(cells) != 64: return

    for square, (cell, piece_html) in cells.items():
        if cell is None or piece_html is None: continue

        piece = board.piece_at(square)
        symbol = piece.symbol() if piece else None
        svg_content = _piece_svgs.get(symbol, '')

        piece_html.set_content(svg_content)
        cell.tooltip(f"{chess.piece_name(piece.piece_type).capitalize()} on {chess.square_name(square)}" if piece else chess.square_name(square))

        cell.classes(remove='highlighted legal-move') # Pass string
        if square == selected_square:
            cell.classes(add='highlighted')
        elif square in legal_moves_for_selected:
            cell.classes(add='legal-move')

def on_cell_click(square):
    """ Handles clicks on board cells. """
    global selected_square, legal_moves_for_selected, board

    print(f"Clicked: {chess.square_name(square)}")

    if board.is_game_over():
        ui.notify("Game is over.", type='info')
        return
    # Allow click only if it's the player's turn AND AI is not thinking
    if board.turn != player_color or is_ai_thinking:
         ui.notify("Not your turn or AI is thinking.", type='warning')
         return

    piece_on_clicked = board.piece_at(square)

    if selected_square is None:
        if piece_on_clicked and piece_on_clicked.color == player_color:
            selected_square = square
            legal_moves_for_selected = [m.to_square for m in board.legal_moves if m.from_square == selected_square]
        else:
             legal_moves_for_selected = []
    else:
        is_legal_target = square in legal_moves_for_selected
        if is_legal_target:
            promo_piece = None
            piece_type = board.piece_type_at(selected_square)
            if piece_type == chess.PAWN:
                if (player_color == chess.WHITE and chess.square_rank(square) == 7) or \
                   (player_color == chess.BLACK and chess.square_rank(square) == 0):
                    promo_piece = chess.QUEEN # TODO: Promotion UI

            move = chess.Move(selected_square, square, promotion=promo_piece)
            if move in board.legal_moves:
                san_move = board.san(move)
                board.push(move) # Push Player move
                record_move_history(san_move) # Record Player move
                selected_square = None
                legal_moves_for_selected = []
                update_board_display()
                if not check_game_over():
                    # Check if AI should play
                    if board.turn != player_color:
                         ui.timer(0.05, trigger_ai_move, once=True) # Slightly longer delay before AI
            else:
                 ui.notify("Illegal move.", type='negative')
                 selected_square = None
                 legal_moves_for_selected = []
        elif piece_on_clicked and piece_on_clicked.color == player_color:
             selected_square = square
             legal_moves_for_selected = [m.to_square for m in board.legal_moves if m.from_square == selected_square]
        else:
            selected_square = None
            legal_moves_for_selected = []

    update_board_display() # Update highlights regardless

async def trigger_ai_move():
    """ Starts the AI search, showing overlay. """
    global is_ai_thinking, ai_thinking_overlay
    # Double check conditions before starting
    if not ENGINE_AVAILABLE or is_ai_thinking or board.turn == player_color or board.is_game_over():
        return

    is_ai_thinking = True
    if status_label: status_label.set_text("AI Thinking...")
    if ai_thinking_overlay: ai_thinking_overlay.set_visibility(True)
    print("Triggering AI move...")

    thread_board = board.copy()
    engine_time_limit = 10.0 # TODO: Configurable
    engine_depth_limit = 4  # TODO: Configurable

    try:
        print(f"Running AI search with depth={engine_depth_limit}, time_limit={engine_time_limit}...")
        ai_move = await run.cpu_bound(
             find_best_move, thread_board, engine_depth_limit, engine_time_limit
        )
        print(f"AI finished, move: {ai_move.uci() if ai_move else 'None'}")
        # Use a timer to schedule the result handler in the main event loop
        ui.timer(0.01, lambda move=ai_move: handle_ai_result(move), once=True)
    except Exception as e:
        print(f"Error during AI search execution: {e}")
        ui.timer(0.01, handle_ai_failure, once=True)

def handle_ai_result(ai_move):
    """ Handles AI result, hiding overlay. """
    global is_ai_thinking, board, ai_thinking_overlay
    if ai_move is None:
        handle_ai_failure() # This also hides overlay
        return

    if board.is_game_over():
         is_ai_thinking = False # Reset flag even if game ended while thinking
         if status_label: status_label.set_text("Game Over") # Update status if needed
         if ai_thinking_overlay: ai_thinking_overlay.set_visibility(False)
         return

    try:
        # Check legality in the *current* board state (in case something changed)
        if board.is_legal(ai_move):
            san_move = board.san(ai_move)
            board.push(ai_move) # Push AI move
            record_move_history(san_move) # Record AI move
            update_board_display()
            check_game_over() # Check game state *after* AI move
        else:
             # This case might happen in rare race conditions or if AI logic is flawed
             print(f"Error: AI returned illegal move {ai_move.uci()} in current context")
             if status_label: status_label.set_text("Error: AI illegal move")
             ui.notify("AI returned an illegal move!", type='negative')
    except Exception as e:
        print(f"Error applying AI move {ai_move.uci() if ai_move else 'None'}: {e}")
        if status_label: status_label.set_text("Error applying AI move")
        ui.notify("Error applying AI move.", type='negative')
    finally:
        # Ensure thinking flag and overlay are reset regardless of outcome
        is_ai_thinking = False
        if ai_thinking_overlay: ai_thinking_overlay.set_visibility(False)

def handle_ai_failure():
    """ Handles AI failure (exception or no move found), hiding overlay. """
    global is_ai_thinking, ai_thinking_overlay
    print("AI failed or returned no move.")
    if status_label: status_label.set_text("Error: AI failed")
    ui.notify("AI engine failed.", type='negative')
    is_ai_thinking = False
    if ai_thinking_overlay: ai_thinking_overlay.set_visibility(False)

def check_game_over():
    """ Checks game status and updates label. Returns True if over. """
    global status_label
    if not status_label: return False

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        outcome_text = ""
        if board.is_checkmate(): winner = "Black" if board.turn == chess.WHITE else "White"; outcome_text = f"Checkmate! {winner} wins."
        elif board.is_stalemate(): outcome_text = "Stalemate."
        elif board.is_insufficient_material(): outcome_text = "Draw by insufficient material."
        elif board.is_seventyfive_moves(): outcome_text = "Draw by 75-move rule."
        elif board.is_fivefold_repetition(): outcome_text = "Draw by fivefold repetition."
        elif board.can_claim_draw(): outcome_text = "Draw can be claimed."
        else: outcome_text = f"Game Over: {result}"
        status_label.set_text(outcome_text)
        print(outcome_text)
        return True
    else:
        turn_text = "White's Turn" if board.turn == chess.WHITE else "Black's Turn"
        # Update status based on current turn, considering AI thinking state
        if is_ai_thinking: turn_text = "AI Thinking..."
        elif board.turn != player_color: turn_text = "AI's Turn" # Show AI's turn briefly
        status_label.set_text(turn_text)
        return False

def record_move_history(san_move):
    """ Adds the move in SAN format to the move history UI. """
    global move_history_container
    if not move_history_container: return

    # Determine whose move was just made based on the *new* board.turn
    is_white_move_recorded = board.turn == chess.BLACK
    move_number = board.fullmove_number

    with move_history_container:
        if is_white_move_recorded:
            # White just moved, start a new row
            row_id = f'm{move_number}'
            with ui.row().classes('w-full no-wrap items-center').props(f'id="{row_id}"'):
                 ui.label(f"{move_number}.").classes('w-8 text-right mr-1')
                 ui.label(san_move).classes('w-16 font-mono font-bold text-black') # Style White's move
                 ui.label("...").classes('w-16 font-mono').props(f'id="{row_id}_b"') # Placeholder for Black
        else:
            # Black just moved, update the placeholder in the last row
            # Use the *current* move number as the row ID
            row_id = f'm{move_number}'
            js_code = f'''
                var el = getElement("{row_id}_b");
                if (el) {{
                    el.innerText = "{san_move}";
                    el.classList.add("font-bold", "text-black"); // Style Black's move
                }} else {{
                    console.warn("RecordHistory: Could not find element {row_id}_b to update Black's move.");
                }}
            '''
            ui.run_javascript(js_code)
            print(f"Attempted JS update for black move placeholder {row_id}_b.")


    # Scroll history to bottom
    js_scroll = f'''
        var element = document.getElementById("{move_history_container.id}");
        if (element) {{ element.scrollTop = element.scrollHeight; }}
    '''
    ui.run_javascript(js_scroll)

# CORRECTED on_undo function
def on_undo():
    """
    Undoes the last player move.
    If the AI played last, it undoes the AI move AND the preceding player move.
    """
    global board, selected_square, legal_moves_for_selected, move_history_container
    if is_ai_thinking:
        ui.notify("Cannot undo while AI is thinking.", type='warning')
        return

    moves_to_pop = 0
    try:
        if board.move_stack:
            # Check whose turn it is *before* popping. If it's player's turn, AI moved last.
            ai_played_last = (board.turn == player_color)

            if ai_played_last and len(board.move_stack) >= 1:
                 board.pop() # Pop AI move
                 moves_to_pop += 1

            # Always pop at least the last move (player's move or the one before AI's)
            if len(board.move_stack) >= 1:
                 board.pop() # Pop player's move
                 moves_to_pop += 1
            else: # Should not happen if logic is correct, but handle edge case
                moves_to_pop = 0 # Nothing could be popped

            print(f"Undo: Popped {moves_to_pop} half-moves.")

            # --- UPDATE UI HISTORY ---
            if move_history_container and moves_to_pop > 0:
                try:
                    children = move_history_container.default_slot.children
                    num_rows = len(children)
                    if num_rows > 0:
                        if moves_to_pop == 2: # Undid AI and Player move
                            # Remove the last full row
                            move_history_container.remove(children[-1])
                            print("Removed last row from history UI (undid player + AI).")
                        elif moves_to_pop == 1: # Undid only player move
                            # Reset the black move placeholder in the last row
                            last_row = children[-1]
                            if hasattr(last_row, 'default_slot') and last_row.default_slot.children:
                                row_children = last_row.default_slot.children
                                if len(row_children) >= 3 and isinstance(row_children[-1], ui.label):
                                    black_move_label = row_children[-1]
                                    black_move_label.set_text("...")
                                    black_move_label.classes(remove='font-bold text-black') # Reset style
                                    print("Reset black move placeholder (undid player only).")
                                else: print("Warning: Last history row structure unexpected.")
                            else: print("Warning: Last history element is not a row container.")
                except Exception as ui_e:
                    print(f"Error updating move history UI during undo: {ui_e}")
            # --- END UI HISTORY UPDATE ---

            # Reset selection and update board/status
            selected_square = None
            legal_moves_for_selected = []
            update_board_display()
            check_game_over() # Update status label
            if status_label and not board.is_game_over(): status_label.set_text("Undo successful")

        else:
             ui.notify("No moves to undo.", type='info')
    except IndexError: ui.notify("Error during undo.", type='negative'); print("Error: Less moves on stack than expected.")
    except Exception as e: ui.notify("Unexpected error during undo.", type='negative'); print(f"Unexpected error: {e}")

def on_flip_board():
    """ Toggles the board orientation and updates display. """
    global is_board_flipped
    is_board_flipped = not is_board_flipped
    print(f"Board flipped: {is_board_flipped}")
    # Display update handles rendering based on board state correctly
    update_board_display()


def on_new_game():
    """ Resets the game state and UI. """
    global board, selected_square, legal_moves_for_selected, is_ai_thinking, move_history_container
    if is_ai_thinking:
        ui.notify("AI is thinking, please wait...", type='warning')
        return
    is_ai_thinking = False # Ensure flag is reset
    board.reset()
    selected_square = None
    legal_moves_for_selected = []
    if move_history_container: move_history_container.clear()
    if status_label: status_label.set_text("New Game - White's Turn")
    print("New game started.")
    update_board_display()
    # Trigger AI if it's AI's turn (e.g., player chose Black)
    if board.turn != player_color and not board.is_game_over():
         ui.timer(0.1, trigger_ai_move, once=True)
         
# --- Connect Handler for Initialization ---
async def on_page_connect():
    """ Ensure initial board state is drawn after UI elements are ready. """
    print("Page connected, performing initial draw...")
    ui.timer(0.1, update_board_display, once=True)
    ui.timer(0.15, check_game_over, once=True)

# --- Main UI Definition ---
@ui.page('/')
def build_ui():
    global move_history_container, status_label, cells, ai_thinking_overlay
    cells.clear()

    ui.add_head_html(f'''
    <style>
          :root {{
              --bg-main: #1A1A2E;
              --panel-bg: rgba(26,26,46,0.6);
              --accent: #00E5FF;
              --light-square: #2E2E3E;
              --dark-square: #1A1A2E;
              --highlight: var(--accent);
            }}
        .highlighted {{ outline: 3px solid {HIGHLIGHT_COLOR_CSS}; outline-offset: -3px; border-radius: 2px; }}
        .legal-move::after {{
            content: ''; position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%); width: 20px; height: 20px;
            background-color: {LEGAL_MOVE_HIGHLIGHT_COLOR_CSS}; border-radius: 50%;
            pointer-events: none;
        }}
        .board-cell {{
            width: {CELL_SIZE}px; height: {CELL_SIZE}px;
            display: flex; justify-content: center; align-items: center;
            position: relative; cursor: pointer;
        }}
        .board-cell:hover {{ transform: scale(1.01); }}
        .piece-svg-container {{
             width: 85%; height: 85%; display: flex;
             justify-content: center; align-items: center; pointer-events: none;
        }}
        .piece-svg-container > svg {{ object-fit: contain; }}
        .ai-overlay {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.1); display: flex;
            justify-content: center; align-items: center; z-index: 10; cursor: wait;
        }}
        .sidebar-card {{
           background: var(--panel-bg);
           border-radius: 12px;
           padding: 16px;
           backdrop-filter: blur(8px);
         }}
    </style>
    ''')

    with ui.row().classes('w-full justify-center items-start q-pa-md'):
        with ui.column():
            with ui.element('div').classes('relative'):
                with ui.element('div').classes('ai-overlay').style('display: none;') as overlay_div: pass
                ai_thinking_overlay = overlay_div

                with ui.card().tight().style(f'background-color: {BOARD_BG_COLOR}; padding: 10px;'):
                    with ui.grid(columns=8).classes('gap-0'):
                        for r in range(8):
                            for c in range(8):
                                file, rank = c, 7 - r
                                square = chess.square(file, rank)
                                is_light = bool(chess.BB_SQUARES[square] & chess.BB_LIGHT_SQUARES)
                                color = LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR

                                with ui.element('div').classes('board-cell').style(f'background-color: {color};') \
                                    .on('click', lambda sq=square: on_cell_click(sq)) as cell_div:
                                    piece_html = ui.html().classes('piece-svg-container')
                                    cells[square] = (cell_div, piece_html)

        # Sidebar Area
        with ui.column().classes('ml-4 w-64'):
            status_label = ui.label("Initializing...").classes('text-h6 mb-2')
            ui.label("Move History").classes('text-subtitle1 mb-1')
            with ui.scroll_area().classes('w-full h-64 border rounded q-pa-xs'):
                 move_history_container = ui.column().classes('w-full gap-0')
            with ui.row().classes('w-full justify-between mt-2'):
                 ui.label("W: 00:00").classes('text-body1 font-mono')
                 ui.label("B: 00:00").classes('text-body1 font-mono')
            ui.label("Controls").classes('text-subtitle1 mt-4 mb-1')
            with ui.row().classes('w-full'):
                ui.button("New Game", on_click=on_new_game, icon='refresh')
                ui.button("Undo", on_click=on_undo, icon='undo') # Calls the corrected on_undo
            with ui.row().classes('w-full'):
                ui.button("Flip Board", on_click=on_flip_board, icon='swap_horiz')

# --- Run the App ---
if __name__ in {"__main__", "__mp_main__"}:
    app.on_connect(on_page_connect)
    if ENGINE_AVAILABLE:
        try:
            initialize_board_state(board)
            print("Engine state initialized.")
        except TypeError as te:
             if "positional argument: 'board'" in str(te):
                 try: initialize_board_state(); print("Engine state initialized (without board arg).")
                 except Exception as e_inner: print(f"Warning: Failed to initialize board state (fallback): {e_inner}")
             else: print(f"Warning: Failed to initialize board state: {te}")
        except Exception as e: print(f"Warning: Failed to initialize board state: {e}")

    ui.run(title="NiceGUI Chess AI", reload=False, favicon='♘')

