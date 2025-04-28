import chess
import chess.svg
from nicegui import ui

# Import shared state, config, and other UI modules
from . import state
from . import config
from . import timer
# Import game_logic module
from . import game_logic

# --- Board Display & Interaction Functions ---

def get_square_from_row_col(r: int, c: int) -> chess.Square:
    """Calculates the chess square index based on row, column, and board flip state."""
    rank = 7 - r if not state.is_board_flipped else r
    file = c     if not state.is_board_flipped else 7 - c
    return chess.square(file, rank)

def update_board_display():
    """Updates the visual representation of the chessboard based on the current game state."""
    if not state.cells or len(state.cells) != 64:
        print("Warning: Cells cache not ready for board update.")
        return

    king_to_check = None
    in_check = False
    # Use state.board consistently
    if state.board.is_check() and not state.is_checkmate_state:
        in_check = True
        king_to_check = state.board.king(state.board.turn)

    last_move_from = state.last_move.from_square if state.last_move else None
    last_move_to = state.last_move.to_square if state.last_move else None

    mated_king_square = None
    if state.is_checkmate_state:
        # The king in checkmate is the one whose turn it *was*
        mated_king_square = state.board.king(not state.board.turn)

    # Iterate through visual rows and columns to update each cell
    for r in range(8):
        for c in range(8):
            # 1. Calculate the standard UI square (A1 = bottom-left, 0-63) for accessing the UI cache
            # This assumes state.cells is always keyed 0-63 corresponding to A1-H8.
            ui_rank = 7 - r
            ui_file = c
            ui_square = chess.square(ui_file, ui_rank) # Key for state.cells

            # 2. Calculate the logical square based on visual r, c and flip state
            # This determines which piece *should* be on this visual square.
            logical_square = get_square_from_row_col(r, c) # Use existing helper

            # 3. Retrieve the correct UI cell elements using the standard ui_square key
            if ui_square not in state.cells:
                print(f"DEBUG: UI Square {chess.square_name(ui_square)} (r={r}, c={c}) not found in state.cells cache!")
                continue # Skip if UI element doesn't exist (shouldn't happen in normal operation)
            cell, piece_html = state.cells[ui_square] # Use ui_square key here!
            if cell is None or piece_html is None:
                print(f"DEBUG: Found UI key {chess.square_name(ui_square)} but cell or piece_html is None.")
                continue # Skip if elements are somehow None

            # 4. Get the piece from the logical board using the logical_square
            piece = state.board.piece_at(logical_square) # Use logical_square for board state
            symbol = piece.symbol() if piece else None
            svg_content = config.get_piece_svg(symbol)

            # 5. Set the SVG content of the correct piece_html UI element
            piece_html.set_content(svg_content) # Update the content of the correct UI element

            # --- Update Tooltip and Styling for the correct 'cell' based on the 'logical_square' ---
            tooltip_text = f"{chess.piece_name(piece.piece_type).capitalize()} on {chess.square_name(logical_square)}" if piece else chess.square_name(logical_square)
            # Use standard HTML title attribute for tooltip to avoid duplication
            cell.props(f'title="{tooltip_text}"', remove='title') # Ensure previous title is removed before setting new one

            # Reset classes before applying new ones (on the correct cell)
            cell.classes(remove='highlighted legal-move last-move-from last-move-to check-highlight checkmate-highlight')

            # Apply classes based on current game state, comparing against the logical_square
            if logical_square == state.selected_square: cell.classes(add='highlighted')
            elif logical_square in state.legal_moves_for_selected: cell.classes(add='legal-move')
            if logical_square == last_move_from: cell.classes(add='last-move-from')
            if logical_square == last_move_to: cell.classes(add='last-move-to')

            # Highlight king in check or checkmate
            if state.is_checkmate_state and state.highlight_checkmate_enabled and logical_square == mated_king_square:
                cell.classes(add='checkmate-highlight')
            elif in_check and logical_square == king_to_check:
                 cell.classes(add='check-highlight')

            # Update cell background color based on its standard grid position (ui_square)
            is_light = (chess.square_rank(ui_square) + chess.square_file(ui_square)) % 2 != 0
            colors = config.get_theme_colors() # Get current theme colors
            color = colors["LIGHT_SQUARE_COLOR"] if is_light else colors["DARK_SQUARE_COLOR"]
            cell.style(f'background-color: {color};')


async def prompt_for_promotion() -> chess.PieceType | None:
    """Displays a dialog for the user to choose a promotion piece."""
    piece_map = {'Queen': chess.QUEEN, 'Rook': chess.ROOK, 'Bishop': chess.BISHOP, 'Knight': chess.KNIGHT}
    result = None
    with ui.dialog() as dialog, ui.card():
        ui.label('Promote Pawn To:')
        with ui.row():
            for name, piece_type in piece_map.items():
                ui.button(name, on_click=lambda pt=piece_type: dialog.submit(pt))
    result = await dialog
    print(f"Promotion dialog result: {result}")
    return result if result in piece_map.values() else None


async def on_cell_click(r: int, c: int): # Accept row and col instead of square
    """ Handles clicks on board cells for piece selection and moves. """
    # Determine actual square based on current flip state
    square = get_square_from_row_col(r, c)
    print(f"Click detected on r={r}, c={c}. Flipped={state.is_board_flipped}. Calculated Square: {chess.square_name(square)}") # Debugging

    # --- Input Validation ---
    if state.board.is_game_over():
        ui.notify("Game is over.", type='info'); return
    if state.is_ai_thinking:
        ui.notify("AI is thinking.", type='warning'); return
    if state.board.turn != state.player_color:
        ui.notify("Not your turn.", type='warning'); return

    clicked_sq_name = chess.square_name(square)
    piece_on_clicked = state.board.piece_at(square)
    current_selection = state.selected_square # Cache selection state before modification

    print(f"Click: {clicked_sq_name}. Current Sel: {chess.square_name(current_selection) if current_selection is not None else 'None'}")

    # --- Logic ---
    if current_selection is None:
        # Case 1: No piece selected
        if piece_on_clicked and piece_on_clicked.color == state.player_color:
            state.selected_square = square
            state.legal_moves_for_selected = [m.to_square for m in state.board.legal_moves if m.from_square == state.selected_square]
            print(f"Selected {clicked_sq_name}. Legal targets: {[chess.square_name(s) for s in state.legal_moves_for_selected]}")
        else:
            state.legal_moves_for_selected = [] # Ensure list is empty
    else:
        # Case 2: Piece selected
        move = None
        is_legal_target = square in state.legal_moves_for_selected

        if is_legal_target:
            # Subcase 2a: Clicked legal target -> Attempt Move
            print(f"Attempting move {chess.square_name(current_selection)}->{clicked_sq_name}")
            promo_piece = None
            piece_type = state.board.piece_type_at(current_selection)
            is_pawn_promo = (piece_type == chess.PAWN and chess.square_rank(square) in [0, 7])

            if is_pawn_promo:
                promo_piece = await prompt_for_promotion()
                if promo_piece is None: # User cancelled
                    print("Promotion cancelled.")
                    state.selected_square = None
                    state.legal_moves_for_selected = []
                    update_board_display() # Update UI
                    return # Exit handler

            move = chess.Move(current_selection, square, promotion=promo_piece)

            if move in state.board.legal_moves:
                was_white_turn = state.board.turn == chess.WHITE
                timer.pause_timer() # Use timer module function

                san_move = state.board.san(move)
                state.board.push(move)
                state.last_move = move

                # Apply increment AFTER move
                if was_white_turn: state.white_time_seconds += state.increment_seconds
                else: state.black_time_seconds += state.increment_seconds
                print(f"Applied increment. W: {timer.format_time(state.white_time_seconds)}, B: {timer.format_time(state.black_time_seconds)}")

                record_move_history(san_move)
                timer.update_timer_display()

                # Reset selection state AFTER move
                state.selected_square = None
                state.legal_moves_for_selected = []

                game_over = game_logic.check_game_over() # Use game_logic function

                if not game_over:
                    if state.board.turn != state.player_color: # AI's turn
                        print("Triggering AI move...")
                        ui.timer(0.05, game_logic.trigger_ai_move, once=True) # Use game_logic function
                    else: # Human vs Human (or error?)
                        print("Resuming timer for player.")
                        timer.start_timer() # Resume timer for the same player
                else:
                    print("Game over after player move.")
                    # Timer stopped by check_game_over

            else: # Should not happen
                print(f"ERROR: Move {move.uci()} invalid despite being in legal targets!")
                ui.notify("Illegal move detected (internal error).", type='negative')
                state.selected_square = None
                state.legal_moves_for_selected = []

        elif square == current_selection:
            # Subcase 2b: Clicked same piece -> Deselect
            print(f"Clicked selected piece {clicked_sq_name} again. Deselecting.")
            state.selected_square = None
            state.legal_moves_for_selected = []
        elif piece_on_clicked and piece_on_clicked.color == state.player_color:
            # Subcase 2c: Clicked another friendly piece -> Change Selection
            print(f"Changed selection from {chess.square_name(current_selection)} to {clicked_sq_name}")
            state.selected_square = square
            state.legal_moves_for_selected = [m.to_square for m in state.board.legal_moves if m.from_square == state.selected_square]
        else:
            # Subcase 2d: Clicked irrelevant square -> Deselect
            print(f"Clicked irrelevant square {clicked_sq_name}. Deselecting.")
            state.selected_square = None
            state.legal_moves_for_selected = []

    # --- Update UI ---
    update_board_display()
    print(f"End click handler. New selection: {chess.square_name(state.selected_square) if state.selected_square is not None else 'None'}")


def record_move_history(san_move: str):
    """Adds the latest move in Standard Algebraic Notation to the move history UI."""
    if not state.move_history_container: return

    # Determine if White or Black just moved (turn has already flipped in `state.board`)
    is_white_move_recorded = state.board.turn == chess.BLACK
    move_number = state.board.fullmove_number

    with state.move_history_container:
        if is_white_move_recorded:
            # White just moved, start a new row
            with ui.row().classes('w-full no-wrap items-center q-py-xs'):
                 ui.label(f"{move_number}.").classes('w-8 text-right mr-1 text-grey') # Move number
                 ui.label(san_move).classes('w-16 font-mono font-bold') # White's move
                 # Placeholder for Black's move, cache it
                 placeholder = ui.label("...").classes('w-16 font-mono')
                 state.move_history_black_placeholders[move_number] = placeholder
        else:
            # Black just moved, update the placeholder
            placeholder_label = state.move_history_black_placeholders.get(move_number)
            if placeholder_label:
                placeholder_label.set_text(san_move)
                placeholder_label.classes(add='font-bold') # Make Black's move bold
            else:
                print(f"RecordHistory: Could not find placeholder label for move {move_number}")

    # Scroll history to bottom using the scroll area element
    def scroll_history():
        if state.move_history_scroll_area:
            state.move_history_scroll_area.scroll_to(percent=1.0)
        else:
            print("Warning: Move history scroll area not available for scrolling.")

    ui.timer(0.05, scroll_history, once=True) 