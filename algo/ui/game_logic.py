import chess
from nicegui import ui, run
import time # Added import

# Import shared state and other UI modules
from . import state
from . import timer
from . import board # Import board module
from .ai_handler import ENGINE_AVAILABLE # Keep ENGINE_AVAILABLE import
from .ai_handler import find_best_move # Import the actual engine function interface

# --- Core Game Logic Functions ---

def check_game_over(check_flag_fall: bool = False) -> bool:
    """Checks if the game has ended and updates the status label. Returns True if game over."""
    if not state.status_label: return False # Cannot update status if label doesn't exist

    game_over_reason = None
    current_is_checkmate = False

    # Check python-chess game termination states
    if state.board.is_checkmate():
        game_over_reason = "checkmate"
        current_is_checkmate = True
    elif state.board.is_stalemate():
        game_over_reason = "stalemate"
    elif state.board.is_insufficient_material():
        game_over_reason = "insufficient_material"
    elif state.board.is_seventyfive_moves():
        game_over_reason = "seventyfive_moves"
    elif state.board.is_fivefold_repetition():
        game_over_reason = "fivefold_repetition"

    # Check explicit flag fall if requested
    timed_out = False
    if not game_over_reason and check_flag_fall and (state.white_time_seconds <= 0 or state.black_time_seconds <= 0):
        timed_out = True
        game_over_reason = "timeout"
        current_is_checkmate = False # Timeout overrides checkmate status display

    game_over = game_over_reason is not None

    if game_over:
        timer.stop_timer() # Ensure timer is stopped

        outcome_text = ""
        result = state.board.result(claim_draw=False) # Get result without claiming draw

        if game_over_reason == "checkmate":
            winner = "Black" if state.board.turn == chess.WHITE else "White"
            outcome_text = f"Checkmate! {winner} wins. ({result})"
        elif game_over_reason == "timeout":
            winner_on_time = 'Black' if state.white_time_seconds <= 0 else 'White'
            result = "1-0" if winner_on_time == "White" else "0-1"
            outcome_text = f"Time out! {winner_on_time} wins. ({result})"
        elif game_over_reason == "stalemate":
            outcome_text = f"Stalemate. ({result})"
        elif game_over_reason == "insufficient_material":
            outcome_text = f"Draw: Insufficient Material. ({result})"
        elif game_over_reason == "seventyfive_moves":
            outcome_text = f"Draw: 75 Move Rule. ({result})"
        elif game_over_reason == "fivefold_repetition":
            outcome_text = f"Draw: Repetition. ({result})"
        else: # Fallback
            outcome_text = f"Game Over: {result}"

        state.status_label.set_text(outcome_text)
        print(outcome_text)
    else:
        # Game not over - update turn status
        if state.is_ai_thinking:
            turn_text = "AI Thinking..."
        elif state.board.turn == state.player_color:
            turn_text = "Your Turn"
        else:
            turn_text = "AI's Turn"
        state.status_label.set_text(turn_text)

    # Update global checkmate flag for highlighting
    state.is_checkmate_state = current_is_checkmate

    return game_over


def on_undo():
    """ Undoes the last full move (player + AI if applicable). """
    if state.is_ai_thinking:
        ui.notify("Cannot undo while AI is thinking.", type='warning')
        return

    timer.pause_timer() # Pause timer during undo

    moves_to_pop_count = 0
    try:
        if not state.board.move_stack:
            ui.notify("No moves to undo.", type='info')
            return

        # Determine how many half-moves to pop (1 or 2)
        num_to_pop = 0
        if state.board.turn == state.player_color: # AI played last
            num_to_pop = 2
        else: # Player played last
            num_to_pop = 1

        popped_moves = []
        for _ in range(num_to_pop):
            if state.board.move_stack:
                popped_moves.append(state.board.pop())
                moves_to_pop_count += 1
            else:
                break # Stop if stack becomes empty

        if moves_to_pop_count == 0:
            print("Undo Error: No moves were popped unexpectedly.")
            ui.notify("Undo failed.", type='negative')
            if not state.board.is_game_over(): timer.start_timer()
            return

        print(f"Undo: Popped {moves_to_pop_count} half-moves.")

        # --- UPDATE UI HISTORY ---
        # Simple approach: Clear visual history and placeholder cache.
        # Rebuilding can be slow and complex.
        if state.move_history_container:
             state.move_history_container.clear()
             state.move_history_black_placeholders.clear() # Clear placeholder cache
             print("Undo: Cleared visual move history and placeholder cache.")
             # Optional: Rebuild history here if needed.

        # --- Reset Game State ---
        state.selected_square = None
        state.legal_moves_for_selected = []
        state.last_move = state.board.peek() if state.board.move_stack else None
        state.is_checkmate_state = False # Cannot be checkmate after undo
        board.update_board_display() # Update board visuals
        game_is_over = check_game_over() # Update status label

        if state.status_label and not game_is_over:
            status_text = "Your Turn" if state.board.turn == state.player_color else "AI's Turn"
            state.status_label.set_text(f"Undo successful. {status_text}")

        # Resume timer or trigger AI
        if not game_is_over:
            if state.board.turn != state.player_color and ENGINE_AVAILABLE:
                print("Triggering AI move after undo...")
                # Note: AI move will now start its own timer when triggered
                ui.timer(0.05, trigger_ai_move, once=True) # Let AI move again
            else:
                print("Resuming timer after undo.")
                timer.start_timer() # Resume timer for player

    except IndexError:
        ui.notify("Error: Cannot undo further.", type='warning')
        if not state.board.is_game_over(): timer.start_timer()
    except Exception as e:
        ui.notify(f"Error during undo: {e}", type='negative')
        print(f"Unexpected error during undo: {e}")
        if not state.board.is_game_over(): timer.start_timer()


def on_flip_board():
    """Flips the board orientation and updates the display."""
    state.is_board_flipped = not state.is_board_flipped
    print(f"Board flipped: {state.is_board_flipped}")
    board.update_board_display() # Update display immediately


def on_new_game():
    """Resets the game state, timer, and UI for a new game."""
    if state.is_ai_thinking:
        ui.notify("AI is thinking, please wait before starting a new game.", type='warning')
        return

    timer.stop_timer() # Ensure timer is stopped

    # Reset Game Logic State
    state.is_ai_thinking = False
    state.board.reset()
    state.selected_square = None
    state.legal_moves_for_selected = []
    state.last_move = None
    state.is_checkmate_state = False

    # Reset UI State
    if state.move_history_container: state.move_history_container.clear()
    state.move_history_black_placeholders.clear() # Clear placeholder cache
    if state.ai_thinking_overlay: state.ai_thinking_overlay.set_visibility(False)

    # --- Initialize Timer State ---
    state.white_time_seconds = float(state.initial_time_minutes * 60)
    state.black_time_seconds = float(state.initial_time_minutes * 60)
    state.last_tick_timestamp = None
    state.is_timer_running = False
    timer.update_timer_display() # Show initial times

    # Ensure the timer object exists but is inactive
    if state.game_timer is None:
        state.game_timer = ui.timer(0.1, timer.tick_timer, active=False)
        print("Game Timer Created in on_new_game (inactive).")
    else:
        state.game_timer.deactivate()

    # Update Status Label
    initial_turn_text = "Your Turn" if state.player_color == chess.WHITE else "AI's Turn"
    if state.status_label: state.status_label.set_text(f"New Game - {initial_turn_text}")
    print(f"New game started. Player: {'White' if state.player_color == chess.WHITE else 'Black'}. Time: {state.initial_time_minutes}m + {state.increment_seconds}s.")

    # Update Board Display
    board.update_board_display()

    # --- Start Timer or Trigger AI ---
    if state.board.turn == chess.WHITE: # White always moves first
        if state.player_color == chess.WHITE:
            print("Player is White, starting timer.")
            timer.start_timer()
        elif ENGINE_AVAILABLE: # Player chose Black, AI is White
            print("Player chose Black, triggering initial AI move.")
            # AI move will start its own timer now
            ui.timer(0.1, trigger_ai_move, once=True)
        else: # Player chose Black, no AI
             print("Player is Black, AI unavailable. Waiting for White's move.")
    else:
        print("Warning: Board turn is not White after reset.")

# --- AI Move Trigger & Handling --- (Moved from ai_handler.py)

async def trigger_ai_move():
    """Initiates the AI move calculation in a separate thread, WITH the timer running."""
    # Prevent concurrent AI thinking or if it's not AI's turn
    if not ENGINE_AVAILABLE or state.is_ai_thinking or state.board.turn == state.player_color or state.board.is_game_over():
        return

    # timer.pause_timer() # REMOVED - Timer should run during AI thinking
    state.is_ai_thinking = True
    if state.status_label: state.status_label.set_text("AI Thinking...")
    if state.ai_thinking_overlay: state.ai_thinking_overlay.set_visibility(True)
    print("Triggering AI move...")

    # Copy board state for the thread
    thread_board = state.board.copy()
    current_depth = state.engine_depth_limit
    current_time_limit = state.engine_time_limit

    # Start the timer for the AI's turn just before starting the calculation
    timer.start_timer() # ADDED - Start timer for AI's turn

    try:
        print(f"Running AI search with depth={current_depth}, time_limit={current_time_limit}...")
        ai_move = await run.cpu_bound(
             find_best_move, thread_board, current_depth, current_time_limit
        )
        print(f"AI finished, move: {ai_move.uci() if ai_move else 'None'}")
        # Schedule result handling back in the main event loop
        ui.timer(0.01, lambda move=ai_move: handle_ai_result(move), once=True)
    except Exception as e:
        print(f"Error during AI search execution: {e}")
        # Ensure UI/timer state is reset even on search error
        state.is_ai_thinking = False
        if state.ai_thinking_overlay: state.ai_thinking_overlay.set_visibility(False)
        # Resume timer only if the game isn't over yet
        # REMOVED timer.start_timer() - Timer should be running unless game ended
        pass # Keep block structure if needed
        # Schedule failure handling in the main loop
        ui.timer(0.01, handle_ai_failure, once=True)


def handle_ai_result(ai_move: chess.Move | None):
    """Handles the result from the AI calculation thread."""
    # Reset thinking flag FIRST, regardless of outcome
    state.is_ai_thinking = False
    if state.ai_thinking_overlay: state.ai_thinking_overlay.set_visibility(False)

    if ai_move is None:
        # AI failed or error occurred
        handle_ai_failure() # Call local function now
        # If AI fails, timer might still be running - should it stop or continue?
        # For now, let it continue until game ends naturally or player moves.
        # if not state.board.is_game_over(): timer.start_timer() # Removed
        return

    # Check if game ended while AI was thinking
    if state.board.is_game_over():
         print("Game ended while AI was thinking. Ignoring AI move.")
         # Timer should have been stopped by check_game_over inside tick_timer
         return

    try:
        # Double-check legality
        if state.board.is_legal(ai_move):
             was_ai_turn_white = state.board.turn == chess.WHITE
             # Timer should be running (started in trigger_ai_move)

             san_move = state.board.san(ai_move)
             state.board.push(ai_move) # Make the AI move
             state.last_move = ai_move # Update last move state

             # Apply increment AFTER AI move
             if was_ai_turn_white: state.white_time_seconds += state.increment_seconds
             else: state.black_time_seconds += state.increment_seconds
             print(f"AI applied increment. W: {timer.format_time(state.white_time_seconds)}, B: {timer.format_time(state.black_time_seconds)}")

             board.record_move_history(san_move) # Use board module function
             timer.update_timer_display() # Use timer module function
             board.update_board_display() # Use board module function

             # Check game over BEFORE potentially restarting timer
             game_over = check_game_over() # Call local function now

             # Reset timestamp AFTER AI processing but BEFORE player's turn starts
             # This ensures the player's time starts accurately from this point.
             if not game_over and state.last_tick_timestamp is not None:
                  state.last_tick_timestamp = time.time()
                  print(f"DEBUG: handle_ai_result - Resetting tick timestamp for player's turn to {state.last_tick_timestamp:.2f}")

             if not game_over:
                 # Check if it's now the human player's turn
                 if state.board.turn == state.player_color:
                     # timer.start_timer() # REMOVED - Timer should still be running
                     print("DEBUG: handle_ai_result - Player's turn, timer continues.")
                 else:
                     print("Warning: AI vs AI detected? Timer continues.") # Timer continues for next AI
             else:
                 print("Game over after AI move.")
                 # Timer stopped by check_game_over

        else: # Illegal move from AI
             print(f"Error: AI returned illegal move {ai_move.uci()}")
             if state.status_label: state.status_label.set_text("Error: AI illegal move")
             ui.notify("AI returned an illegal move!", type='negative')
             # Let timer continue if game not over, as it remains AI's turn effectively
             # if not state.board.is_game_over(): timer.start_timer() # Removed

    except Exception as e:
        print(f"Error applying AI move {ai_move.uci()}: {e}")
        if state.status_label: state.status_label.set_text("Error applying AI move")
        ui.notify(f"Error applying AI move: {e}", type='negative')
        # Let timer continue if game not over
        # if not state.board.is_game_over(): timer.start_timer() # Removed


def handle_ai_failure():
    """Handles the case where the AI engine fails to produce a move."""
    # Note: Thinking flag/overlay reset handled by caller. Timer is assumed to be running.
    print("AI failed or returned no move.")
    if state.status_label: state.status_label.set_text("Error: AI failed")
    ui.notify("AI engine failed to find a move.", type='negative')