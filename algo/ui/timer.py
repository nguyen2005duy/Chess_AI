import time
import chess
from nicegui import ui

# Import shared state variables
from . import state
# Import necessary functions from the same package
from .game_logic import check_game_over # Relative import

# --- Timer Functions ---
def format_time(seconds: float) -> str:
    """Formats seconds into MM:SS format."""
    if seconds < 0: seconds = 0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def update_timer_display():
    """Updates the timer labels in the UI using values from the state module."""
    if state.white_timer_label: state.white_timer_label.set_text(f"W: {format_time(state.white_time_seconds)}")
    if state.black_timer_label: state.black_timer_label.set_text(f"B: {format_time(state.black_time_seconds)}")

def start_timer():
    """Starts or resumes the game timer if it's not already running and the game isn't over."""
    if not state.is_timer_running and not state.board.is_game_over():
        state.last_tick_timestamp = time.time()
        state.is_timer_running = True
        print(f"DEBUG: start_timer() - Setting is_timer_running=True, timestamp={state.last_tick_timestamp:.2f}")
        if state.game_timer:
            state.game_timer.activate()
            print("DEBUG: start_timer() - NiceGUI timer activated.")
        else:
            # This case should ideally not happen if initialized correctly
            print("Warning: Game timer not found when trying to start.")
            # Create and activate as fallback - consider if this is desired behavior
            # state.game_timer = ui.timer(0.1, tick_timer, active=True)

def pause_timer():
    """Pauses the game timer and updates the time elapsed since the last tick."""
    if state.is_timer_running:
        state.is_timer_running = False
        print(f"DEBUG: pause_timer() - Setting is_timer_running=False. Previous timestamp: {state.last_tick_timestamp}")
        # Record any elapsed time since the last tick before pausing fully
        if state.last_tick_timestamp:
            current_time = time.time()
            elapsed = current_time - state.last_tick_timestamp
            if state.board.turn == chess.WHITE:
                state.white_time_seconds -= elapsed
            else:
                state.black_time_seconds -= elapsed
            state.last_tick_timestamp = None # Clear timestamp, effective pause
            update_timer_display() # Show deduction immediately
        if state.game_timer:
            state.game_timer.deactivate() # Stop further ticking calls
            print("DEBUG: pause_timer() - NiceGUI timer deactivated.")
        # print("Timer Paused") # Replaced by DEBUG log

def stop_timer():
    """Stops the game timer completely (e.g., for game over or new game)."""
    state.is_timer_running = False
    state.last_tick_timestamp = None
    print("DEBUG: stop_timer() - Setting is_timer_running=False, timestamp=None.")
    if state.game_timer:
        state.game_timer.deactivate()
        print("DEBUG: stop_timer() - NiceGUI timer deactivated.")
    # print("Timer Stopped") # Replaced by DEBUG log

def tick_timer():
    """Called periodically by ui.timer to update player clocks."""
    # Only run if timer is active, initialized, and game not over
    if not state.is_timer_running or state.last_tick_timestamp is None or state.board.is_game_over():
        return

    current_time = time.time()
    elapsed = current_time - state.last_tick_timestamp # Time since last tick/start/resume

    # Subtract time from the player whose turn it is
    if state.board.turn == chess.WHITE:
        state.white_time_seconds -= elapsed
    else:
        state.black_time_seconds -= elapsed

    state.last_tick_timestamp = current_time # Update timestamp for next tick

    # Check for flag fall
    if state.white_time_seconds <= 0 or state.black_time_seconds <= 0:
        state.white_time_seconds = max(0.0, state.white_time_seconds) # Ensure non-negative
        state.black_time_seconds = max(0.0, state.black_time_seconds) # Ensure non-negative
        update_timer_display()
        winner_on_time = 'Black' if state.white_time_seconds <= 0 else 'White'
        print(f"Time out! {winner_on_time} wins on time.")
        stop_timer()
        # check_game_over will update status and handle game end state
        check_game_over(check_flag_fall=True)
    else:
        update_timer_display()