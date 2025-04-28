import chess
from nicegui import ui

# --- Game State ---
board: chess.Board = chess.Board()
selected_square: chess.Square | None = None
legal_moves_for_selected: list[chess.Square] = []
is_ai_thinking: bool = False
last_move: chess.Move | None = None
player_color: chess.Color = chess.WHITE
is_board_flipped: bool = False
highlight_checkmate_enabled: bool = True
is_checkmate_state: bool = False
engine_depth_limit: int = 6
engine_time_limit: float = 5.0

# --- Timer State ---
initial_time_minutes: int = 5
increment_seconds: int = 0
white_time_seconds: float = initial_time_minutes * 60.0
black_time_seconds: float = initial_time_minutes * 60.0
last_tick_timestamp: float | None = None # Timestamp of the last timer tick or move completion
game_timer: ui.timer | None = None
is_timer_running: bool = False # Explicit flag to control ticking

# --- UI Element Cache ---
# Using dictionaries for cells is fine. For single elements, direct assignment is okay.
cells: dict[chess.Square, tuple[ui.element, ui.html]] = {}
status_label: ui.label | None = None
move_history_container: ui.column | None = None
move_history_scroll_area: ui.scroll_area | None = None # Cache the scroll area itself
move_history_black_placeholders: dict[int, ui.label] = {} # Cache for black move labels
ai_thinking_overlay: ui.element | None = None
white_timer_label: ui.label | None = None
black_timer_label: ui.label | None = None
# --- Theme State ---
theme_toggle_button: ui.button | None = None # Cache for theme button element