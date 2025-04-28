import chess
import sys
import os
from nicegui import run, ui, app

script_dir = os.path.dirname(os.path.abspath(__file__))
algo_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(algo_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
if algo_dir not in sys.path:
    sys.path.append(algo_dir)

# Import modularized UI components
from . import state
from . import config
from . import timer
from . import board
from . import ai_handler
from . import game_logic

try:
    from algo.board_state import initialize_board_state
    ENGINE_INIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import 'initialize_board_state': {e}. Engine state might not be pre-initialized.")
    ENGINE_INIT_AVAILABLE = False
    def initialize_board_state(*args, **kwargs): pass # Dummy


# --- Theme Handling ---
def apply_theme(theme_name: str | None = None, update_board: bool = True):
    """Applies the theme by setting config, updating body class, toggling button icon, and optionally updating the board."""
    if theme_name is None:
        theme_name = "dark" if config.get_current_theme_name() == "light" else "light"

    config.set_theme(theme_name)
    is_dark = theme_name == "dark"
    icon = 'dark_mode' if is_dark else 'light_mode'

    # Update body class via JS
    ui.run_javascript(f'''
        document.body.classList.remove('light-theme', 'dark-theme');
        document.body.classList.add('{theme_name}-theme');
    ''')

    # Update button icon if it exists
    if state.theme_toggle_button:
        state.theme_toggle_button.props(f"icon={icon}")

    if update_board:
        board.update_board_display()
    # Update theme-dependent elements like timer label classes
    if state.white_timer_label and state.black_timer_label:
        state.white_timer_label.classes(remove='timer-white timer-black', add='timer-white')
        state.black_timer_label.classes(remove='timer-white timer-black', add='timer-black')


# --- Connect Handler ---
async def on_page_connect():
    """ Schedules initial UI updates and applies initial theme class after connection. """
    print("Page connected, scheduling initial draw and applying theme...")
    # Apply initial theme (set update_board=False to prevent redundant update)
    apply_theme(config.get_current_theme_name(), update_board=False)
    # Schedule other non-board updates
    ui.timer(0.15, timer.update_timer_display, once=True)
    ui.timer(0.2, lambda: game_logic.check_game_over(check_flag_fall=False), once=True)
    print("Initial updates scheduled and theme applied.")


# --- Main UI Definition ---
@ui.page('/')
def build_ui():
    """Builds the main user interface for the chess application."""
    print("build_ui: Entering function...")
    # Clear caches on page build/reload
    state.cells.clear()
    state.move_history_black_placeholders.clear()
    state.theme_toggle_button = None

    # --- CSS Styling with Theming ---
    light_colors = config.light_theme
    dark_colors = config.dark_theme

    ui.add_head_html(f'''
    <style>
        /* Global variables */
        :root {{
            --cell-size: {config.CELL_SIZE}px;
            --board-border-color: rgba(0, 0, 0, 0.2);
            --board-border-width: 2px;
        }}

        /* Light Theme */
        .light-theme {{
            --bg-main: {light_colors["BACKGROUND_COLOR"]};
            --panel-bg: rgba(240, 240, 240, 0.85);
            --text-color: {light_colors["TEXT_COLOR"]};
            --text-muted: #555555;
            --accent: #007bff;
            --light-square: {light_colors["LIGHT_SQUARE_COLOR"]};
            --dark-square: {light_colors["DARK_SQUARE_COLOR"]};
            --highlight: {light_colors["HIGHLIGHT_COLOR_CSS"]};
            --legal-move-dot: {light_colors["LEGAL_MOVE_HIGHLIGHT_COLOR_CSS"]};
            --last-move-from: {light_colors["LAST_MOVE_FROM_COLOR_CSS"]};
            --last-move-to: {light_colors["LAST_MOVE_TO_COLOR_CSS"]};
            --check-highlight: {light_colors["CHECK_HIGHLIGHT_COLOR_CSS"]};
            --checkmate-highlight: {light_colors["CHECKMATE_HIGHLIGHT_COLOR_CSS"]};
            --checkmate-shadow: {light_colors["CHECKMATE_SHADOW_CSS"]};
            --timer-bg-w: #E0E0E0;
            --timer-text-w: #000000;
            --timer-bg-b: #707070;
            --timer-text-b: #FFFFFF;
            --card-border: rgba(0, 0, 0, 0.1);
            --scroll-bg: rgba(0, 0, 0, 0.05);
            --board-border-color: rgba(0, 0, 0, 0.3);
            --theme-button-color: #444444;
        }}

        /* Dark Theme */
        .dark-theme {{
            --bg-main: {dark_colors["BACKGROUND_COLOR"]};
            --panel-bg: rgba(46, 46, 62, 0.8);
            --text-color: {dark_colors["TEXT_COLOR"]};
            --text-muted: #AAAAAA;
            --accent: #00E5FF;
            --light-square: {dark_colors["LIGHT_SQUARE_COLOR"]};
            --dark-square: {dark_colors["DARK_SQUARE_COLOR"]};
            --highlight: {dark_colors["HIGHLIGHT_COLOR_CSS"]};
            --legal-move-dot: {dark_colors["LEGAL_MOVE_HIGHLIGHT_COLOR_CSS"]};
            --last-move-from: {dark_colors["LAST_MOVE_FROM_COLOR_CSS"]};
            --last-move-to: {dark_colors["LAST_MOVE_TO_COLOR_CSS"]};
            --check-highlight: {dark_colors["CHECK_HIGHLIGHT_COLOR_CSS"]};
            --checkmate-highlight: {dark_colors["CHECKMATE_HIGHLIGHT_COLOR_CSS"]};
            --checkmate-shadow: {dark_colors["CHECKMATE_SHADOW_CSS"]};
            --timer-bg-w: #F5F5F5;
            --timer-text-w: #000000;
            --timer-bg-b: #505050;
            --timer-text-b: #FFFFFF;
            --card-border: rgba(255, 255, 255, 0.1);
            --scroll-bg: rgba(255, 255, 255, 0.05);
            --board-border-color: rgba(255, 255, 255, 0.3);
            --theme-button-color: #FFFFFF;
        }}

        /* General Styles using Variables */
        body {{
            background-color: var(--bg-main);
            color: var(--text-color);
            transition: background-color 0.3s ease, color 0.3s ease;
            margin: 0; padding: 0;
        }}
        .highlighted {{
            outline: 3px solid var(--highlight);
            outline-offset: -3px; border-radius: 2px;
        }}
        .legal-move::after {{
            content: ''; position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%); width: 18px; height: 18px;
            background-color: var(--legal-move-dot); border-radius: 50%;
            pointer-events: none; box-shadow: 0 0 5px rgba(0,0,0,0.3);
        }}
        .board-cell {{
            width: var(--cell-size); height: var(--cell-size);
            display: flex; justify-content: center; align-items: center;
            position: relative; cursor: pointer;
            transition: background-color 0.1s ease;
            box-sizing: border-box;
        }}
        .board-cell:hover {{ background-color: rgba(128, 128, 128, 0.2) !important; }}
        .piece-svg-container {{
            width: 85%; height: 85%;
            display: flex; justify-content: center; align-items: center;
            pointer-events: none;
        }}
        .piece-svg-container > svg {{ object-fit: contain; }}
        .ai-overlay {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            display: flex; justify-content: center; align-items: center;
            z-index: 10; cursor: wait; backdrop-filter: blur(4px);
            border-radius: inherit;
            color: white;
        }}
        .last-move-from {{ background-color: var(--last-move-from) !important; }}
        .last-move-to {{ background-color: var(--last-move-to) !important; }}
        .check-highlight {{ background-color: var(--check-highlight) !important; border-radius: 50%; }}
        .checkmate-highlight {{
             background-color: var(--checkmate-highlight) !important; border-radius: 50%;
             box-shadow: 0 0 10px 3px var(--checkmate-shadow);
        }}
        /* Component Styles using Variables */
        .board-card {{
            border: var(--board-border-width) solid var(--board-border-color);
            transition: border-color 0.3s ease;
        }}
        .q-card {{
            background: var(--panel-bg); backdrop-filter: blur(5px);
            border: 1px solid var(--card-border);
            color: var(--text-color);
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
        }}
        .timer-label {{
             padding: 6px 10px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
             transition: background-color 0.3s ease, color 0.3s ease;
        }}
        .timer-white {{ background-color: var(--timer-bg-w); color: var(--timer-text-w); }}
        .timer-black {{ background-color: var(--timer-bg-b); color: var(--timer-text-b); }}

        .history-scroll-area {{
             border: 1px solid var(--card-border);
             border-radius: 4px;
             background-color: var(--scroll-bg);
             transition: border-color 0.3s ease, background-color 0.3s ease;
        }}
        .history-move-number {{ color: var(--text-muted); transition: color 0.3s ease; }}

        /* Adjust input fields etc for theme */
        .q-field--outlined .q-field__control {{ border-color: var(--card-border); transition: border-color 0.3s ease; }}
        .q-field__native, .q-field__label {{ color: var(--text-color); transition: color 0.3s ease; }}
        .q-radio, .q-switch {{ color: var(--accent); }}
        .q-btn--flat {{ color: var(--accent); }}
        .q-separator {{ background-color: var(--card-border); transition: background-color 0.3s ease; }}
        .q-tooltip {{ background-color: #333; color: white; }}

        .theme-toggle-button .q-icon {{
             color: var(--theme-button-color) !important;
             transition: color 0.3s ease;
        }}

        /* Quasar overrides */
        .q-field--dense .q-field__control {{ height: 40px !important; }}
        .q-field--dense .q-field__label {{ top: 10px !important; }}

    </style>
    ''')

    # --- Theme Toggle Button ---
    with ui.page_sticky(position='top-right', x_offset=20, y_offset=20):
        initial_icon = 'dark_mode' if config.get_current_theme_name() == "dark" else 'light_mode'
        state.theme_toggle_button = ui.button(icon=initial_icon, on_click=lambda: apply_theme()) \
            .props('flat round dense') \
            .classes('theme-toggle-button') \
            .tooltip('Toggle Light/Dark Theme')


    # --- Main Layout Row ---
    with ui.row().classes('w-full justify-center items-start q-pa-md pt-20'):

        # --- Board Column ---
        with ui.column().classes('items-center'):
            with ui.element('div').classes('relative'): # Overlay container
                # AI Thinking Overlay
                with ui.element('div').classes('ai-overlay items-center justify-center').style('display: none;') as overlay:
                     state.ai_thinking_overlay = overlay
                     ui.spinner(size='xl', color='white')

                # Board Grid in a Card with border
                with ui.card().tight().classes('shadow-5 rounded-borders board-card'):
                    with ui.grid(columns=8).classes('gap-0'):
                        for r in range(8):
                            for c in range(8):
                                initial_square_idx = chess.square(c, 7 - r)
                                is_light = (r + c) % 2 != 0
                                initial_color = config.light_theme["LIGHT_SQUARE_COLOR"] if is_light else config.light_theme["DARK_SQUARE_COLOR"]
                                # Create cell div
                                with ui.element('div').classes('board-cell').style(f'background-color: {initial_color};') \
                                    .on('click', lambda row=r, col=c: board.on_cell_click(row, col)) as cell_div:
                                    # Create HTML element for piece SVG
                                    piece_html = ui.html().classes('piece-svg-container')
                                # Store elements in cache
                                state.cells[initial_square_idx] = (cell_div, piece_html)

        # --- Sidebar Column ---
        with ui.column().classes('ml-4 w-72'):

            # Status Label
            state.status_label = ui.label("Initializing...").classes('text-h6 mb-2 text-center w-full')

            # Timers Row
            with ui.row().classes('w-full justify-between mt-1 mb-3'):
                 state.white_timer_label = ui.label("W: 00:00").classes('text-h6 font-mono timer-label timer-white')
                 state.black_timer_label = ui.label("B: 00:00").classes('text-h6 font-mono timer-label timer-black')

            # History Card
            with ui.card().classes('q-pa-sm q-mb-md w-full'):
                ui.label("Move History").classes('text-subtitle1 mb-1')
                state.move_history_scroll_area = ui.scroll_area().classes('w-full h-64 history-scroll-area')
                with state.move_history_scroll_area:
                     state.move_history_container = ui.column().classes('w-full gap-0 q-pa-xs')

            # Controls Card
            with ui.card().classes('q-pa-sm q-mb-md w-full'):
                ui.label("Controls").classes('text-subtitle1 mb-1 text-center')
                with ui.row().classes('w-full justify-around'):
                    ui.button(icon='refresh', on_click=game_logic.on_new_game).tooltip("New Game").props('flat round color=primary')
                    ui.button(icon='undo', on_click=game_logic.on_undo).tooltip("Undo Move").props('flat round color=secondary')
                    ui.button(icon='swap_horiz', on_click=game_logic.on_flip_board).tooltip("Flip Board").props('flat round color=secondary')

            # Settings Card
            with ui.card().classes('q-pa-sm q-mb-md w-full'):
                ui.label("Settings").classes('text-subtitle1 mb-2 text-center')

                # Player Color Setting
                with ui.row().classes('items-center w-full justify-between'):
                     ui.label("Play as:").classes('q-mr-sm')
                     ui.radio({chess.WHITE: 'White', chess.BLACK: 'Black'},
                              value=state.player_color,
                              on_change=lambda e: setattr(state, 'player_color', e.value)) \
                        .props('inline dense')

                ui.separator().classes('q-my-sm')

                # AI Settings
                with ui.row().classes('items-center w-full justify-between'):
                     ui.label("AI:").classes('q-mr-sm')
                     with ui.row().classes('items-center'):
                         ui.number("Depth", value=state.engine_depth_limit, min=1, max=10, step=1, format='%d',
                                   on_change=lambda e: setattr(state, 'engine_depth_limit', int(e.value))) \
                             .props('dense outlined style="width: 75px;"').tooltip("AI Search Depth")
                         ui.number("Time(s)", value=state.engine_time_limit, min=0.1, max=60.0, step=0.5, format='%.1f',
                                   on_change=lambda e: setattr(state, 'engine_time_limit', float(e.value))) \
                             .props('dense outlined style="width: 75px;"').tooltip("AI Max Thinking Time (seconds)")

                ui.separator().classes('q-my-sm')

                # Time Control Settings
                with ui.row().classes('items-center w-full justify-between'):
                     ui.label("Time:").classes('q-mr-sm')
                     with ui.row().classes('items-center'):
                         ui.number("Mins", value=state.initial_time_minutes, min=1, max=60, step=1, format='%d',
                                   on_change=lambda e: setattr(state, 'initial_time_minutes', int(e.value))) \
                             .props('dense outlined style="width: 75px;"').tooltip("Initial Minutes per Player")
                         ui.number("Inc(s)", value=state.increment_seconds, min=0, max=30, step=1, format='%d',
                                   on_change=lambda e: setattr(state, 'increment_seconds', int(e.value))) \
                             .props('dense outlined style="width: 75px;"').tooltip("Increment in Seconds per Move")

                ui.separator().classes('q-my-sm')

                # Highlight Checkmate Setting
                with ui.row().classes('items-center w-full justify-between'):
                    ui.label("Highlight Checkmate:")
                    ui.switch(value=state.highlight_checkmate_enabled,
                                on_change=lambda e: setattr(state, 'highlight_checkmate_enabled', e.value)).props('dense color=primary')

    # --- Initial Board Draw ---
    # Schedule the first board update with a small delay after UI build
    print("build_ui: Scheduling initial board draw...")
    ui.timer(0.05, board.update_board_display, once=True) # Small delay (50ms)
    print("build_ui: Initial board draw scheduled.")


# --- Run the App ---
if __name__ in {"__main__", "__mp_main__"}:
    # Ensure the global timer object is created before the app starts running
    if state.game_timer is None:
        state.game_timer = ui.timer(0.1, timer.tick_timer, active=False)
        print("Game Timer Created (inactive).")

    # Register the connect handler
    app.on_connect(on_page_connect)

    if ENGINE_INIT_AVAILABLE:
        try:
            initialize_board_state(state.board)
            print("Engine state initialized with board.")
        except TypeError as e:
            if "positional argument: 'board'" in str(e) or "unexpected keyword argument 'board'" in str(e):
                 try:
                     initialize_board_state()
                     print("Engine state initialized (fallback without board).")
                 except Exception as fallback_e:
                     print(f"Warning: Fallback engine state initialization failed: {fallback_e}")
            else:
                 print(f"Warning: Failed to initialize engine state: {e}")
        except Exception as e:
            print(f"Warning: Failed to initialize engine state: {e}")

    # Start the NiceGUI app
    ui.run(title="NiceGUI Chess AI",
           reload=False,
           favicon='♘',
           uvicorn_reload_excludes='.py',
           )