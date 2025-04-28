import chess
import chess.svg

# --- Constants ---
CELL_SIZE = 80

# --- Theme Definitions ---
light_theme = {
    "HIGHLIGHT_COLOR_CSS": "yellow",
    "LEGAL_MOVE_HIGHLIGHT_COLOR_CSS": "rgba(31, 119, 180, 0.4)", # Slightly opaque blue
    "LIGHT_SQUARE_COLOR": "#EEEED2", # Creamy white
    "DARK_SQUARE_COLOR": "#769656", # Muted green
    "LAST_MOVE_FROM_COLOR_CSS": "rgba(255, 255, 0, 0.3)", # Transparent yellow
    "LAST_MOVE_TO_COLOR_CSS": "rgba(255, 255, 0, 0.5)",   # Slightly more opaque yellow
    "CHECK_HIGHLIGHT_COLOR_CSS": "rgba(255, 0, 0, 0.6)",    # Semi-transparent red
    "CHECKMATE_HIGHLIGHT_COLOR_CSS": "rgba(139, 0, 0, 0.8)", # Darker, more opaque red
    "CHECKMATE_SHADOW_CSS": "darkred",                  # Shadow for checkmate highlight
    "TEXT_COLOR": "#000000", # Black text for light mode
    "BACKGROUND_COLOR": "#FFFFFF", # White background for light mode
}

dark_theme = {
    "HIGHLIGHT_COLOR_CSS": "gold", # Brighter yellow for dark mode
    "LEGAL_MOVE_HIGHLIGHT_COLOR_CSS": "rgba(100, 149, 237, 0.5)", # Lighter blue
    "LIGHT_SQUARE_COLOR": "#B3B3B3", # Greyish light squares
    "DARK_SQUARE_COLOR": "#666666", # Darker grey squares
    "LAST_MOVE_FROM_COLOR_CSS": "rgba(255, 215, 0, 0.4)", # Gold highlight (transparent)
    "LAST_MOVE_TO_COLOR_CSS": "rgba(255, 215, 0, 0.6)",   # Gold highlight (more opaque)
    "CHECK_HIGHLIGHT_COLOR_CSS": "rgba(255, 69, 0, 0.7)",    # Orangey-red
    "CHECKMATE_HIGHLIGHT_COLOR_CSS": "rgba(255, 0, 0, 0.8)", # Brighter red
    "CHECKMATE_SHADOW_CSS": "red",                      # Shadow for checkmate highlight
    "TEXT_COLOR": "#FFFFFF", # White text for dark mode
    "BACKGROUND_COLOR": "#2E2E2E", # Dark grey background for dark mode
}

_current_theme_name = "light"
_themes = {"light": light_theme, "dark": dark_theme}

def set_theme(theme_name: str):
    """Sets the current theme."""
    global _current_theme_name
    if theme_name in _themes:
        _current_theme_name = theme_name
        print(f"Theme set to: {theme_name}")
    else:
        print(f"Warning: Theme '{theme_name}' not found. Keeping '{_current_theme_name}'.")

def get_theme_colors() -> dict[str, str]:
    """Returns the dictionary of colors for the current theme."""
    return _themes[_current_theme_name]

def get_current_theme_name() -> str:
    """Returns the name of the current theme."""
    return _current_theme_name

# --- Pre-render SVGs ---
_piece_svgs: dict[str, str] = {}

def _render_svgs():
    """Pre-renders piece SVGs for faster loading."""
    global _piece_svgs
    if _piece_svgs: # Avoid re-rendering if already done
        return

    print("Pre-rendering piece SVGs...")
    temp_svgs = {}
    for piece in [chess.Piece(pt, col) for pt in chess.PIECE_TYPES for col in chess.COLORS]:
        svg_string = chess.svg.piece(piece)
        # Ensure SVG scales correctly within its container by adding width/height attributes
        if svg_string and svg_string.startswith('<svg '):
            svg_string = svg_string.replace('<svg ', '<svg width="100%" height="100%" ', 1)
        temp_svgs[piece.symbol()] = svg_string
    _piece_svgs = temp_svgs
    print("Finished pre-rendering SVGs.")

def get_piece_svg(symbol: str | None) -> str:
    """Returns the pre-rendered SVG for a piece symbol, or an empty string."""
    if not _piece_svgs:
        _render_svgs() # Render on first access if needed
    return _piece_svgs.get(symbol, '') if symbol else ''

# Initial render call
_render_svgs()