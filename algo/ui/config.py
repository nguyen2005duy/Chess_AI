import chess
import chess.svg

# --- Constants ---
CELL_SIZE = 80

# --- Theme Definitions ---
light_theme = {
    "HIGHLIGHT_COLOR_CSS": "yellow",
    "LEGAL_MOVE_HIGHLIGHT_COLOR_CSS": "rgba(31, 119, 180, 0.4)",
    "LIGHT_SQUARE_COLOR": "#EEEED2",
    "DARK_SQUARE_COLOR": "#769656",
    "LAST_MOVE_FROM_COLOR_CSS": "rgba(255, 255, 0, 0.3)",
    "LAST_MOVE_TO_COLOR_CSS": "rgba(255, 255, 0, 0.5)",
    "CHECK_HIGHLIGHT_COLOR_CSS": "rgba(255, 0, 0, 0.6)",
    "CHECKMATE_HIGHLIGHT_COLOR_CSS": "rgba(139, 0, 0, 0.8)",
    "CHECKMATE_SHADOW_CSS": "darkred",
    "TEXT_COLOR": "#000000",
    "BACKGROUND_COLOR": "#FFFFFF",
}

dark_theme = {
    "HIGHLIGHT_COLOR_CSS": "gold",
    "LEGAL_MOVE_HIGHLIGHT_COLOR_CSS": "rgba(100, 149, 237, 0.5)",
    "LIGHT_SQUARE_COLOR": "#B3B3B3",
    "DARK_SQUARE_COLOR": "#666666",
    "LAST_MOVE_FROM_COLOR_CSS": "rgba(255, 215, 0, 0.4)",
    "LAST_MOVE_TO_COLOR_CSS": "rgba(255, 215, 0, 0.6)",
    "CHECK_HIGHLIGHT_COLOR_CSS": "rgba(255, 69, 0, 0.7)",
    "CHECKMATE_HIGHLIGHT_COLOR_CSS": "rgba(255, 0, 0, 0.8)",
    "CHECKMATE_SHADOW_CSS": "red",
    "TEXT_COLOR": "#FFFFFF",
    "BACKGROUND_COLOR": "#2E2E2E",
}

_current_theme_name = "light"
_themes = {"light": light_theme, "dark": dark_theme}

def set_theme(theme_name: str):
    global _current_theme_name
    if theme_name in _themes:
        _current_theme_name = theme_name
        print(f"Theme set to: {theme_name}")
    else:
        print(f"Warning: Theme '{theme_name}' not found. Keeping '{_current_theme_name}'.")

def get_theme_colors() -> dict[str, str]:
    return _themes[_current_theme_name]

def get_current_theme_name() -> str:
    return _current_theme_name

# --- Pre-render SVGs ---
_piece_svgs: dict[str, str] = {}

def _render_svgs():
    global _piece_svgs
    if _piece_svgs:
        return

    print("Pre-rendering piece SVGs...")
    temp_svgs = {}
    for piece in [chess.Piece(pt, col) for pt in chess.PIECE_TYPES for col in chess.COLORS]:
        svg_string = chess.svg.piece(piece)
        if svg_string and svg_string.startswith('<svg '):
            svg_string = svg_string.replace('<svg ', '<svg width="100%" height="100%" ', 1)
        temp_svgs[piece.symbol()] = svg_string
    _piece_svgs = temp_svgs
    print("Finished pre-rendering SVGs.")

def get_piece_svg(symbol: str | None) -> str:
    if not _piece_svgs:
        _render_svgs()
    return _piece_svgs.get(symbol, '') if symbol else ''

# Initial render call
_render_svgs()
