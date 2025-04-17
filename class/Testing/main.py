import pygame
import chess
import chess.svg
import numpy as np
import torch
from chess_env import ChessEnv
from model1 import Model1

# Initialize Pygame
pygame.init()

# Window dimensions
WIDTH, HEIGHT = 600, 600
square_size = WIDTH // 8  # 8x8 grid

# Colors for the board squares
WHITE = pygame.Color("white")
BLACK = pygame.Color("gray")

# Set up the Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess AI")

# Load chess piece images
# (Replace these with actual file paths for your chess pieces)
images = {
    "K": pygame.image.load("images/white_king.png"),
    "Q": pygame.image.load("images/white_queen.png"),
    "R": pygame.image.load("images/white_rook.png"),
    "B": pygame.image.load("images/white_bishop.png"),
    "N": pygame.image.load("images/white_knight.png"),
    "P": pygame.image.load("images/white_pawn.png"),
    "k": pygame.image.load("images/black_king.png"),
    "q": pygame.image.load("images/black_queen.png"),
    "r": pygame.image.load("images/black_rook.png"),
    "b": pygame.image.load("images/black_bishop.png"),
    "n": pygame.image.load("images/black_knight.png"),
    "p": pygame.image.load("images/black_pawn.png")
}

# Convert the FEN string into a usable format
def fen_to_input(fen):
    board = chess.Board(fen)
    board_array = np.zeros(64)  # 8x8 board flattened into a 64-length array

    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            piece_value = 1 if piece.color == chess.WHITE else -1  # White or Black
            if piece.piece_type == chess.PAWN:
                board_array[square] = piece_value * 1
            elif piece.piece_type == chess.KNIGHT:
                board_array[square] = piece_value * 3
            elif piece.piece_type == chess.BISHOP:
                board_array[square] = piece_value * 3
            elif piece.piece_type == chess.ROOK:
                board_array[square] = piece_value * 5
            elif piece.piece_type == chess.QUEEN:
                board_array[square] = piece_value * 9
            elif piece.piece_type == chess.KING:
                board_array[square] = piece_value * 1000
    return board_array

# Draw the board
def draw_board(board):
    for row in range(8):
        for col in range(8):
            rect = pygame.Rect(col * square_size, row * square_size, square_size, square_size)
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, rect)

            # Get the piece at this square and draw it
            piece = board.piece_at(row * 8 + col)
            if piece:
                piece_image = images.get(piece.symbol())
                if piece_image:
                    piece_rect = piece_image.get_rect(center=rect.center)
                    screen.blit(piece_image, piece_rect)

# Function to display the game state
def display_game(board_fen):
    # Create the chess board object from FEN
    board = chess.Board(board_fen)

    # Fill the background
    screen.fill(pygame.Color("black"))

    # Draw the board
    draw_board(board)

    # Update the screen
    pygame.display.flip()

# Display the initial board
def play_game_with_ai(model, env):
    state = env.reset()
    done = False

    while not done:
        # Convert the current FEN to an input format for the model
        state_input = fen_to_input(state)
        state_input = torch.tensor(state_input, dtype=torch.float32).unsqueeze(0)

        # Get the number of legal moves
        legal_moves = env.get_legal_moves()

        # Modify the model to output a score for each legal move
        model = Model1(len(legal_moves))

        # Predict the next move using the model
        with torch.no_grad():
            predicted_values = model(state_input).squeeze().numpy()

        # Select the best move based on the predicted values
        best_move_index = np.argmax(predicted_values)
        move = legal_moves[best_move_index]

        # Perform the move
        next_state, reward, done = env.step(move)
        state = next_state

        # Render the current state of the board with Pygame
        display_game(state)
        pygame.time.wait(500)  # Wait briefly before the next move

    print("Game Over!")
    pygame.quit()

# Initialize Pygame and the environment
env = ChessEnv()  # You should already have this
model = Model1(len(env.get_legal_moves()))  # Initialize the model with the correct number of legal moves

# Play a game with the AI (uncomment if you have a trained model)
# model.load_state_dict(torch.load("models/trained_model_final.pth"))
model.eval()

# Play the game with AI
play_game_with_ai(model, env)
