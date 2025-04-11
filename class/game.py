import pygame
import chess
from const import *
from dragger import Dragger
from piece import WrappedPiece  # your wrapper class


class Game:
    def __init__(self):
        self.board = chess.Board()
        self.dragger = Dragger()

    # Show methods
    def show_background(self, screen):
        for row in range(ROWS):
            for col in range(COLS):
                color = (235, 235, 208) if (row + col) % 2 == 0 else (119, 148, 85)
                rect = pygame.Rect(col * SQSIZE, row * SQSIZE, SQSIZE, SQSIZE)
                pygame.draw.rect(screen, color, rect)

    def show_move(self, screen):


    def show_pieces(self, screen):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8

                # Skip rendering the dragged piece
                if self.dragger.piece and self.dragger.piece.row == row and self.dragger.piece.col == col:
                    continue

                # Wrap python-chess piece in your Piece class
                wrapper_piece = WrappedPiece(piece, row, col)
                wrapper_piece.set_texture(size=80)
                img = pygame.image.load(wrapper_piece.texture)
                img_center = col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2
                wrapper_piece.texture_rect = img.get_rect(center=img_center)
                screen.blit(img, wrapper_piece.texture_rect)
