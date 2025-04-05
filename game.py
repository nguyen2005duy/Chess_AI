import pygame
from const import *
from board import Board


class Game:
    def __init__(self):
        self.board = Board()

    # Show methods
    def show_background(self, screen):
        for row in range(ROWS):
            for col in range(COLS):
                if (row + col) % 2 == 0:
                    color = (235, 235, 208)  # light green
                else:
                    color = (119, 148, 85)  # dark green
                rect = pygame.Rect(col * SQSIZE, row * SQSIZE, SQSIZE, SQSIZE)
                pygame.draw.rect(screen, color, rect)

    def show_pieces(self, screen):
        for row in range(ROWS):
            for col in range(COLS):
                # has piece?
                if self.board.squares[row][col].has_piece():
                    piece = self.board.squares[row][col].piece
                    img = pygame.image.load(piece.texture)
                    img_center = col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2
                    piece.texture_rect = img.get_rect(center=img_center)
                    screen.blit(img, piece.texture_rect)
