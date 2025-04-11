import pygame
import chess
from const import *
from dragger import Dragger
from piece import WrappedPiece  # your wrapper class


class Game:
    def __init__(self):
        self.board = chess.Board()
        self.dragger = Dragger()
        self.last_move = []
        self.is_check = False
        self.white_check = False
        self.black_check = False

    # Show methods
    def show_background(self, screen):
        for row in range(ROWS):
            for col in range(COLS):
                color = (235, 235, 208) if (row + col) % 2 == 0 else (119, 148, 85)
                rect = pygame.Rect(col * SQSIZE, row * SQSIZE, SQSIZE, SQSIZE)
                pygame.draw.rect(screen, color, rect)
        self.show_last_move(screen)
        if self.is_check:
            self.show_check_mate(screen)

    def show_last_move(self, screen):
        last_move = self.last_move
        for i in range(len(self.last_move)):
            color = (246, 246, 105) if i == 0 else (186, 202, 43)
            rect = pygame.Rect(last_move[i][1] * SQSIZE, last_move[i][0] * SQSIZE, SQSIZE, SQSIZE)
            pygame.draw.rect(screen, color, rect)

    def show_check_mate(self, screen):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.symbol() == 'k' and self.black_check:
                row = 7 - (square // 8)
                col = square % 8
                color = (255, 0, 0)
                rect = pygame.Rect(col * SQSIZE, row * SQSIZE, SQSIZE, SQSIZE)
                pygame.draw.rect(screen, color, rect)
            elif piece and piece.symbol() == 'K' and self.white_check:
                row = 7 - (square // 8)
                col = square % 8
                color = (255, 0, 0)
                rect = pygame.Rect(col * SQSIZE, row * SQSIZE, SQSIZE, SQSIZE)
                pygame.draw.rect(screen, color, rect)

    def show_pieces(self, screen):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8

                # Skip rendering the dragged piece
                if self.dragger.piece and self.dragger.piece.row == row and self.dragger.piece.col == col:
                    continue

                # Dung wrapper de wrap piece cua python-chess
                wrapper_piece = WrappedPiece(piece, row, col)
                wrapper_piece.set_texture(size=80)
                img = pygame.image.load(wrapper_piece.texture)
                img_center = col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2
                wrapper_piece.texture_rect = img.get_rect(center=img_center)
                screen.blit(img, wrapper_piece.texture_rect)
