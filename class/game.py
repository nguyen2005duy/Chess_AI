import pygame
import chess
import sys
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

    def show_moves(self, screen):
        if self.dragger.dragging and self.dragger.piece:
            move_color = (0, 0, 0, 100) 
            capture_color = (200, 0, 0, 150) 
      
            move_surface = pygame.Surface((SQSIZE, SQSIZE), pygame.SRCALPHA)
            pygame.draw.circle(move_surface, move_color, (SQSIZE // 2, SQSIZE // 2), SQSIZE // 6)

            capture_surface = pygame.Surface((SQSIZE, SQSIZE), pygame.SRCALPHA)
            pygame.draw.circle(capture_surface, capture_color, (SQSIZE // 2, SQSIZE // 2), SQSIZE // 2, width=5) # Draw a ring for captures


            initial_row, initial_col = self.dragger.initial_row, self.dragger.initial_col
            from_square = chess.square(initial_col, 7 - initial_row)

            for move in self.board.legal_moves:
                if move.from_square == from_square:
                    to_square = move.to_square
                    to_row = 7 - chess.square_rank(to_square)
                    to_col = chess.square_file(to_square)

                    is_capture = self.board.is_capture(move)

                    blit_pos = (to_col * SQSIZE, to_row * SQSIZE)

                    if is_capture:
                        screen.blit(capture_surface, blit_pos)
                    else:
                        screen.blit(move_surface, blit_pos)
    
    def show_promotion_menu(self, surface):
        white_piece = self.board.turn
        screen = surface
        color = (234, 235, 200) if not white_piece else (119, 154, 88) #light green - black piece
        screen.fill(color)

        str = 'white' if white_piece else 'black'

        promotion_pieces = ['queen', 'rook', 'bishop', 'knight']
        piece_keys = ['q', 'r', 'b', 'n']
        piece_images = []

        for name in promotion_pieces:
            path = f'../assets/images/imgs-128px/{str}_{name}.png'
            image = pygame.image.load(path)
            image = pygame.transform.scale(image, (SQSIZE, SQSIZE))
            piece_images.append(image)

        section_height = HEIGHT//4
        rects = []


        color1 = (119, 154, 88) if not white_piece else (234, 235, 200)
        for i in range(4):
            y = i * section_height
            rect = pygame.Rect(0, y, WIDTH, section_height)
            rects.append(rect)

            pygame.draw.rect(screen, color1, rect, 3)

            img_x = (WIDTH - SQSIZE) // 2
            img_y = y + (section_height - SQSIZE) // 2
            
            screen.blit(piece_images[i], (img_x, img_y))
        
        pygame.display.update()

        while True: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    for i, rect in enumerate(rects):
                        if rect.collidepoint(mouse_x, mouse_y):
                            return piece_keys[i]
                        
    def show_color_selection_menu(self, surface):
        screen = surface
        font = pygame.font.SysFont("Arial", 30)
        screen.fill((30, 30, 30))

        white_rect = pygame.Rect(WIDTH // 4, HEIGHT // 2 - 60, WIDTH // 2, 50)
        black_rect = pygame.Rect(WIDTH // 4, HEIGHT // 2 + 20, WIDTH // 2, 50)

        while True:
            pygame.draw.rect(screen, (240, 240, 240), white_rect)
            pygame.draw.rect(screen, (50, 50, 50), black_rect)

            white_text = font.render("White", True, (0, 0, 0))
            black_text = font.render("Black", True, (255, 255, 255))

            screen.blit(white_text, white_text.get_rect(center=white_rect.center))
            screen.blit(black_text, black_text.get_rect(center=black_rect.center))

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if white_rect.collidepoint(event.pos):
                        return 'white'
                    elif black_rect.collidepoint(event.pos):
                        return 'black'


