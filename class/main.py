import sys
import pygame
import chess
import os
from const import *
from game import Game
from piece import WrappedPiece
from minmaxAI import minmaxAI
from dragger import Dragger

from ChessAI import *


MENU = 0
PLAYING = 1


class Main:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess_AI")
        self.game = Game()
        self.dragger = self.game.dragger  # use dragger from Game class
        self.minmaxAI = minmaxAI()
        self.chess_ai = ChessAI()
        self.font = pygame.font.SysFont("Arial", 40)
        self.small_font = pygame.font.SysFont("Arial", 30)
        self.state = MENU
        self.play_button_rect = None
        self.quit_button_rect = None
        self.player_color = None

    def draw_menu(self, screen):
        screen.fill((20, 20, 20))

        title_surf = self.font.render("Chess AI", True, (220, 220, 220))
        title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 4))
        screen.blit(title_surf, title_rect)

        play_surf = self.small_font.render("Play Game", True, (200, 200, 200))
        self.play_button_rect = play_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        pygame.draw.rect(
            screen, (50, 50, 50), self.play_button_rect.inflate(20, 10)
        )  
        screen.blit(play_surf, self.play_button_rect)

        quit_surf = self.small_font.render("Quit", True, (200, 200, 200))
        self.quit_button_rect = quit_surf.get_rect(
            center=(WIDTH // 2, HEIGHT // 2 + 60)
        )
        pygame.draw.rect(
            screen, (50, 50, 50), self.quit_button_rect.inflate(20, 10)
        ) 
        screen.blit(quit_surf, self.quit_button_rect)

    def mainloop(self):
        screen = self.screen
        game = self.game
        dragger = self.dragger
        board = game.board  # This is a python-chess board
        minmaxAI = self.minmaxAI
        while True:
            if self.state == MENU:
                self.draw_menu(screen)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = event.pos
                        if (
                            self.play_button_rect
                            and self.play_button_rect.collidepoint(pos)
                        ):
                            self.player_color = game.show_color_selection_menu(screen)
                            self.state = PLAYING 
                        elif (
                            self.quit_button_rect
                            and self.quit_button_rect.collidepoint(pos)
                        ):
                            pygame.quit()
                            sys.exit()

            elif self.state == PLAYING:
                game.show_background(screen)
                game.show_pieces(screen)

                if dragger.dragging:
                    game.show_moves(screen) 
                    dragger.update_blit(screen)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        dragger.update_mouse(event.pos)

                        row = dragger.mouseY // SQSIZE
                        col = dragger.mouseX // SQSIZE
                        square = chess.square(col, 7 - row)
                        piece = board.piece_at(square)

                        if piece is not None and piece.color == board.turn:
                            wrapped_piece = WrappedPiece(piece, row, col)
                            dragger.save_initial(event.pos)
                            dragger.drag_piece(wrapped_piece)

                    elif event.type == pygame.MOUSEMOTION:
                        if dragger.dragging:
                            dragger.update_mouse(event.pos)

                    elif event.type == pygame.MOUSEBUTTONUP:
                        if dragger.dragging:
                            dragger.update_mouse(event.pos)
                            released_row = dragger.mouseY // SQSIZE
                            released_col = dragger.mouseX // SQSIZE
                            # 7- tai vi python-chess row bi nguoc voi ca row minh dung o dragger.mouseY
                            from_square = chess.square(
                                dragger.initial_col, 7 - dragger.initial_row
                            )
                            to_square = chess.square(released_col, 7 - released_row)

                            piece = board.piece_at(from_square)

                            # Phong hau, can mot cai GUI de phong cac con khac
                            is_promotion = (
                                piece
                                and piece.piece_type == chess.PAWN
                                and (
                                    chess.square_rank(to_square) == 0
                                    or chess.square_rank(to_square) == 7
                                )
                            )

                            #promotion GUI - done
                            if is_promotion:
                                promotion_menu = game.show_promotion_menu(screen)
                                chosen_piece = None
                                if promotion_menu == 'q':
                                    chosen_piece = chess.QUEEN
                                elif promotion_menu == 'r':
                                    chosen_piece = chess.ROOK
                                elif promotion_menu == 'b':
                                    chosen_piece = chess.BISHOP
                                elif promotion_menu == 'n':
                                    chosen_piece = chess.KNIGHT

                                move = chess.Move(
                                    from_square, to_square, promotion=chosen_piece
                                )
                            else:
                                move = chess.Move(from_square, to_square)

                            if move in board.legal_moves:
                                game.last_move = [
                                    [dragger.initial_row, dragger.initial_col],
                                    [released_row, released_col],
                                ]
                                script_dir = os.path.dirname(__file__)
                                sound_path = os.path.join(
                                    script_dir, "..", "assets", "sounds", "move.wav"
                                )
                                pygame.mixer.Sound(os.path.abspath(sound_path)).play()
                                board.push(move)

                                #AI move
                                #bug: nếu chọn quân đen --> AI đi quân trắng nhưng lượt đi đầu
                                #tiên của quân trắng lại là mình chơi
                                if ((board.turn and self.player_color == 'black') or 
                                    (not board.turn and self.player_color == 'white')):
                                    pygame.display.update()
                                    #ai_move: chess.Move = minmaxAI.calculate_move(board)
                                    ai_move: chess.Move = self.chess_ai.calculate_move(board)
                                    from_square: chess.Move.from_square = (
                                        ai_move.from_square
                                    )
                                    to_square: chess.Move.to_square = ai_move.to_square
                                    from_row = 7 - from_square // 8
                                    from_col = from_square % 8
                                    to_row = 7 - to_square // 8
                                    to_col = to_square % 8
                                    game.last_move = [
                                        [from_row, from_col],
                                        [to_row, to_col],
                                    ]
                                    board.push(ai_move)
                                

                                 # Kiem tra co dang bi chieu tuong hay khong
                                if board.is_check():
                                    print(
                                        "Checkmate!"
                                    ) 
                                    if ((board.turn and self.player_color == 'black') or 
                                        (not board.turn and self.player_color == 'white')): 
                                        game.is_check = True
                                        game.white_check = True
                                    else:
                                        game.is_check = True
                                        game.black_check = True
                                else:
                                    game.is_check = False
                                    game.white_check = False
                                    game.black_check = False

                            dragger.undrag_piece() 
            pygame.display.update()


if __name__ == "__main__":
    main = Main()
    main.mainloop()
