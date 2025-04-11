import sys
import pygame
import chess
from const import *
from game import Game
from piece import WrappedPiece
from minmaxAI import minmaxAI
from dragger import Dragger


class Main:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess_AI')
        self.game = Game()
        self.dragger = self.game.dragger  # use dragger from Game class
        self.minmaxAI = minmaxAI()

    def mainloop(self):
        screen = self.screen
        game = self.game
        dragger = self.dragger
        board = game.board  # This is a python-chess board
        minmaxAI = self.minmaxAI
        while True:
            game.show_background(screen)
            game.show_pieces(screen)

            if dragger.dragging:
                dragger.update_blit(screen)

            for event in pygame.event.get():

                if event.type == pygame.MOUSEBUTTONDOWN:
                    dragger.update_mouse(event.pos)

                    row = dragger.mouseY // SQSIZE
                    col = dragger.mouseX // SQSIZE
                    square = chess.square(col, 7 - row)
                    piece = board.piece_at(square)
                    dragger.save_initial(event.pos)
                    if piece is not None and piece.color:
                        # Wrap the piece for rendering
                        wrapped_piece = WrappedPiece(piece, row, col)
                        dragger.drag_piece(wrapped_piece)

                elif event.type == pygame.MOUSEMOTION:
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        dragger.update_blit(screen)

                elif event.type == pygame.MOUSEBUTTONUP:
                    print("Mouse_up")
                    released_row = dragger.mouseY // SQSIZE
                    released_col = dragger.mouseX // SQSIZE
                    # 7- tai vi python-chess row bi nguoc voi ca row minh dung o dragger.mouseY
                    from_square = chess.square(dragger.initial_col, 7 - dragger.initial_row)
                    to_square = chess.square(released_col, 7 - released_row)
                    piece = board.piece_at(from_square)
                    # Phong hau, can mot cai GUI de phong cac con khac
                    is_promotion = (
                            piece and
                            piece.piece_type == chess.PAWN and
                            (released_row == 0 or released_row == 7)
                    )
                    if is_promotion:
                        move = chess.Move(from_square, to_square,
                                          promotion=chess.QUEEN)
                    else:
                        move = chess.Move(from_square, to_square)
                    if move in board.legal_moves:
                        game.last_move = [[dragger.initial_row, dragger.initial_col], [released_row, released_col]]
                        pygame.mixer.Sound('../assets/sounds/move.wav').play()
                        board.push(move)
                    # Ai_move
                    if not board.turn:
                        ai_move: chess.Move = minmaxAI.ai_move(board)
                        from_square: chess.Move.from_square = ai_move.from_square
                        to_square: chess.Move.to_square = ai_move.to_square
                        from_row = 7 - from_square // 8
                        from_col = from_square % 8
                        to_row = 7 - to_square // 8
                        to_col = to_square % 8
                        game.last_move = [[from_row, from_col], [to_row, to_col]]
                        board.push(ai_move)
                    # Kiem tra co dang bi chieu tuong hay khong
                    if board.is_check():
                        print("Check mate!")
                        if board.turn:
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


                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()


if __name__ == "__main__":
    main = Main()
    main.mainloop()
