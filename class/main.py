import sys
import pygame
import chess
from const import *
from game import Game
from piece import WrappedPiece

from dragger import Dragger


class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess_AI')
        self.game = Game()
        self.dragger = self.game.dragger  # use dragger from Game class

    def mainloop(self):
        screen = self.screen
        game = self.game
        dragger = self.dragger
        board = game.board  # This is a python-chess board

        while True:
            game.show_background(screen)
            game.show_pieces(screen)

            if dragger.dragging:
                dragger.update_blit(screen)

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    print(event.pos)
                    dragger.update_mouse(event.pos)

                    row = dragger.mouseY // SQSIZE
                    col = dragger.mouseX // SQSIZE
                    square = chess.square(col, 7 - row)
                    piece = board.piece_at(square)
                    print(piece)
                    dragger.save_initial(event.pos)
                    if piece is not None:
                        # Wrap the piece for rendering
                        wrapped_piece = WrappedPiece(piece, row, col)
                        dragger.drag_piece(wrapped_piece)

                elif event.type == pygame.MOUSEMOTION:
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        dragger.update_blit(screen)

                elif event.type == pygame.MOUSEBUTTONUP:
                    released_row = dragger.mouseY // SQSIZE
                    released_col = dragger.mouseX // SQSIZE
                    # 7- tai vi python-chess row bi nguoc voi ca row minh dung o dragger.mouseY
                    from_square = chess.square(dragger.initial_col, 7 - dragger.initial_row)
                    to_square = chess.square(released_col, 7 - released_row)
                    move = chess.Move(from_square, to_square)
                    print(str(dragger.initial_row) + " " + str(dragger.initial_col))
                    print(str(released_row) + " " + str(released_col))
                    print(move)
                    if move in board.legal_moves:
                        game.show_move()
                        board.push(move)
                    dragger.undrag_piece()

                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()


if __name__ == "__main__":
    main = Main()
    main.mainloop()
