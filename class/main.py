import sys
from const import *
from game import *
from dragger import Dragger


class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess_AI')
        self.game = Game()
        self.dragger = Dragger()
        pass

    def mainloop(self):
        screen = self.screen
        game = self.game
        dragger = self.dragger
        board = self.game.board
        for squares in game.board.squares:
            for square in squares:
                print(square.has_piece(), end=" ")
            print()
        while True:
            game.show_background(screen)
            game.show_pieces(screen)
            for event in pygame.event.get():
                # CLick
                if event.type == pygame.MOUSEBUTTONDOWN:
                    dragger.update_mouse(event.pos)
                    clicked_row = dragger.mouseX // SQSIZE
                    clicked_column = dragger.mouseY // SQSIZE
                    print(str(clicked_row) + " " + str(clicked_column))
                    if board[clicked_row][clicked_column].has_piece():
                        print(True)
                    print(event.pos)
                # Move
                elif event.type == pygame.MOUSEMOTION:
                    # dragger.update_mouse(event.pos)

                    pass
                # Click release
                elif event.type == pygame.MOUSEBUTTONUP:
                    pass
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()


main = Main()
main.mainloop()
