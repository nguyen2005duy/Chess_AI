import pygame
from const import *


class Dragger:
    def __init__(self):
        self.mouseX = float(0)
        self.mouseY = float(0)
        self.piece = None
        self.dragging = False
        self.initial_row = 0
        self.initial_col = 0
    # screen thay vi surface
    # blit methods
    def update_blit(self, screen):
        # texture
        self.piece.set_texture(size=128)
        texture = self.piece.texture
        # img
        img = pygame.image.load(texture)
        # rect
        img_center = (self.mouseX, self.mouseY)
        self.piece.texture_rect = img.get_rect(center=img_center)
        # update blit
        screen.blit(img, self.piece.texture_rect)

    def update_mouse(self, pos):
        self.mouseX, self.mouseY = pos  # (newx, new y)

    def save_initial(self, pos):
        #X
        self.initial_col = pos[0] // SQSIZE
        #Y
        self.initial_row = pos[1] // SQSIZE

    def drag_piece(self, piece):
        self.piece = piece
        self.dragging = True

    def undrag_piece(self):
        self.piece = None
        self.dragging = False
        self.initial_row = -1
        self.initial_col = -1
