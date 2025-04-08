import pygame
from const import *


class Dragger:
    def __init__(self):
        self.mouse_x = float(0)
        self.mouse_y = float(0)

    def update_mouse(self, pos):
        self.mouseX, self.mouseY = pos  # (newx, new y)
