import pygame
import os
import os.path as osp

path_file = os.path.dirname(__file__)
path_images = osp.join(path_file, 'images')

def load_bird_male():
    obj = 'bird_male.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)

def load_bird_female():
    obj = 'bird_female.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)

def load_background():
    obj = 'background.jpg'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)

def load_obstacle():
    obj = 'obstacle.png'
    obj_path = osp.join(path_images, obj)
    return pygame.image.load(obj_path)