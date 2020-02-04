# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 22:25:41 2019

@author: Алексей
"""
import pygame
import game 
import ga
import numpy as np

pygame.init()
clock = pygame.time.Clock()
pygame.display.set_caption("Snake: NN & GA")
screen = pygame.display.set_mode((game.W, game.H))
screen.fill(game.BLACK)

total = []

cur_gen = 1
num_in_gen = 800
mutation_rate = 0.12
inp_n = 24
h1_n = 24
h2_n = 24
out_n = 4

    
cur_val = 50
for i in range(cur_val, 100):
    cur_gen = i
    cur_fit, maximum = ga.go_through_gen(cur_gen, num_in_gen, inp_n, h1_n, h2_n, out_n, screen, clock, mutation_rate)
    print("Gen " + str(i) + ": summ = " + str(cur_fit), " max: " + str(maximum))

pygame.quit()