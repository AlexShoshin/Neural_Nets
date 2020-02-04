# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:16:12 2019

@author: Алексей
"""

import pygame
import sys
import random
import numpy as np

FPS = 200
W = 750
H = 440
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
SNAKE_PART_SIZE = 12
DELTA = 1
D = SNAKE_PART_SIZE + DELTA
border_x = 330
border_y = 20
border_len = 390

def compute_nn_inputs(snake, food):
    head = snake[0]
    inputs = np.zeros((12, 1))
    if(head.left == food.left):
        if(head.top < food.top):
            inputs[0] = 30 / ((food.top - head.top ) / D + 1)
        else:
            inputs[6] = 30 / ((head.top - food.top ) / D + 1)
    if(head.top == food.top):
        if(head.left < food.left):
            inputs[3] = 30 / ((food.left - head.left) / D + 1)
        else:
            inputs[9] = 30 / ((head.left - food.left) / D + 1)
            
    inputs[1] = (-10) / ((head.top - border_y) / D + 1)
    inputs[4] = (-10) / ((border_x + border_len - head.left - 1) / D + 1)
    inputs[7] =  (-10) / ((border_y + border_len - head.top - 1) / D + 1)
    inputs[10] =  (-10) / ((head.left - border_x) / D + 1)
    
    for i in range(1, len(snake)):
        if (snake[i].left == head.left):
            if(snake[i].top < head.top):
                inputs[2] = (-10) / ((head.top - snake[i].top - 1) / D + 1)
            else:
                inputs[8] = (-10) / ((snake[i].top - head.top - 1) / D + 1)
        if (snake[i].top == head.top):
            if(snake[i].left  < head.left):
                inputs[5] = (-10) / ((head.left - snake[i].left - 1) / D + 1)
            else:
                inputs[11] =  (-10) / ((snake[i].left - head.left - 1) / D + 1)
    
    
    return inputs.T



def compute_nn_inputs_v2(snake, food):
    inputs = np.zeros((1, 24))
    head = snake[0]
    if(food.left == head.left):
        if(food.top > head.top):
            inputs[0, 12] = 30 / ((food.top - head.top) / D + 1)
        else:
            inputs[0, 0] = 30 / ((head.top - food.top) / D + 1)
    if(food.top == head.top):
        if(food.left > head.left):
            inputs[0, 6] = 30 / ((food.left - head.left) / D + 1)
        else:
            inputs[0, 18] = 30 / ((head.left - food.left) / D + 1)
    if(food.left - head.left == food.top - head.top):
        if(food.left > head.left):
            inputs[0, 9] = 30 / ((food.left - head.left) / D + 1)
        else:
            inputs[0, 21] = 30 / ((head.left - food.left) / D + 1)
    if(food.left - head.left == head.top - food.top):
        if(food.left > head.left):
            inputs[0, 3] = 30 / ((food.left - head.left) / D + 1)
        else:
            inputs[0, 15] = 30 / ((head.left - food.left) / D + 1)
            
    inputs[0, 1] = (-10) / ((head.top - border_y) / D + 1)
    inputs[0, 7] = (-10) / ((border_x + border_len - head.left) / D + 1)
    inputs[0, 4] = np.maximum(inputs[0, 1], inputs[0, 7])
    inputs[0, 13] = (-10) / ((border_y + border_len - head.top) / D + 1)
    inputs[0, 10] = np.maximum(inputs[0, 7], inputs[0, 13])
    inputs[0, 19] = (-10) / ((head.left - border_x) / D + 1)
    inputs[0, 16] = np.maximum(inputs[0, 13], inputs[0, 19])
    inputs[0, 22] = np.maximum(inputs[0, 0], inputs[0, 19])
    
    
    for i in range(1, int(inputs[0, 1]) + 1):
        rect = pygame.Rect(head.left, head.top - i * D, D, D)
        if(rect.collidelist(snake) != -1):
            inputs[0, 2] = (-10) / i
            break
    for i in range(1, int(inputs[0, 4]) + 1):
        rect = pygame.Rect(head.left + i * D, head.top - i * D, D, D)
        if(rect.collidelist(snake) != -1):
            inputs[0, 5] = (-10) / i
            break
    for i in range(1, int(inputs[0, 7]) + 1):
        rect = pygame.Rect(head.left + i * D, head.top, D, D)
        if(rect.collidelist(snake) != -1):
            inputs[0, 8] = (-10) / i
            break
    for i in range(1, int(inputs[0, 10]) + 1):
        rect = pygame.Rect(head.left + i * D, head.top + i * D, D, D)
        if(rect.collidelist(snake) != -1):
            inputs[0, 11] = (-10) / i
            break
    for i in range(1, int(inputs[0, 13]) + 1):
        rect = pygame.Rect(head.left, head.top + i * D, D, D)
        if(rect.collidelist(snake) != -1):
            inputs[0, 14] = (-10) / i
            break
    for i in range(1, int(inputs[0, 16]) + 1):
        rect = pygame.Rect(head.left - i * D, head.top + i * D, D, D)
        if(rect.collidelist(snake) != -1):
            inputs[0, 17] = (-10) / i
            break
    for i in range(1, int(inputs[0, 19]) + 1):
        rect = pygame.Rect(head.left - i * D, head.top, D, D)
        if(rect.collidelist(snake) != -1):
            inputs[0, 20] = (-10) / i
            break
    for i in range(1, int(inputs[0, 22]) + 1):
        rect = pygame.Rect(head.left - i * D, head.top - i * D, D, D)
        if(rect.collidelist(snake) != -1):
            inputs[0, 23] = (-10) / i
            break
    
    return inputs
    


def play_game(nn, screen, clock, cur_num, cur_gen, num_in_gen, mutation_rate):
    
    score = 0
    mooves_left = 200
    snake = []
    init_len = 3
    
    f1 = pygame.font.SysFont('arial', 20)
    f2 = pygame.font.SysFont('arial', 35)
    text2 = f1.render('Current snake in generation:' + str(cur_num) + " / " + str(num_in_gen), 1, WHITE)
    text2_rect = text2.get_rect(center=(143, 300))
    text1 = f1.render('Current generation:' + str(cur_gen), 1, WHITE)
    text1_rect = text1.get_rect(center=(85, 250))
    text3 = f1.render('Moves left:' + str(mooves_left), 1, WHITE)
    text3_rect = text3.get_rect(center=(66, 350))
    text4 = f2.render('Evolution of snake,', 1, WHITE)
    text4_rect = text4.get_rect(center=(150, 50))
    text5 = f2.render('controlled by Neural Net', 1, WHITE)
    text5_rect = text5.get_rect(center=(162, 90))
    text6 = f2.render('and trained by GenAlg', 1, WHITE)
    text6_rect = text6.get_rect(center=(162, 130))
    text7 = f1.render('Mutation rate: ' + str(mutation_rate), 1, WHITE)
    text7_rect = text7.get_rect(center=(80, 400))
    

    head_position_x = 525
    head_position_y = 215

    position_x = head_position_x
    position_y = head_position_y

    for i in range(init_len):
        snake.append(pygame.Rect(position_x, position_y, SNAKE_PART_SIZE, SNAKE_PART_SIZE))
        position_x -= SNAKE_PART_SIZE + DELTA

    direction = 1
    prev_x = 0
    prev_y = 0
    temp_x = 0
    temp_y = 0
    ate = 0

    food_x = border_x + (SNAKE_PART_SIZE + DELTA) * random.randint(1, 29)
    food_y = border_y + (SNAKE_PART_SIZE + DELTA) * random.randint(1, 29)
    food = pygame.Rect(food_x, food_y, SNAKE_PART_SIZE, SNAKE_PART_SIZE)

    end_flag = 0
    while (end_flag == 0):
    
        #if itten, food relocates
        if(ate == 1):
            score += 1
            mooves_left += 85
            snake.append(pygame.Rect(snake[len(snake) - 1].left, snake[len(snake) - 1].top, SNAKE_PART_SIZE, SNAKE_PART_SIZE))
            ate = 0
            food.left = border_x + (SNAKE_PART_SIZE + DELTA) * random.randint(1, 29)
            food.top = border_y + (SNAKE_PART_SIZE + DELTA) * random.randint(1, 29)
            while(food.collidelist(snake) != -1):
                food.left = border_x + (SNAKE_PART_SIZE + DELTA) * random.randint(1, 29)
                food.top = border_y + (SNAKE_PART_SIZE + DELTA) * random.randint(1, 29)
    
    
        #handle ivents
    
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
        nn_inp = compute_nn_inputs_v2(snake, food)
        nn_out = nn.query(nn_inp)
    
   
        choise = np.argmax(nn_out)
        if(np.abs(choise - direction) != 2):
            direction = choise
    
    
        #change head position
        prev_x = snake[0].left
        prev_y = snake[0].top
        if(direction == 1):
            snake[0].move_ip(SNAKE_PART_SIZE + DELTA, 0)
        elif(direction == 2):
            snake[0].move_ip(0, SNAKE_PART_SIZE + DELTA)
        elif(direction == 3):
            snake[0].move_ip(-SNAKE_PART_SIZE - DELTA, 0)
        elif(direction == 0):
            snake[0].move_ip(0, -SNAKE_PART_SIZE - DELTA)
    
    
        if(snake[0].left == food.left and snake[0].top == food.top):
            ate = 1
    
        #move rest of snake
        for i in range(1, len(snake)):
            temp_x = snake[i].left
            temp_y = snake[i].top
            snake[i].move_ip(prev_x - snake[i].left, prev_y - snake[i].top)
            prev_x = temp_x
            prev_y = temp_y
        
        #check for border collision
        if(snake[0].left < border_x or snake[0].left >= border_x + border_len or 
           snake[0].top < border_y or snake[0].top >= border_y + border_len):
            end_flag = 1
            return score
        
        #check for snake itself collision
        collide = snake[1:]
        if(snake[0].collidelist(collide) != -1):
            end_flag = 1
            return score
        
        mooves_left -= 1
        if(mooves_left == 0):
            end_flag = 1
            return score
        text3 = f1.render('Moves left: ' + str(mooves_left), 1, WHITE)
        
        #draw
        screen.fill(BLACK)
        pygame.draw.rect(screen, RED, food)
    
        for i in range(len(snake)):
            pygame.draw.rect(screen, WHITE, snake[i])
    
        pygame.draw.line(screen, WHITE, (border_x, border_y), (border_x + border_len, border_y))
        pygame.draw.line(screen, WHITE, (border_x + border_len, border_y), (border_x + border_len, border_y + border_len))
        pygame.draw.line(screen, WHITE, (border_x, border_y + border_len), (border_x + border_len, border_y + border_len))
        pygame.draw.line(screen, WHITE, (border_x, border_y), (border_x, border_y + border_len))
        screen.blit(text1, text1_rect)
        screen.blit(text2, text2_rect)
        screen.blit(text3, text3_rect)
        screen.blit(text4, text4_rect)
        screen.blit(text5, text5_rect)
        screen.blit(text6, text6_rect)
        screen.blit(text7, text7_rect)
        clock.tick(FPS)
        pygame.display.update()
    return score

        



