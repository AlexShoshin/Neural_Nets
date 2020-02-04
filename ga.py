# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:37:16 2019

@author: Алексей
"""

import numpy as np
import nn
import game

def setup(num_in_gen, inp, h1, h2, out):
    inp_all = np.random.normal(0.0, pow(h1, -0.5), (h1 * inp, num_in_gen))
    h1_all = np.random.normal(0.0, pow(h1, -0.5), (h2 * h1, num_in_gen))
    h2_all = np.random.normal(0.0, pow(h1, -0.5), (out * h2, num_in_gen))
    
    return inp_all, h1_all, h2_all

def load_new_gen(cur_gen):
    inp_all = np.load("data/raw_gen_" + str(cur_gen - 1) + "_inp_data.npy")
    h1_all = np.load("data/raw_gen_" + str(cur_gen - 1) + "_h1_data.npy")
    h2_all = np.load("data/raw_gen_" + str(cur_gen - 1) + "_h2_data.npy")
    
    return inp_all, h1_all, h2_all

def get_weights(inp_all, h1_all, h2_all, inp_n, h1_n, h2_n, out_n, i):
    cur_inp = inp_all[:, i]
    cur_inp = np.reshape(cur_inp, (h1_n, inp_n))
    cur_h1 = h1_all[:, i]
    cur_h1 = np.reshape(cur_h1, (h2_n, h1_n))
    cur_h2 = h2_all[:, i]
    cur_h2 = np.reshape(cur_h2, (out_n, h2_n))
    
    return cur_inp, cur_h1, cur_h2

def count_fitness(total, num_in_gen, cur_gen):
    fitness = np.zeros((1, num_in_gen))
    summ = 0
    maxim = np.amax(total)
    for i in range(num_in_gen):
        fitness[0, i] = (total[i])
        summ += total[i]
    
    np.save("maxs/max_ate_in_gen" + str(cur_gen) + ".npy", maxim)
    np.save("sums/sum_ate_in_gen" + str(cur_gen) + ".npy", summ)
    
    fitness = fitness**2
    return fitness, summ, maxim

def create_next_gen_snake(inp_all, h1_all, h2_all, fitness, num_in_pop, mutation_rate):
    if(np.sum(fitness) != 0):
        probs = fitness / np.sum(fitness)
        par1_ind = choose_parent(probs)
        par2_ind = choose_parent(probs)
    else:
        par1_ind = np.random.randint(0, num_in_pop)
        par2_ind = np.random.randint(0, num_in_pop)
    
    new_inp, new_h1, new_h2 = crossover(par1_ind, par2_ind, inp_all, h1_all, h2_all)
    new_inp, new_h1, new_h2 = mutation(new_inp, new_h1, new_h2, mutation_rate)
    return new_inp, new_h1, new_h2
    
def choose_parent(probs):
    r = np.random.random()
    index = 0
    while(r > 0):
        r -= probs[0, index]
        index += 1
    index -= 1
    return index

def crossover(par1_ind, par2_ind, inp_all, h1_all, h2_all):
    inp_point = int(inp_all.shape[0] / 2)
    par1_inp = inp_all[:, par1_ind]
    par2_inp = inp_all[:, par2_ind]
    new_inp = par1_inp[0:inp_point]
    new_inp = np.concatenate((new_inp, par2_inp[inp_point:]), axis=0)
    
    h1_point = int(h1_all.shape[0] / 2)
    par1_h1 = h1_all[:, par1_ind]
    par2_h1 = h1_all[:, par2_ind]
    new_h1 = par1_h1[0:h1_point]
    new_h1 = np.concatenate((new_h1, par2_h1[h1_point:]), axis=0)
    
    h2_point = int(h2_all.shape[0] / 2)
    par1_h2 = h2_all[:, par1_ind]
    par2_h2 = h2_all[:, par2_ind]
    new_h2 = par1_h2[0:h2_point]
    new_h2 = np.concatenate((new_h2, par2_h2[h2_point:]), axis=0)
    
    new_inp = np.reshape(new_inp, (-1, 1))
    new_h1 = np.reshape(new_h1, (-1, 1))
    new_h2 = np.reshape(new_h2, (-1, 1))
    
    return new_inp, new_h1, new_h2

def mutation(new_inp, new_h1, new_h2, mutation_rate):
    
    for i in range(new_inp.shape[0]):
        r = np.random.random()
        if(r < mutation_rate):
            new_inp[i, 0] += np.random.uniform(-1, 1)
            
    for i in range(new_h1.shape[0]):
        r = np.random.random()
        if(r < mutation_rate):
            new_h1[i, 0] += np.random.uniform(-1, 1)
            
    for i in range(new_h2.shape[0]):
        r = np.random.random()
        if(r < mutation_rate):
            new_h2[i, 0] += np.random.uniform(-1, 1)
    return new_inp, new_h1, new_h2
    
def create_next_generation(inp_all, h1_all, h2_all, fitness, num_in_pop, cur_gen, mutation_rate):
    next_inp, next_h1, next_h2 = create_next_gen_snake(inp_all, h1_all, h2_all, fitness, num_in_pop, mutation_rate)
    for i in range(1, num_in_pop):
        new_inp, new_h1, new_h2 = create_next_gen_snake(inp_all, h1_all, h2_all, fitness, num_in_pop, mutation_rate)
        next_inp = np.concatenate((next_inp, new_inp), axis=1)
        next_h1 = np.concatenate((next_h1, new_h1), axis=1)
        next_h2 = np.concatenate((next_h2, new_h2), axis=1)
    np.save("data/raw_gen_" + str(cur_gen) + "_inp_data.npy", next_inp)
    np.save("data/raw_gen_" + str(cur_gen) + "_h1_data.npy", next_h1)
    np.save("data/raw_gen_" + str(cur_gen) + "_h2_data.npy", next_h2)
    
def compute_aver_gen_fitness(fitness, cur_gen):
    s = np.sum(fitness)
    aver = s / fitness.shape[1]
    arr = np.array([aver])
    np.save("fitness/fitness_gen_" + str(cur_gen) + "_aver.npy", arr)
    
    return aver
    
def go_through_gen(cur_gen, num_in_gen, inp_n, h1_n, h2_n, out_n, screen, clock, mutation_rate):
    total = []
    if(cur_gen == 1):
        inp_all, h1_all, h2_all = setup(num_in_gen, inp_n, h1_n, h2_n, out_n)
    else:
        inp_all, h1_all, h2_all = load_new_gen(cur_gen)
    for i in range(num_in_gen):
        cur_inp, cur_h1, cur_h2 = get_weights(inp_all, h1_all, h2_all, inp_n, h1_n, h2_n, out_n, i)
        NN = nn.neuralNetwork(inp_n, h1_n, h2_n, out_n, cur_inp, cur_h1, cur_h2)
        flag = game.play_game(NN, screen, clock, i, cur_gen, num_in_gen, mutation_rate)
        total.append(flag)
    
    fitness, summ, maximum = count_fitness(total, num_in_gen, cur_gen)
    create_next_generation(inp_all, h1_all, h2_all, fitness, num_in_gen, cur_gen, mutation_rate)
    #cur_fit = compute_aver_gen_fitness(fitness, cur_gen)
    
    return summ, maximum
    
    