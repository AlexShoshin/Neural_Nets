# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:55:49 2019

@author: Алексей
"""

import numpy as np



def relu(x):
    return np.maximum(0, x)

class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes, cur_inp, cur_h1, cur_h2):
        self.inodes = input_nodes
        self.hnodes_1 = hidden_nodes_1
        self.hnodes_2 = hidden_nodes_2
        self.onodes = output_nodes
        
        
        self.wih = cur_inp
        self.whh = cur_h1
        self.who = cur_h2
        
        
    def query(self, inputs_list):
        #transforming incoming list to 2d array
        inputs = np.array(inputs_list, ndmin = 2).T
        
        hidden_inputs_1 = np.dot(self.wih, inputs)
        hidden_outputs_1 = relu(hidden_inputs_1)
        
        hidden_inputs_2 = np.dot(self.whh, hidden_outputs_1)
        hidden_outputs_2 = relu(hidden_inputs_2)
        
        output_inputs = np.dot(self.who, hidden_outputs_2)
        output_finals = np.exp(output_inputs) / np.sum(np.exp(output_inputs))
        
        return output_finals
    

    
