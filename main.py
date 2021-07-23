# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:54:30 2021

@author: Victor Nobre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from activation_function import BinaryStep
from activation_function import SignFunction

datasets = pd.read_csv('datasets/dataset-treinamento.csv')
X = datasets.iloc[:,0:3].values
d = datasets.iloc[:,3:].values


p = Perceptron(X, d)
p.train()


datasets = pd.read_csv('datasets/dataset-teste.csv')
X = datasets.iloc[:,0:3].values

p = Perceptron(X, d)
p.testes()

for i in range(len(X)):
    if d[i] == -5:
        plt.plot(X[i, -1], X[i,0], 'ro')
    else:
        plt.plot(X[i, 0], X[i,1], 'ro')
x_plot = np.arange(-6, 6)
y_plot = list(map(lambda x: (1 * (p.W[0]/p.W[1])*x)+(p.theta/p.W[1]), x_plot ))
plt.plot(x_plot, y_plot)




