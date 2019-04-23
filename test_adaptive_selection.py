import os
import datetime

import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils as uls
from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
from algorithms.adaptive_selection_GA import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing


# setup logger
file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "LogFiles/" + (str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) + "_log.csv"))
logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')

#++++++++++++++++++++++++++
# THE DATA
# restrictions:
# - MNIST digits (8*8)
# - 33% for testing
# - flattened input images
#++++++++++++++++++++++++++
# import data
digits = datasets.load_digits()
flat_images = np.array([image.flatten() for image in digits.images])

print(flat_images.shape)
print(digits.target_names)

# setup random state
seed = 1
random_state = uls.get_random_state(seed)

# split data
X_train, X_test, y_train, y_test = train_test_split(flat_images, digits.target, test_size=0.33, random_state=random_state)

#++++++++++++++++++++++++++
# THE ANN
# restrictions:
# - 2 h.l. with sigmoid a.f.
# - softmax a.f. at output
# - 20% for validation
#++++++++++++++++++++++++++
# ann's ingridients
hl1 = 10
hl2 = 10
hidden_architecture = np.array([[hl1, sigmoid], [hl2, sigmoid]])
n_weights = X_train.shape[1]*hl1+hl1*hl2+hl2*len(digits.target_names)
validation_p = 0.2
# create ann
ann_i = ANN(hidden_architecture, softmax, accuracy_score,
                   (X_train, y_train), random_state, validation_p, digits.target_names)

#++++++++++++++++++++++++++
# THE PROBLEM INSTANCE
#++++++++++++++++++++++++++
validation_threshold = 0.07
ann_op_i = ANNOP(search_space=(-2, 2, n_weights), fitness_function=ann_i.stimulate,
                 minimization=False, validation_threshold=validation_threshold)

#++++++++++++++++++++++++++
# THE OPTIMIZATION
# restrictions:
# - 5000 f.e./run
# - 50 f.e./generation
# - use at least 5 runs for benchmarks
#+++++++++++++++++++++++++

seed = 0
random_state = uls.get_random_state(seed)
ps = 40
n_gen = np.floor(5000/ps).astype(int)
p_c = 0.95
p_m = 0.95
radius = 0.5
pressure = 0.15
update_rate = 1.01
max_pressure = 0.3
min_iterations = 20


ga = GeneticAlgorithm(ann_op_i, random_state, ps, pressure, update_rate,
                       uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m,'test')
ga.initialize(plot=True)
ga.search(n_gen, min_iterations,max_pressure, True, True, plot=True)

print("Training fitness of the best solution: %.2f" % ga.best_solution.fitness)
print("Validation fitness of the best solution: %.2f" % ga.best_solution.validation_fitness)

#++++++++++++++++++++++++++
# TEST
#++++++++++++++++++++++++++
ann_i._set_weights(ga.best_solution.representation)
y_pred = ann_i.stimulate_with(X_test, False)
print("Unseen Accuracy of the best solution: %.2f" % accuracy_score(y_test, y_pred))


# ### PARALLEL ###
# import itertools
# import pandas as pd
# import joblib
# from joblib import Parallel,delayed
#
# ps = [40]
# radius = [0.5]
# n_gen = [np.floor(5000/40).astype(int)]
# p_m = [0.9,0.95,0.99]
# p_c = [0.9,0.95,0.99]
# seed = [5,6]
# pressure = [0.1,0.15,0.2]
# update_rate = [1.01,1.02,1.05]
# min_iterations = [15,20,25]
# max_pressure = [0.30,0.35,0.42]
# values = [p_m, p_c, seed, pressure, update_rate, min_iterations,max_pressure, radius, n_gen, ps]
# combinations = list(itertools.product(*values))
#
# #combinations = [[0.9,0.95,1,0.2,1.03,20,0.40,0.5,125,40],[0.95,0.95,1,0.25,1.03,20,0.40,0.5,125,40],[0.95,0.95,1,0.25,1.03,20,0.40,0.5,125,50]]
#
# lista = []
# df = pd.DataFrame()
#
# def eval(comb, index,len):
#     seed = comb[2]
#     ps = comb[9]
#     n_gen = comb[8]
#     radius = comb[7]
#     p_c = comb[1]
#     p_m = comb[0]
#     random_state = uls.get_random_state(comb[2])
#     pressure = comb[3]
#     update_rate = comb[4]
#     min_iterations = comb[5]
#     max_pressure = comb[6]
#     print('Current position: {} out of {}'.format(index, (len - 1)))
#     ga = GeneticAlgorithm(ann_op_i, random_state, ps, pressure, update_rate, uls.two_point_crossover, p_c, uls.parametrized_ball_mutation(radius), p_m,'test')
#     ga.initialize(plot=False)
#     ga.search(n_gen, min_iterations,max_pressure, False, False, False)
#
#     ann_i._set_weights(ga.best_solution.representation)
#     y_pred = ann_i.stimulate_with(X_test, False)
#     unseen_fitness = accuracy_score(y_test, y_pred)
#
#     lista=[ps, radius,p_m, p_c, seed, pressure, update_rate, min_iterations,max_pressure,ga.best_solution.fitness, unseen_fitness, ga.pheno_list,ga.geno_list,ga.fitnesses]
#     return lista
#
#
# a=Parallel(n_jobs=-1)(delayed(eval)(combination, combinations.index(combination),len(combinations)) for combination in combinations)
#
# df=pd.DataFrame(a, columns=['ps','radius','p_m', 'p_c','seed','pressure','update_rate','min_iterations','max_pressure','fitness','unseen fitness','phenotypic diversity','genotypic diversity','best solution fitnesses'])
# df.to_csv("Adaptive_Selection_seedX.csv", encoding='utf-8', index=True)
#
