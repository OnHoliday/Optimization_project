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
from algorithms.genetic_algorithm import GeneticAlgorithm
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

make_plots = False

if make_plots:
    n_images = 25
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(digits.images[i], cmap=plt.cm.binary)
        plt.xlabel("Value: %d" % digits.target_names[digits.target[i]], fontsize=12)
    plt.suptitle('Example of the training data',  fontsize=30)
    plt.show()

# setup random state
seed = 0
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
    # THE SEARCH
    # restrictions:
    # - 5000 offsprings/run max*
    # - 50 offsprings/generation max*
    # - use at least 5 runs for your benchmarks
    # * including reproduction
    #++++++++++++++++++++++++++

import polimorphism as poli
from polimorphism import neighborhood_polimorphism,crossover_polimorphism,truncation_mutation_polimorphism
# define parameters for the subpopulations and the main population
seed = 5
random_state = uls.get_random_state(seed)
p_c = 0.95
p_m = 0.95
evaluations_subpop = 2000
selection_subpop = uls.parametrized_tournament_selection(0.35)
selection_main = uls.parametrized_tournament_selection(0.35)
mutation = uls.parametrized_percentage_ball_mutation(0.5,0.15)
crossover = uls.two_point_crossover
ps_subpop = 40
n_subpops = 2
#Make sure that all initializations are counted into the fitness evaluation limitation:
n_gen_subpops = np.floor((evaluations_subpop-((n_subpops-1)*ps_subpop))/n_subpops/ps_subpop).astype(int)

ps_main = 40
#deduct the initialization from the available evaluations
n_gen = np.floor((5000-evaluations_subpop-ps_main)/ps_main).astype(int)

# define the subpop GA's here. If you want more, add another ga3,ga4..
ga1 = GeneticAlgorithm(ann_op_i, random_state, ps_subpop, selection_subpop,
                   crossover, p_c, mutation, p_m)
ga2 = GeneticAlgorithm(ann_op_i, random_state, ps_subpop, selection_subpop,
                   crossover, p_c, mutation, p_m)

# all subpop GA's need to be added to this list
search_algorithms = [ga1, ga2]

# define main GA
main_ga = GeneticAlgorithm(ann_op_i, random_state, ps_main, selection_main,
                   crossover, p_c, mutation, p_m)

# define the polimorphism method
truncation_mutation_polimorphism(search_algorithms, main_ga, random_state, n_gen_subpops, n_gen,mutation)
crossover_polimorphism(search_algorithms, main_ga, random_state, n_gen_subpops, n_gen,uls.one_point_crossover)




#++++++++++++++++++++++++++
# TEST
# - test algorithms on unseen data
#++++++++++++++++++++++++++

# ann_i._set_weights(poli.best_solution.representation)
# y_pred = ann_i.stimulate_with(X_test, False)
# accuracy = accuracy_score(y_test, y_pred)
#
# print("Unseen Accuracy of %s: %.2f" % (poli.__class__, accuracy_score(y_test, y_pred)))