import os
import datetime

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils as uls
from problems.ANNOP import ANNOP
from ANN.ANN import ANN, softmax, sigmoid
# from algorithms.genetic_algorithm import GeneticAlgorithm
# from algorithms.CURRENT_AGA import GeneticAlgorithm_AD
from algorithms.AGA_4p import GeneticAlgorithm_AD


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

make_plots = True

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
n_weights = X_train.shape[1]*hl1 + hl1*hl2 + hl2*len(digits.target_names)
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
#++++++++++++++++++++++++++

# GA PArameters
n_gen =
ps = 40
p_c_max = 1
p_c_min = 0.9
p_m_max = 1
radius = 0.5
pressure = 0.15
prob_of_mut = 0.75
update_rate = 1
max_pressure = 0.3
min_iterations = 0


ga2 = GeneticAlgorithm_AD(ann_op_i, random_state, ps, pressure, update_rate,
                      uls.crossover_weight, p_c_max, p_c_min, uls.parametrized_ball_mutation(radius), p_m_max, 'test')
ga2.initialize(plot=True)
ga2.search(n_gen, min_iterations, max_pressure, True, True)

# uls.save_object(ga2.best_solution.representation, 'C:\\Users\\Konrad\\Desktop\\NOVA IMS\\CIFO_HOMEWORK\\FINAL\\MASTER\\representation.pkl')

plt.show()

# Fitness Boxplot
sns.set_style('darkgrid')
ga2.df2.boxplot()
plt.xticks(np.arange(126, step = 5))
plt.xticks(size=18)
plt.yticks(size=18)
plt.title('Fitness distribution of best performing GA')
plt.show()

ga2.best_solution.print_()
print("Training fitness of the best solution: %.2f" % ga2.best_solution.fitness)


# #++++++++++++++++++++++++++
# # TEST
# #++++++++++++++++++++++++++
ann_i._set_weights(ga2.best_solution.representation)
y_pred = ann_i.stimulate_with(X_test, False)
print("Unseen Accuracy of the best solution: %.2f" % accuracy_score(y_test, y_pred))

if make_plots:
    n_images = 25
    images = X_test[0:n_images].reshape((n_images, 8, 8))
    f = plt.figure(figsize=(10, 10))
    for i in range(n_images):
        sub = f.add_subplot(5, 5, i + 1)
        sub.imshow(images[i], cmap=plt.get_cmap("Greens") if y_pred[i] == y_test[i] else plt.get_cmap("Reds"))
        plt.xticks([])
        plt.yticks([])
        sub.set_title('y^: %i, y: %i' % (y_pred[i], y_test[i]))
    f.suptitle('Testing classifier on unseen data')
    plt.show()

ann_i._set_weights(ga.best_solution.representation)
y_pred = ann_i.stimulate_with(X_test, False)
print("Unseen Accuracy of the best solution: %.2f" % accuracy_score(y_test, y_pred))
if make_plots:
    n_images = 25
    images = X_test[0:n_images].reshape((n_images, 8, 8))
    f = plt.figure(figsize=(10, 10))
    for i in range(n_images):
        sub = f.add_subplot(5, 5, i + 1)
        sub.imshow(images[i], cmap=plt.get_cmap("Greens") if y_pred[i] == y_test[i] else plt.get_cmap("Reds"))
        plt.xticks([])
        plt.yticks([])
        sub.set_title('y^: %i, y: %i' % (y_pred[i], y_test[i]))
    f.suptitle('Testing classifier on unseen data')
    plt.show()