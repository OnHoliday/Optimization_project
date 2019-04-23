import logging
import numpy as np
from functools import reduce
import random
from operator import itemgetter, attrgetter, methodcaller
from random_search import RandomSearch
from genetic_algorithm import GeneticAlgorithm
import utils as uls
from solutions.solution import Solution



def crossover_polimorphism(sub_algorithms, main_ga, random_state, n_gen_subpops, n_gen, crossover):
    # initialize search algorithms
    [algorithm.initialize() for algorithm in sub_algorithms]
    main_ga.initialize()
    # execute search
    [algorithm.search(n_iterations=n_gen_subpops, report=True) for algorithm in sub_algorithms]
    subpop_solutions = [algorithm.best_solution for algorithm in sub_algorithms]
    initial_pop = subpop_solutions

    while len(initial_pop) < main_ga.population_size:

        positions = random_state.randint(low = 0, high = len(subpop_solutions),
        size = 2)
        off1, off2 = crossover(subpop_solutions[positions[0]].representation,
                                             subpop_solutions[positions[1]].representation, random_state)
        off1 = Solution(off1)
        off2 = Solution(off2)
        main_ga.problem_instance.evaluate(off1)
        main_ga.problem_instance.evaluate(off2)
        initial_pop.append(off1)
        initial_pop.append(off2)
    while len(initial_pop) > main_ga.population_size:
        initial_pop.pop()

    main_ga.population = initial_pop
    main_ga.best_solution = main_ga._get_elite(main_ga.population)

    main_ga.search(n_gen, report=True, log=True)



def neighborhood_polimorphism(sub_algorithms, main_ga, random_state, n_gen_subpops, n_gen, radius):
    # initialize search algorithms
    [algorithm.initialize() for algorithm in sub_algorithms]
    main_ga.initialize()
    # execute search
    [algorithm.search(n_iterations=n_gen_subpops, report=True) for algorithm in sub_algorithms]
    subpop_solutions = [algorithm.best_solution for algorithm in sub_algorithms]
    fitnesses = [solution.fitness for solution in subpop_solutions]
    best_of_subpop = subpop_solutions[np.argmax(fitnesses)]

    initial_pop = [best_of_subpop]
    while len(initial_pop) < main_ga.population_size:
        new_individual = uls.ball_mutation(best_of_subpop.representation, radius, random_state)
        new_individual = Solution(new_individual)
        main_ga.problem_instance.evaluate(new_individual)
        initial_pop.append(new_individual)


    while len(initial_pop) > main_ga.population_size:
        initial_pop.pop()

    main_ga.population = initial_pop
    main_ga.best_solution = main_ga._get_elite(main_ga.population)

    main_ga.search(n_gen, report=True, log=True)


def mutation_polimorphism(sub_algorithms, main_ga, random_state, n_gen_subpops, n_gen, mutation):
    # initialize search algorithms
    [algorithm.initialize() for algorithm in sub_algorithms]
    main_ga.initialize()
    # execute search
    [algorithm.search(n_iterations=n_gen_subpops, report=True) for algorithm in sub_algorithms]
    all_individuals = []
    for algorithm in sub_algorithms:
        all_individuals.extend([solution for solution in algorithm.population])
    subpop_solutions = [algorithm.best_solution for algorithm in sub_algorithms]
    initial_pop = subpop_solutions
    print(len(all_individuals))
    while len(initial_pop) < main_ga.population_size:
        position = random.sample(range(0, len(all_individuals)), 1)
        #position = random_state.randint(0,len(all_individuals),1).astype(int)
        off1 = mutation(all_individuals[position[0]].representation, random_state)
        off1 = Solution(off1)
        main_ga.problem_instance.evaluate(off1)
        initial_pop.append(off1)

    main_ga.population = initial_pop
    main_ga.best_solution = main_ga._get_elite(main_ga.population)

    main_ga.search(n_gen, report=True, log=True)


def truncation_mutation_polimorphism(sub_algorithms, main_ga, random_state, n_gen_subpops, n_gen, mutation):
    # initialize search algorithms
    [algorithm.initialize() for algorithm in sub_algorithms]
    main_ga.initialize()
    # execute search
    [algorithm.search(n_iterations=n_gen_subpops, report=True) for algorithm in sub_algorithms]
    all_individuals = []
    for algorithm in sub_algorithms:
        all_individuals.extend({'solution':solution,'fitness':solution.fitness} for solution in algorithm.population)
    all_individuals_sorted = sorted(all_individuals, key=itemgetter('fitness'),reverse=True)
    chosen_individuals = all_individuals_sorted[0:int(len(all_individuals_sorted)*0.4)]
    subpop_solutions = [algorithm.best_solution for algorithm in sub_algorithms]
    all_individuals = [i['solution'] for i in chosen_individuals]
    initial_pop = subpop_solutions
    while len(initial_pop) < main_ga.population_size:
        position = random.sample(range(0, len(all_individuals)), 1)
        #position = random_state.randint(0,len(all_individuals),1).astype(int)
        off1 = mutation(all_individuals[position[0]].representation, random_state)
        off1 = Solution(off1)
        main_ga.problem_instance.evaluate(off1)
        initial_pop.append(off1)

    main_ga.population = initial_pop
    main_ga.best_solution = main_ga._get_elite(main_ga.population)

    main_ga.search(n_gen, report=True, log=True)

