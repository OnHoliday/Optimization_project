import logging
import numpy as np
from functools import reduce
from matplotlib.font_manager import FontProperties
from matplotlib import pylab
from random_search import RandomSearch
from solutions.solution import Solution
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.gridspec as gridspec
import utils as uls
from utils import parametrized_tournament_selection

class GeneticAlgorithm(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size,
                selection, crossover, p_c, mutation, p_m, index):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.selection = selection
        self.population_size = population_size
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.index = index

    def initialize(self,plot=True):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)
        if plot:
            self.initial_hist = self.best_solution.representation
            return self.initial_hist

    def search(self, n_iterations,report=False, log=False,plot=True):
        geno_diversity = []
        pheno_diversity = []
        std = []
        fitness = []

        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        elite = self.best_solution
        self.inertia_list = [0]
        self.iteration_list=[0]
        self.df2=pd.DataFrame()

        for iteration in range(n_iterations):
            offsprings = []
            offsprings.extend([elite])

            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state, self.inertia_list[iteration], self.iteration_list[iteration]) for _ in range(2)]


                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)

                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                    all = [p1, p2, off1, off2]
                    fitnesses = np.array([i.fitness for i in all])
                    indexes = fitnesses.argsort()[-2:][::-1]
                    off1 = all[indexes[0]]
                    off2 = all[indexes[1]]

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                offsprings.extend([off1, off2])


            while len(offsprings) > len(self.population):
                offsprings.pop()

            self.df2[iteration] = self._phenotypic_diversity(offsprings)
            elite_offspring = self._get_elite(offsprings)
            elite = self._get_best(elite, elite_offspring)

            if report:
                self._verbose_reporter_inner(elite, iteration)

            if log:
                log_event = [iteration, elite.fitness, elite.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, self._random_state, None, self.p_m, self._phenotypic_diversity_shift(offsprings), self._gen_pop_diversity(self.population)], self._std(self.population)
                logger.info(','.join(list(map(str, log_event))))

            geno_diversity.append(self._gen_pop_diversity(offsprings))
            pheno_diversity.append(self._phenotypic_diversity_score(offsprings))
            std.append(self._std(offsprings))
            fitness.append(elite.fitness)

            self.population = offsprings
            # self._update_pressure()

            self.inertia_list.append(self._gen_pop_diversity(offsprings))
            self.iteration_list.append(iteration)
        self.geno_list = geno_diversity
        self.pheno_list = pheno_diversity
        self.fitnesses = fitness
        self.best_solution = elite
        if plot:
            gs = gridspec.GridSpec(4, 2)
            plt.figure()
            ax = plt.subplot(gs[0, 0])
            plt.plot(geno_diversity)
            plt.title("genotypic diversity")

            ax = plt.subplot(gs[1, 0])
            plt.plot(pheno_diversity)
            plt.title("phenotypic diversity")

            ax = plt.subplot(gs[2, 0])
            plt.plot(fitness)
            plt.title("fitness")

            ax = plt.subplot(gs[0,1])
            plt.title("Weight Histogram")
            plt.hist(self.initial_hist, alpha=0.5, label='initial_distribution')
            plt.hist(self.best_solution.representation, alpha=0.5, label='best_solution_distribution')
            fontP = FontProperties()
            fontP.set_size('small')
            pylab.legend(prop=fontP,loc=9, bbox_to_anchor=(0.5, -0.1), ncol=1)

            ax = plt.subplot(gs[2,1])
            plt.title("Weights in best solution")
            plt.plot(self.initial_hist, alpha=0.5, label='initial_distribution')
            plt.plot(self.best_solution.representation, alpha=0.5, label='best_solution_distribution')
            pylab.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=1)

            ax = plt.subplot(gs[3,:])
            self.df2.boxplot()
            plt.tight_layout()
            plt.savefig('GA_stats_' + str(self.index) + '.png')
            plt.show()


    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def _mutation(self, individual):
        mutant = self.mutation(individual.representation, self._random_state)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _phenotypic_diversity_shift(self, offsprings):
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings)-np.std(fitness_parents)

    def _phenotypic_diversity_score(self, offsprings):
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings)

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions

    def _gen_pop_diversity(self, population):
        solutions = []
        for each in population:
            solutions.append(each.representation)
        to_scale = np.array(solutions)
        from sklearn.preprocessing import normalize
        scaled_solutions = normalize(to_scale, axis=0, norm='l2')
        centroid = np.sum(scaled_solutions, axis=0) / len(population)
        a = 0
        for each in scaled_solutions:
            a += np.sum(np.power(np.subtract(each, centroid), 2), axis=0)
            inertia = a * len(population)
        return inertia

    def _std(self,population):
        solutions = []
        for each in population:
            solutions.append(each.representation)
        gen_std=np.std(solutions)
        return gen_std

    def _phenotypic_diversity(self, offsprings):
        pheno2 = []
        for each in offsprings:
            pheno2.append(each.fitness)

        return pheno2
