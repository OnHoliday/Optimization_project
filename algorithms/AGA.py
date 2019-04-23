### ADD TO Adaptive_selection_GA.PY ###

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


class GeneticAlgorithm_AD(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size,
                 pressure, update_rate, crossover, p_c_max, p_c_min, mutation, p_m_max, index):
        # pressure = initial selection pressure
        # update_rate = constant to update selection pressure with
        RandomSearch.__init__(self, problem_instance, random_state)
        self.pressure = pressure
        self.update_rate = update_rate
        self.population_size = population_size
        self.crossover = crossover
        self.p_c_max = p_c_max
        self.p_c_min = p_c_min
        self.p_c = p_c_max
        self.mutation = mutation
        self.p_m_max = p_m_max
        self.p_m = p_m_max
        self.index = index

    def initialize(self, plot=True):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)
        if plot:
            self.initial_hist = self.best_solution.representation
        return self.initial_hist

    def search(self, n_iterations, min_iterations, max_pressure, report=False, log=False, plot=True):
        # min_iterations = iterations where pressure stays constant in the beginning
        # max_pressure = maximum selection pressure we are going up to
        geno_diversity = []
        pheno_diversity = []
        std = []
        fitness = []

        self.geno_shift = 0
        self.pheno_shift = []
        #self.geno_score = []
        self.fitness_list = []
        self.gen_mean = []
        self.gen_std = []
        self.df=pd.DataFrame()
        self.df2=pd.DataFrame()
        self.inertia_list = []
        self.param = 0
        self.iteration = -1

        # I didn't adjust the logger
        if log:
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__]
            logger = logging.getLogger(','.join(list(map(str, log_event))))

        elite = self.best_solution

        # for each iteration, check whether iteration is < min_iteration and either start updating pressure or not
        # if pressure exceeds max_pressure it should stay at max_pressure
        for iteration in range(n_iterations):
            # if iteration < min_iterations:
            #     self.pressure = self.pressure
            # else:
            #     self._update_pressure()
            # self._update_pressure()
            self.pressure += -self.geno_shift*0.04 +  0.0000003 * (np.power(self.iteration, 2))
            if self.pressure > max_pressure:
                self.pressure = max_pressure
            self.selection = uls.parametrized_tournament_selection(self.pressure)#2(self.pressure, 0.7)
            offsprings = []

            # if iteration == 70 or iteration == 80 or  iteration == 100:
            #     while len(offsprings) < len(self.population)-6:
            #         off1, off2 = p1, p2 = [
            #             self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in
            #             range(2)]
            #
            #         if self._random_state.uniform() < self.p_c:
            #             off1, off2 = self._crossover(p1, p2)
            #
            #         if self._random_state.uniform() < self.p_m:
            #             off1 = self._mutation(off1)
            #             off2 = self._mutation(off2)
            #
            #         if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
            #             self.problem_instance.evaluate(off1)
            #             self.problem_instance.evaluate(off2)
            #         offsprings.extend([off1, off2])
            #
            #     while len(offsprings) < len(self.population):
            #         offsprings.extend([self.generate_ranom_indiv()])
            #
            # else:
            if iteration % 150 == 0:
                while len(offsprings) < len(self.population) - 4:
                    off1, off2 = p1, p2 = [
                        self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in
                        range(2)]

                    if self._random_state.uniform() < self.p_c:
                        off1, off2 = self._crossover(p1, p2)

                    if self._random_state.uniform() < self.p_m:
                        off1 = self._mutation(off1)
                        off2 = self._mutation(off2)

                    if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                        self.problem_instance.evaluate(off1)
                        self.problem_instance.evaluate(off2)
                    offsprings.extend([off1, off2])

                while len(offsprings) < len(self.population):
                    offsprings.extend([self._generate_random_valid_solution() for i in range(4)])
            else:
                while len(offsprings) < len(self.population):
                    off1, off2 = p1, p2 = [
                        self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in
                        range(2)]

                    if self._random_state.uniform() < self.p_c:
                        off1, off2 = self._crossover(p1, p2)

                    if self._random_state.uniform() < self.p_m:
                        off1 = self._mutation(off1)
                        off2 = self._mutation(off2)

                    if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                        self.problem_instance.evaluate(off1)
                        self.problem_instance.evaluate(off2)
                    offsprings.extend([off1, off2])

            while len(offsprings) > len(self.population):
                offsprings.pop()

            self.pheno_shift.append(self._phenotypic_diversity_shift(offsprings))
            self.df2[iteration] = self._phenotypic_diversity(offsprings)
            # self.geno_shift.append(self._gen_pop_diversity_shift(offsprings))
            #self.geno_score.append(self._gen_pop_diversity(offsprings))
            self.gen_mean.append(self._gen_mean(offsprings))
            self.gen_std.append(self._std(offsprings))
            self.df[iteration]=self._reps(offsprings)
            #self.df2[iteration]=self._phenotypic_diversity(offsprings)
            self.iteration += 1
            self.inertia_list.append(self._gen_pop_diversity(offsprings))
            self.param = self.inertia_list[iteration]/self.inertia_list[0]

            if self.iteration > 1:
                self.geno_shift = (self.inertia_list[self.iteration] - self.inertia_list[self.iteration-1])/self.inertia_list[self.iteration-1]
            else: self.geno_shift = 0

            # print(self.geno_shift)

            if self.iteration > 1:
                self.geno_shift = (self.inertia_list[self.iteration] - self.inertia_list[self.iteration-1])/self.inertia_list[self.iteration-1]
            else: self.geno_shift = 0

            self.param = self.inertia_list[iteration] / self.inertia_list[0]
            self.p_c = self.param*(self.p_c_max-self.p_c_min)+self.p_c_min
            self.p_m -= self.geno_shift*0.1*self.p_m #(0.005 -  0.005 *(1-self.param)) #-(-((self.param*100-100)/100)*self.p_m_max)
            # self.pressure += -self.geno_shift * 0.05 + 0.0000003 * (np.power(self.iteration, 2))


            if self.p_m > 1:
                self.p_m = 1

            # print(self.pheno_shift[self.iteration])
            # print(self.geno_shift[self.iteration])

            elite_offspring = self._get_elite(offsprings)
            elite = self._get_best(elite, elite_offspring)

            if report:
                print("> > > > > > > > > > > >")
                print("P_C: %.2f" % self.p_c)
                print("P_M: %.2f" % self.p_m)
                print("Pool_Size: %.2f" % self.pressure)
                self._verbose_reporter_inner(elite, iteration)
                print("> > > > > > > > > > > >")


            if log:
                log_event = [iteration, elite.fitness,
                             elite.validation_fitness if hasattr(off2, 'validation_fitness') else None,
                             self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, self._random_state, None, self.p_m,
                             self._phenotypic_diversity_shift(offsprings),
                             self._gen_pop_diversity(self.population)], self._std(self.population)
                logger.info(','.join(list(map(str, log_event))))

            geno_diversity.append(self._gen_pop_diversity(offsprings))
            pheno_diversity.append(self._phenotypic_diversity_score(offsprings))
            std.append(self._std(offsprings))
            fitness.append(elite.fitness)

            self.population = offsprings

            # self._update_pressure()

        self.best_solution = elite
        if plot:
            gs = gridspec.GridSpec(3, 2)
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

            ax = plt.subplot(gs[0, 1])
            plt.title("Weight Histogram")
            plt.hist(self.initial_hist, alpha=0.5, label='initial_distribution')
            plt.hist(self.best_solution.representation, alpha=0.5, label='best_solution_distribution')
            fontP = FontProperties()
            fontP.set_size('small')
            pylab.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=1)

            ax = plt.subplot(gs[2, 1])
            plt.title("Weights in best solution")
            plt.plot(self.initial_hist, alpha=0.5, label='initial_distribution')
            plt.plot(self.best_solution.representation, alpha=0.5, label='best_solution_distribution')
            pylab.legend(prop=fontP, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=1)

            plt.tight_layout()
            plt.savefig('GA_stats_' + str(self.index) + '.png')
            # plt.show()

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
        return np.std(fitness_offsprings) - np.std(fitness_parents)

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

    def _gen_mean(self, population):
        solutions = []
        for each in population:
            solutions.append(each.representation)
        gen_mean = np.average(solutions)
        return gen_mean

    def _std(self, population):
        solutions = []
        for each in population:
            solutions.append(each.representation)
        gen_std = np.std(solutions)
        return gen_std

    def _reps(self, population):
        solution_list = []
        for each in population:
            for w in each.representation:
                solution_list.append(w)
        return solution_list

    def _geno_coef_var(self, population):
        solutions = []
        for each in population:
            solutions.append(each.representation)
        coef_var = (np.std(solutions)/np.average(solutions))*100
        return coef_var

    def _disper_index(self, population):
        solutions = []
        for each in population:
            solutions.append(each.representation)
        disper_index = (np.square(np.std(solutions))/np.average(solutions))
        return disper_index

    def _disper_quantile(self, population):
        solutions = []
        for each in population:
            solutions.append(each.representation)
        disper_quantile = (np.quantile(solutions, q=0.75)-np.quantile(solutions, q=0.25))/(np.quantile(solutions, q=0.75)+np.quantile(solutions, q=0.25))
        return

    def _update_pressure(self):
        self.pressure *= self.update_rate

    def _phenotypic_diversity(self, offsprings):
        pheno2 = []
        for each in offsprings:
            pheno2.append(each.fitness)
        return pheno2

    # def generate_ranom_indiv(self):
    #     solution = self._generate_random_valid_solution()
    #     return solution