import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functools import reduce

def get_random_state(seed):
    return np.random.RandomState(seed)


def random_boolean_1D_array(length, random_state):
    return random_state.choice([True, False], length)


def bit_flip(bit_string, random_state):
    neighbour = bit_string.copy()
    index = random_state.randint(0, len(neighbour))
    neighbour[index] = not neighbour[index]

    return neighbour


def parametrized_iterative_bit_flip(prob):
    def iterative_bit_flip(bit_string, random_state):
        neighbor = bit_string.copy()
        for index in range(len(neighbor)):
            if random_state.uniform() < prob:
                neighbor[index] = not neighbor[index]
        return neighbor

    return iterative_bit_flip


def random_float_1D_array(hypercube, random_state):
    return np.array([random_state.uniform(tuple_[0], tuple_[1])
                     for tuple_ in hypercube])


def random_float_cbound_1D_array(dimensions, l_cbound, u_cbound, random_state):
    return random_state.uniform(lower=l_cbound, upper=u_cbound, size=dimensions)


def parametrized_ball_mutation(radius):
    def ball_mutation(point, random_state):
        return np.array(
            [random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])

    return ball_mutation

def parametrized_ball_mutation_dynamic(radius):
    def ball_mutation(point, random_state):
        radius2=radius*5
        if random_state.uniform()<0.75:
            return np.array(
            [random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])
        else:
            return np.array(
            [random_state.uniform(low=coordinate - radius2, high=coordinate + radius2) for coordinate in point])
    return ball_mutation


def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=0)#len(point.shape) % 2 - 1)


def rastrigin(point):
    a = len(point) * 10 if len(point.shape) <= 2 else point.shape[0] * 10
    return a + np.sum(point ** 2 - 10. * np.cos(np.pi * 2. * point), axis=0)


def one_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_]))
    off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    return off1_r, off2_r


def generate_cbound_hypervolume(dimensions, l_cbound, u_cbound):
    return [(l_cbound, u_cbound) for _ in range(dimensions)]


def parametrized_ann(ann_i):
    def ann_ff(weights):
        return ann_i.stimulate(weights)

    return ann_ff


def parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population) * pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return tournament_selection


def parametrized_tournament_selection_fitscale(pressure):
    def tournament_selection(population, minimization, random_state, inertia, iteration):
        tournament_pool_size = int(len(population) * pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)
        if inertia > 6000:
            if minimization:
                return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
            else:
                return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)
        else:
            if minimization:
                return reduce(lambda x, y: x if fitness_share(x, population, iteration) <= fitness_share(y, population, iteration) else y, tournament_pool)
            else:
                return reduce(lambda x, y: x if fitness_share(x, population, iteration) >= fitness_share(y, population, iteration) else y, tournament_pool)

    return tournament_selection


class Dplot():

    def background_plot(self, hypercube, function_):
        dim1_min = hypercube[0][0]
        dim1_max = hypercube[0][1]
        dim2_min = hypercube[1][0]
        dim2_max = hypercube[1][1]

        x0 = np.arange(dim1_min, dim1_max, 0.1)
        x1 = np.arange(dim2_min, dim2_max, 0.1)
        x0_grid, x1_grid = np.meshgrid(x0, x1)

        x = np.array([x0_grid, x1_grid])
        y_grid = function_(x)

        # plot
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_xlim(dim1_min, dim1_max)
        self.ax.set_ylim(dim2_min, dim2_max)
        self.ax.plot_surface(x0_grid, x1_grid, y_grid, rstride=1, cstride=1, color="green", alpha=0.15) # cmap=cm.coolwarm,

    def iterative_plot(self, points, z, best=None):
        col = "k" if best is None else np.where(z == best, 'r', 'k')
        size = 75 if best is None else np.where(z == best, 150, 75)
        self.scatter = self.ax.scatter(points[0], points[1], z, s=size, alpha=0.75, c=col)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.scatter.remove()


def parametrized_random_point_mutation(n):
    def random_point_mutation(point, random_state):
        positions = random_state.sample(range(0, len(point)),n)
        for p in positions:
            point[p] = random_state.uniform(np.argmin(point), np.argmax(point))
        return point
    return random_point_mutation

# define number of positions to apply ball mutation
def parametrized_random_point_ball_mutation(radius,n):
    def random_point_ball_mutation(point, random_state):
        positions = random_state.sample(range(0, len(point)),n)
        for p in positions:
            point[p] = np.array(
                    [random_state.uniform(low=point[p] - radius, high=point[p] + radius)])
        return point
    return random_point_ball_mutation

# define probability for each position to be mutated
def parametrized_percentage_ball_mutation(radius,prob):
    def random_percentage_ball_mutation(point, random_state):
        for p in range(len(point)):
            if random_state.uniform() < prob:
                point[p] = random_state.uniform(low=point[p] - radius, high=point[p] + radius)
        return point
    return random_percentage_ball_mutation


def roulette_wheel(population, minimization, random_state):
    fitness_cases = np.array([ind.fitness for ind in population])
    if minimization:
        pass
    else:
        total_fit = np.sum(fitness_cases)
        fitness_proportions = np.divide(fitness_cases, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return population[indexes[0][0]]


def pool_selection(population, minimization, random_state):
    fitness_cases = np.array([ind.fitness for ind in population])
    fitness_cases.sort()
    if minimization:
        pass
    else:
        fit2 = []
        for i in range(len(fitness_cases)):
            fit2 += list(np.repeat(fitness_cases[i], i))
        random_value = random_state.randint(len(fit2))
        for i in range(len(population)):
            if population[i].fitness == fit2[random_value]:
                return population[i]

def steady_selection(population, minimization, random_state):
    sel_df = pd.DataFrame(population)
    fitness_cases = np.array([ind.fitness for ind in population])
    sel_df['fit'] = fitness_cases
    leng = len(fitness_cases)
    # random_value = random_state.randint(leng)
    if minimization:
        pass
    else:
        sel_df = sel_df.sort_values('fit', ascending=False)
        sel_df.drop(sel_df.tail(np.floor(leng * 0.75).astype(int)).index, inplace=True)
        offspring = np.array(sel_df[0])
        fitness_cases = np.array([np.power(ind.fitness,2) for ind in offspring])
        total_fit = np.sum(fitness_cases)
        fitness_proportions = np.divide(fitness_cases, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return offspring[indexes[0][0]]
        # pop = roulette_wheel(offspring, True, random_state)
    # return roulette_wheel(offspring, True, random_state)
    # return np.random.choice(offspring)


def two_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    point = random_state.randint(len_)
    point2 = random_state.randint(len_)
    if point > point2:
        off1_r = np.concatenate((p1_r[0:point2], p2_r[point2:point],p1_r[point:len_]))
        off2_r = np.concatenate((p2_r[0:point2], p1_r[point2:point],p2_r[point:len_]))
    elif point <= point2:
        off1_r = np.concatenate((p1_r[0:point], p2_r[point:point2],p1_r[point2:len_]))
        off2_r = np.concatenate((p2_r[0:point], p1_r[point:point2],p2_r[point2:len_]))

    return off1_r, off2_r

def uniform_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    p_c  = 0.75
    off1_r = []
    off2_r = []
    for i in range(len_):
        if random_state.random_sample() >= p_c:
            off1_r.append(p1_r[i])
            off2_r.append(p2_r[i])
        else:
            off1_r.append(p2_r[i])
            off2_r.append(p1_r[i])

    return off1_r, off2_r

def generate_cbound_hypervolume(dimensions, l_cbound, u_cbound):
  return [(l_cbound, u_cbound) for _ in range(dimensions)]



def parametrized_uniform_crossover(prob):
    def uniform_crossover(p1_r, p2_r, random_state):
        len_ = len(p1_r)
        p_c = prob
        off1_r = []
        off2_r = []
        for i in range(len_):
            if random_state.random_sample() >= p_c:
                off1_r.append(p2_r[i])
                off2_r.append(p1_r[i])
            else:
                off1_r.append(p2_r[i])
                off2_r.append(p1_r[i])

        return off1_r, off2_r
    return uniform_crossover

def geometric_crossover_radcliff(p1_r, p2_r,p3_r, p4_r, random_state):
    len_ = len(p1_r)
    off_1 = []
    off_2 = []
    for i in range(len_):
        r = random_state.uniform()
        off_1.append(r*p1_r[i] + (1-r)*p2_r[i])
        r = random_state.uniform()
        off_2.append(r * p3_r[i] + (1 - r) * p4_r[i])
    return off_1, off_2

def geometric_crossover_michalewicz(p1_r, p2_r,p3_r, p4_r, random_state):
    len_ = len(p1_r)
    off_1 = []
    off_2 = []
    for i in range(len_):
        r = random_state.uniform()
        off_1.append(r*(p1_r[i] - p2_r[i]) + p1_r[i])
        r = random_state.uniform()
        off_2.append(r*(p4_r[i] - p3_r[i]) + p4_r[i])
    return off_1, off_2

def fitness_scaling(population):
    ind_reps=np.array([each.representation for each in population])
    a=np.ones((len(population), len(population)))
    for i in range(len(ind_reps)):
        for j in range(len(ind_reps)):
            a[i, j]=np.sum(np.power(np.subtract(ind_reps[i], ind_reps[j]), 2), axis=0)
    from sklearn.preprocessing import normalize
    scaled_a=normalize(a, axis=0, norm='max')
    reversed_a=1-scaled_a
    total_a=np.sum(reversed_a, axis=1)
    i=0
    new_fitness=[]
    for each in population:
        new_fitness.append(np.divide(each.fitness, total_a[i]))
        i+=1
    return new_fitness

def roulette_wheel_fitscale(population, minimization, random_state):
    fitness_cases=np.array(fitness_scaling(population))
    if minimization:
        pass
    else:
        total_fit = np.sum(fitness_cases)
        fitness_proportions = np.divide(fitness_cases, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return population[indexes[0][0]]


def three_point_crossover(p1_r, p2_r, random_state):
    len_ = len(p1_r)
    points = random_state.randint(len_,size = 3)
    points.sort()
    off1_r = np.concatenate((p1_r[0:points[0]], p2_r[points[0]:points[1]], p1_r[points[1]:points[2]], p1_r[points[2]:len_]))
    off2_r = np.concatenate((p2_r[0:points[0]], p1_r[points[0]:points[1]], p2_r[points[1]:points[2]], p2_r[points[2]:len_]))
    return off1_r, off2_r


def parametrized_uniform_crossover(prob):
    def uniform_crossover(p1_r, p2_r, random_state):
        len_ = len(p1_r)
        for i in range(len_):
            if random_state.random_sample() <= prob:
                p1_r = np.concatenate(
                    (p1_r[0:i], p2_r[i:len_]))
                p2_r = np.concatenate(
                    (p2_r[0:i], p1_r[i:len_]))
        off1_r = p1_r
        off2_r = p2_r
        return off1_r, off2_r
    return uniform_crossover

def crossover_weight(p1_r, p2_r,p3_r, p4_r, random_state):
    len_ = len(p1_r)
    off1_r = []
    off2_r = []
    for i in range(len_):
        if random_state.uniform() >= 0.5:
            off1_r.append(p1_r[i])
            off2_r.append(p3_r[i])
        else:
            off1_r.append(p2_r[i])
            off2_r.append(p4_r[i])
    return off1_r, off2_r

def geometric_crossover_ver2(p1_r, p2_r,p3_r, p4_r, random_state):
    len_ = len(p1_r)
    off_1 = []
    off_2 = []
    for i in range(len_):
        r = random_state.uniform()
        off_1.append(r*(p1_r[i] - p2_r[i]) + p1_r[i])
        r = random_state.uniform()
        off_2.append(r*(p4_r[i] - p3_r[i]) + p4_r[i])
    return off_1, off_2

def parametrized_tournament_selection2(pressure, tail = 0.65):
    def stady_selection_tournament(population, minimization, random_state):
        sel_df = pd.DataFrame(population)
        fitness_cases = np.array([ind.fitness for ind in population])
        sel_df['fit'] = fitness_cases
        leng = len(fitness_cases)
        # random_value = random_state.randint(leng)
        if minimization:
            pass
        else:
            sel_df = sel_df.sort_values('fit', ascending=False)
            sel_df.drop(sel_df.tail(np.floor(leng * tail).astype(int)).index, inplace=True)
            offspring = np.array(sel_df[0])
            tournament_pool_size = int(len(population) * pressure)
            tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=True)
            if minimization:
                return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
            else:
                return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return stady_selection_tournament

def linearScaling(population):
    fitness_raw = np.array([ind.fitness for ind in population])
    a = max(fitness_raw)
    b = -min(fitness_raw)/np.std(fitness_raw)*2
    fitness_scaled = np.array([a + ind.fitness*b for ind in population])
    return fitness_scaled

def parametrizedTopScalling(preassure):
    def topScaling(population):
        sel_df = pd.DataFrame(population)
        fitness_raw = np.array([ind.fitness for ind in population])
        sel_df['fit'] = fitness_raw
        leng = len(fitness_raw)
        sel_df = sel_df.sort_values('fit', ascending=False)
        for i in range(leng):
            if i < np.floor(preassure*leng):
                sel_df['fit'][i] = 1
            else: sel_df['fit'][i] = 0
        fitness_scaled = np.array(sel_df[0])
        return fitness_scaled
    return topScaling

def sigmaScaling(population):
    fitness_raw = np.array([ind.fitness for ind in population])
    avg_fit = np.average(fitness_raw)
    std_fit = np.std(fitness_raw)
    fitness_scaled = np.array([1 + ind.fitness - avg_fit/2*std_fit for ind in population])
    return fitness_scaled


def BoltzmanSelection(population, minimization, random_state):
    fitness_raw = np.array([ind.fitness for ind in population])
    control_parameter = 10
    update_rate = 0.9
    avg_fit = np.average(fitness_raw)

    for ind in range(len(fitness_raw)):
        fitness_raw[ind] = np.exp(fitness_raw[ind]/ control_parameter)/ np.exp(avg_fit/ control_parameter)
    control_parameter *= update_rate
    if minimization:
        pass
    else:
        total_fit = np.sum(fitness_raw)
        fitness_proportions = np.divide(fitness_raw, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return population[indexes[0][0]]

def inner_innertia(indiv):
    ind_reps = indiv.representation
    a = np.ones((len(ind_reps), len(ind_reps)))
    for i in range(len(ind_reps)):
            a[i] = np.sum(np.power(np.subtract(ind_reps[i], 0), 2), axis=0)
    from sklearn.preprocessing import normalize
    inner_innertia = np.sum(a)
    return inner_innertia

def eval_func(indiv):
    indiv.innertia = inner_innertia(indiv)/50000000
    new_eval = np.sum(indiv.fitness + indiv.innertia)
    return new_eval

def parametrized_tournament_selection_new_eval(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)

        if minimization:
            return reduce(lambda x, y: x if eval_func(x) <= eval_func(y) else y, tournament_pool)
        else:
            # print('a')
            # print(eval_func(tournament_pool[0]))
            # print(tournament_pool[0].fitness)
            # print(tournament_pool[0].innertia)
            # print('b')
            return reduce(lambda x, y: x if eval_func(x) >= eval_func(y) else y, tournament_pool)

    return tournament_selection

def random_crossover_method(p1_r, p2_r, random_state):
    p = random_state.random_sample()
    if p <= 0.25:
        off1_r, off2_r = one_point_crossover(p1_r, p2_r, random_state)
    elif p <= 0.5:
        off1_r, off2_r = two_point_crossover(p1_r, p2_r, random_state)
    elif p <= 0.75:
        off1_r, off2_r = uniform_crossover(p1_r, p2_r, random_state)
    else:
        off1_r, off2_r = three_point_crossover(p1_r, p2_r, random_state)
    return off1_r, off2_r



def roulette_wheel_fitscale(population, minimization, random_state):
    iteration=0
    iteration = counter(iteration)
    fitness_cases = fitness_scaling2(population, iteration)#np.array([ind.fitness for ind in population])
    if minimization:
        pass
    else:
        total_fit = np.sum(fitness_cases)
        fitness_proportions = np.divide(fitness_cases, total_fit)
        cumulative_fitness_proportion = np.cumsum(fitness_proportions)
        random_value = random_state.uniform()
        indexes = np.argwhere(cumulative_fitness_proportion >= random_value)
        return population[indexes[0][0]]


def fitness_scaling2(population, iteration):
    radius=20
    scaler=10+0.2*iteration
    alpha=1
    ind_reps=np.array([each.representation for each in population])
    a=np.ones((len(population), len(population)))
    for i in range(len(ind_reps)):
        for j in range(len(ind_reps)):
            a[i, j] = np.sum(np.power(np.subtract(ind_reps[i], ind_reps[j]), 2), axis=0)
            if a[i,j] < radius:
                a[i, j]=1-np.power((np.divide(a[i,j], radius)),alpha)
            else:
                a[i,j]=0
    # from sklearn.preprocessing import normalize
    # #scaled_a=normalize(a, axis=0, norm='max')
    #
    # total_a=np.sum(reversed_a, axis=1)
    total_a=np.sum(a, axis=1)
    i=0
    new_fitness=[]
    for each in population:
        new_fitness.append(np.divide(np.power(each.fitness, scaler), total_a[i]))
        i+=1
    return new_fitness


def parametrized_creep_mutation(prob):
    def creep_mutation(point, random_state):
        for p in range(len(point)):
            if random_state.uniform() < prob:
                point[p] = random_state.uniform(min(point), max(point))
        return point
    return creep_mutation

# def parametrized_swap_mutation(length):
#     def swap_mutation(point,random_state):
#         positions = random_state.randint(0,len(point)-length,2)
#         for i in range(0,length):
#             point[positions[0]+i], point[positions[1]+i] = point[positions[1]+i],point[positions[0]+i]
#         return point
#     return swap_mutation

def swap_mutation(point,random_state):
    length = random_state.randint(0,100)
    print(length)
    start1 = random_state.randint(0,(len(point)-length -1))
    print(start1)
    end1 = start1 + length
    print(end1)
    if start1 > length:
        if (len(point) - (end1+1)) > length:
            start2 = random_state.choice([random_state.randint(0,start1-length),random_state.randint((end1+1),(len(point))-length)])
        else:
            start2 = random_state.randint(0,start1-length)
    else:
        start2 = random_state.randint((end1+1),len(point)-length)
    print(start2)
    for i in range(0,length):
        point[start1+i], point[start2+i] = point[start2+i],point[start1+i]
    return point

def random_shuffle_mutation(point,random_state):
    start = random_state.randint(0,len(point)-10)
    end = random_state.randint(start,start+10)

    new_string = [point[i] for i in range(start,end)]
    random_state.shuffle(new_string)
    for i in range(start,end):
        pointer = 0
        point[i] = new_string[pointer]
        pointer+=1
    return point


def random_shuffle_mutation_v2(point,random_state):
    #random string length
    start = random_state.randint(0,len(point))
    end = random_state.randint(start,len(point))

    new_string = [point[i] for i in range(start,end)]
    random_state.shuffle(new_string)
    for i in range(start,end):
        pointer = 0
        point[i] = new_string[pointer]
        pointer+=1
    return point

def inversion_mutation(point,random_state):
    start = random_state.randint(0, len(point))
    end = random_state.randint(start, len(point))
    point[start:end].reverse()

    return point

def _calculate_new_fitness(indiv, population):
    distances = [np.linalg.norm(np.array(indiv.representation) - np.array(solution.representation)) for solution in population]
    normalized_distances = distances / np.linalg.norm(distances)
    normalized_distances = [1 - dist for dist in normalized_distances]
    coefficient = np.sum(normalized_distances)
    indiv.custom_fitness = indiv.fitness / coefficient
    return indiv.fitness / coefficient

def parametrized_tournament_selection_ver3(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population) * pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)
        if minimization:
            return reduce(lambda x, y: x if _calculate_new_fitness(x, population) <= _calculate_new_fitness(y, population) else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if _calculate_new_fitness(x, population) >= _calculate_new_fitness(y, population) else y, tournament_pool)

    return tournament_selection

def fitness_share(indiv, population, iteration):
    alpha=1+0.1*iteration
    distances = [np.linalg.norm(np.array(indiv.representation) - np.array(solution.representation)) for solution in population]
    normalized_distances = distances / np.linalg.norm(distances)
    normalized_distances = [1 - dist for dist in normalized_distances]
    coefficient = np.sum(normalized_distances)
    indiv.custom_fitness = indiv.fitness / coefficient
    return np.power(indiv.fitness, alpha)/ coefficient

def shuffle(p1_r, random_state):
    r = np.arange(840)
    random_state.shuffle(r)
    off_1 = []
    for i in r:
        off_1.append(p1_r[i])
    return r, off_1

def cycle_crossover(p1_r, p2_r, random_state):
    ordered_list = np.arange(840)
    ordered = p2_r
    shuf_list, shuffled = shuffle(p1_r, random_state)
    position = 0
    index = 0
    index_list = []
    off_1 =  np.zeros(840)
    off_2 = np.zeros(840)
    while shuf_list[position] != 0:
        off_1[position] = shuffled[index]
        index_list.append(index)
        position = shuf_list[index]
        index = ordered_list[position]
    for w in index_list:
        off_2[w] = p1_r[w]
    for q in range(len(off_1)):
        if off_1[q] == 0:
            off_1[q] = p1_r[q]
        if off_2[q] == 0:
            off_2[q] = p2_r[q]
    a = 0
    off_1_reordered = np.zeros(840)
    for i in shuf_list:
        off_1_reordered[a] = off_1[i]
        a += 1
    return off_1_reordered, off_2

def parametrized_reverse_sign_mutation(prob):
    def reverse_sign_mutation(point, random_state):
        for p in range(len(point)):
            if random_state.uniform() < prob:
                point[p] = (-1) * point[p]
        return point
    return reverse_sign_mutation


def reverse_sign_mutation2(point, random_state):
    for p in (random_state.randint(0, len(point), 50)):
        point[p] = (-1) * point[p]
    return point

def parametrized_nNode_mutation(n):
    def mutate_node(point,random_state):
        for i in range(n):
            node = random_state.randint(0, 30)
            indexes = []
            if node <= 10:
                for i in range(64):
                    indexes.append(node + i*10)
            else:
                for i in range(10):
                    indexes.append( 640 + (node+ i*10 ))
            for j in indexes:
                new_weight = random_state.normal(0, 0.5)
                point[j] = new_weight
        return point
    return mutate_node

def crossover_nodes(p1_r, p2_r, random_state):
    off_1, off_2 = p1_r, p2_r
    for node in range(30):
        indexes = []
        if node <= 10:
            for i in range(64):
                indexes.append(node + i * 10)
            if random_state.uniform() >= 0.5:
                for j in indexes:
                    off_1[j] = p2_r[j]
                    off_2[j] = p1_r[j]
            else:
                for j in indexes:
                    off_1[j] = p1_r[j]
                    off_2[j] = p2_r[j]
        else:
            for i in range(10):
                indexes.append(640 + (node + i*10 ))
            if random_state.uniform() >= 0.5:
                for j in indexes:
                    off_1[j] = p2_r[j]
                    off_2[j] = p1_r[j]
            else:
                for j in indexes:
                    off_1[j] = p1_r[j]
                    off_2[j] = p2_r[j]
    return off_1, off_2

import pickle

def save_object(representation, fullpath):
    with open(fullpath, 'wb') as output:
        pickle.dump(representation, output, pickle.HIGHEST_PROTOCOL)