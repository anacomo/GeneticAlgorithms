from functools import partial
from collections import namedtuple
from math import log2, ceil
from typing import List, Callable, Tuple
from random import choices, randint, randrange, random, uniform
from copy import deepcopy
from matplotlib import pyplot
from numpy import array, arange

# ? type aliases

Chromosome = List[int]
Population = List[Chromosome]
FitnessFunction = Callable[[Chromosome], int]
PopulateFunction = Callable[[], Population]
SelectionFunction = Callable[[Population, FitnessFunction], Tuple[Chromosome, Chromosome]]
CrossoverFunc = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]
MutationFunc = Callable[[Chromosome], Chromosome]

Parameters = Tuple[int, int, int]
Bounds = Tuple[int, int]


def generate_chromosome(bounds: Bounds, length: int, precision: int) -> Chromosome:
    float_chromosome = round(uniform(bounds[0], bounds[1]), precision)
    # print("the number is ", float_chromosome)
    float_chromosome *= 10 ** precision
    float_chromosome = int(float_chromosome)

    binary_chromosome = format(float_chromosome, "0" + str(length - 1) +"b")
    if binary_chromosome[0] == '-':
        list_chromosome = [-1]
    else:
        list_chromosome = [1]
    if binary_chromosome[0] == '-':
        binary_chromosome = '0' + binary_chromosome[1 : ]
    list_chromosome += [int(x) for x in binary_chromosome]
    # print(list_chromosome)
    return list_chromosome

# uses dimension and generate_chromosome parameters
def generate_population(dimension: int, bounds: Bounds, length: int, precision: int) -> Population:
    return [generate_chromosome(bounds, length, precision) for _ in range(dimension)]

# ? ugly function make it more readable
def chromosome_to_float(list_chromosome: Chromosome, precision: int) -> float:
    result = 0
    copy = deepcopy(list_chromosome)
    sign = copy[0]
    del copy[0]
    ln = len(copy)
    copy.reverse()
    for i in range(ln):
        result += copy[i] * 2 ** i
    result *= sign
    result = result / (10 ** precision)
    #print("float number is", f)
    return round(result, precision)

def apply_function(parameters: Parameters, x: float) -> float:
    return parameters[0] * (x ** 2) + parameters[1] * x + parameters[2]

# uses chromosome, parameters of the functiohn and bounds
def fitness(list_chromosome: Chromosome, parameters: Parameters, bounds: Bounds, precision: int) -> float:
    nr = chromosome_to_float(list_chromosome, precision)
    if nr < bounds[0] or nr > bounds[1]:
        return 0
    return apply_function(parameters, nr)
    # parameters[0] * (nr ** 2) + parameters[1] * nr + parameters[2]
    

def selection_pair(population: Population, fitness_function: FitnessFunction) -> Population:
    return choices(
        population = population,
        weights = [fitness_function(chromosome) for chromosome in population],
        k = 2
    )

def single_point_crossover(a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
    if len(a) != len(b):
        print(chromosome_to_float(a, 2), chromosome_to_float(b, 2))
        print(a, b)
        raise ValueError("Chromosomes a and b shoulf be the same length")

    length = len(a)
    if length < 2:
        return a, b
    point = randint(1, length - 1)
    return a[0 : point] + b[point : ], b[0 : point] + a[point : ]

def mutation(chromosome: Chromosome, probability: float) -> Chromosome:
    index = randrange(len(chromosome))
    chromosome[index] = chromosome[index] if random() > probability else abs(chromosome[index] - 1)
    return chromosome

def calculate_crossover_probability(chromosome: Chromosome, population: Population, parameters: Parameters, bounds: Bounds, precision: int) -> float:
    return fitness(chromosome) / sum([fitness(item, parameters, bounds, precision) for item in population])

def show_function(x_axis: List[float], y_axis: List[float], bounds: Bounds, parameters: Parameters, precision: int):
    x = arange(bounds[0], bounds[1], 10 ** (-precision))
    y = [apply_function(parameters, i) for i in x]
    pyplot.plot(x, y)
    x1 = array(x_axis)
    y1 = array(y_axis)
    pyplot.scatter(x1, y1)
    pyplot.show()

# ? uses populate function, fitness function, selection function, mutation function 
def evolution( 
    populate_function: PopulateFunction,
    fitness_function: FitnessFunction,
    show_function,
    selection_function: SelectionFunction = selection_pair,
    crossover_function: CrossoverFunc = single_point_crossover,
    mutation_function: MutationFunc = mutation,
    mutation_probability: float = 0,
    generation_limit: int = 100
) -> Population:
    population = populate_function()

    x = []
    y = []

    for i in range(generation_limit):
        population = sorted(population, key = lambda chromosome: fitness_function(chromosome), reverse = True)
        next_generation = population[0 : int(5 / 100 * len(population))] # elitistic selection
        for j in range(int(len(population) / 2) - 1):
            parents = selection_function(population, fitness_function)
            offspring_a, offspring_b = crossover_function(parents[0], parents[1])
            offspring_a, offspring_b = mutation_function(offspring_a, mutation_probability), mutation_function(offspring_b, mutation_probability)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
        x.append(chromosome_to_float(population[0], 2)) #todo NU UITA PRECIZIA
        y.append(fitness_function(population[0]))
        #print("The best solution is ", chromosome_to_float(population[0], 10))
    show_function(x, y)
    print(x, y)
    return population


def main():
    dimension : int = 50
    bounds : Bounds = (2, 10)
    parameters : Parameters = (2, 1, 3)
    precision : int = 2
    mutation_probability : float = 0.9
    nr_generations : int = 50
    length = 2 + ceil(log2(max(abs(bounds[0] * 10 ** precision), abs(bounds[1] * 10 ** precision))))
    # print(generate_chromosome(bounds, length, precision))
    population = generate_population(dimension, bounds, length, precision)
    #print(population)
    #print([fitness(item, parameters, bounds, precision) for item in population])
    # show_function(bounds, parameters, precision)
    final_population = evolution(
        populate_function = partial(generate_population, dimension = dimension, bounds = bounds, length = length, precision = precision), 
        fitness_function = partial(fitness, parameters = parameters, bounds = bounds, precision = precision),
        show_function = partial(show_function, bounds = bounds, parameters = parameters, precision = precision),
        selection_function = selection_pair,
        crossover_function = single_point_crossover,
        mutation_function = mutation,
        mutation_probability = mutation_probability,
        generation_limit = nr_generations
        )
    #print(final_population)


if __name__ == "__main__":
    main()