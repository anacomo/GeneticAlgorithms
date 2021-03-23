from functools import partial
from collections import namedtuple
from math import inf, log2, ceil
from typing import List, Callable, Tuple
from random import choices, randint, randrange, random, uniform
from copy import deepcopy
from matplotlib import pyplot
from numpy import array, arange
import matplotlib
import numpy

file = open("result.txt", "w")

# ? type aliases

Chromosome = List[int]
Population = List[Chromosome]
FitnessFunction = Callable[[Chromosome], int]
PopulateFunction = Callable[[], Population]
SelectionFunction = Callable[[Population, FitnessFunction], Tuple[Chromosome, Chromosome]]
CrossoverFunc = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]
MutationFunc = Callable[[Chromosome], Chromosome]
ChromosomeFloatFunction = Callable[[Chromosome, int], float]
Parameters = Tuple[int, int, int]
Bounds = Tuple[int, int]


def generate_chromosome(bounds: Bounds, length: int, precision: int) -> Chromosome:
    """[function that generates a float (between bounds) and returns a list of 1 and 0 + its sign]

    Args:
        bounds (Bounds): [upper bound and lower bound]
        length (int): [the length of the chromosome]
        precision (int): [number of decimals]

    Returns:
        Chromosome: [the first el. is +-1 and the others are transposed to binary]
    """
    float_chromosome = round(uniform(bounds[0], bounds[1]), precision)
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
    return list_chromosome

def generate_population(dimension: int, bounds: Bounds, length: int, precision: int) -> Population:
    """[function that generates a list of chromosomes using the previous function]

    Args:
        dimension (int): [number of chromosomes that need to be generated]
        bounds (Bounds): [upper and lower bound]
        length (int): [the length of the chromosome]
        precision (int): [number of decimals]

    Returns:
        Population: [description]
    """
    return [generate_chromosome(bounds, length, precision) for _ in range(dimension)]

def chromosome_to_float(list_chromosome: Chromosome, precision: int) -> float:
    """[the opposite of generate chromosome]

    Args:
        list_chromosome (Chromosome): [the number but it s a list of 0 and 1]
        precision (int): [decimals]

    Returns:
        float: [the same number but it's float]
    """
    result = 0
    copy = deepcopy(list_chromosome)
    sign = copy[0]
    del copy[0]
    ln = len(copy)
    copy.reverse()
    for i in range(ln):
        result += copy[i] * 2 ** i
    result = sign * result / (10 ** precision)
    return round(result, precision)

def apply_function(parameters: Parameters, x: float) -> float:
    """[applies the function a * x ^ 2 + b * x + c]

    Args:
        parameters (Parameters): [a, b, c]
        x (float): [x]

    Returns:
        float: [the result of the function]
    """
    return parameters[0] * (x ** 2) + parameters[1] * x + parameters[2]

def fitness(list_chromosome: Chromosome, parameters: Parameters, bounds: Bounds, precision: int) -> float:
    """[FITNESS FUNCTION]

    Args:
        list_chromosome (Chromosome): [chromosome as a list of 1 and 0]
        parameters (Parameters): [parameters of the quadriatic function]
        bounds (Bounds): [upper and lower bound]
        precision (int): [decimals]

    Returns:
        float: [returns 0 if the number is out of the bounds. Otherwise, the result is the function itself]
    """
    nr = chromosome_to_float(list_chromosome, precision)
    if nr < bounds[0] or nr > bounds[1]:
        return -inf
    return apply_function(parameters, nr)

def selection_pair(population: Population, fitness_function: FitnessFunction) -> Population:
    """[THE SELECTION FUNCTION]

    Args:
        population (Population): [list of chromosomes]
        fitness_function (FitnessFunction): [fitness function]

    Returns:
        Population: [returns a list of 2 chromosomes]
    """
    return choices(
        population = population,
        weights = [fitness_function(chromosome) for chromosome in population],
        k = 2
    )

def single_point_crossover(a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
    """[CROSSOVER FUNCTION]

    Args:
        a (Chromosome): [first chromosome]
        b (Chromosome): [second chromosome]

    Returns:
        Tuple[Chromosome, Chromosome]: [returns the two children]
    """
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
    """[THE MUTATION FUNCTION]

    Args:
        chromosome (Chromosome): [description]
        probability (float): [description]

    Returns:
        Chromosome: [the same chromosome, probably with an element changed]
    """
    index = randrange(len(chromosome))
    chromosome[index] = chromosome[index] if random() > probability else abs(chromosome[index] - 1)
    return chromosome

def calculate_crossover_probability(chromosome: Chromosome, population: Population, parameters: Parameters, bounds: Bounds, precision: int) -> float:
    """[CROSSOVER PROBABILITY FUNCTION]

    Args:
        chromosome (Chromosome): [description]
        population (Population): [description]
        parameters (Parameters): [description]
        bounds (Bounds): [description]
        precision (int): [description]

    Returns:
        float: [description]
    """
    return fitness(chromosome, parameters, bounds, precision) / sum([fitness(item, parameters, bounds, precision) for item in population])

def show_function(x_axis: List[float], y_axis: List[float], bounds: Bounds, parameters: Parameters, precision: int):
    """[the function that shows the best results on the graphic]

    Args:
        x_axis (List[float]): [the float values of the best chromosomes for each generation]
        y_axis (List[float]): [the value of the function for each chromosome]
        bounds (Bounds): [upper and lower bound]
        parameters (Parameters): [parameters of the quadriatic function]
        precision (int): [decimals]
    """
    # actual functiohn
    x = arange(bounds[0], bounds[1], 10 ** (-precision))
    y = [apply_function(parameters, i) for i in x]
    pyplot.plot(x, y)

    # best results
    x1 = array(x_axis)
    y1 = array(y_axis)

    # actual best result
    x2 = array([i for i in x1 if apply_function(parameters, i) == max(y1)])
    y2 = array([apply_function(parameters, i) for i in x1 if apply_function(parameters, i) == max(y1)])
    file.write("The optimal solution is " + str(x2[0]) + "\n\n")

    colors = [y2[0] / i * (10 ** precision) for i in y1]
    pyplot.scatter(x1, y1, c = colors, cmap = 'CMRmap')
    pyplot.scatter(x2, y2, cmap = 'GnBu')
    pyplot.colorbar()
    pyplot.show()

def show_population(population: Population, parameters: Parameters, bounds: Bounds, precision: int):
    for i in population:
        if i[0] == 1:
            res = " " + str(i[0]) + " "
        else:
            res = str(i[0]) + " "
        for j in i[1 : ]:
            res += str(j)
        w = "X = " + res + ",  val = "
        if chromosome_to_float(i, precision) > 0:
            w += " " + str(chromosome_to_float(i, precision)) + ",  fitness: " + str(fitness(i, parameters, bounds, precision)) + "\n"
        else:
            w += str(chromosome_to_float(i, precision)) + ",  fitness: " + str(fitness(i, parameters, bounds, precision)) + "\n"
        file.write(w)

def show_crossover_probability(population: Population, parameters: Parameters, bounds: Bounds, precision: int):
    for i in population:
        file.write("X: " + str(chromosome_to_float(i, precision)) + "  Crossover probability: " + str(calculate_crossover_probability(i, population, parameters, bounds, precision)) + "\n")

def evolution( 
    populate_function: PopulateFunction,
    fitness_function: FitnessFunction,
    show_function,
    show_population,
    show_crossover,
    chromosome_to_float: ChromosomeFloatFunction,
    selection_function: SelectionFunction = selection_pair,
    crossover_function: CrossoverFunc = single_point_crossover,
    mutation_function: MutationFunc = mutation,
    mutation_probability: float = 0,
    generation_limit: int = 100
) -> Population:
    """[the evolution function that goes through all generations]

    Args:
        populate_function (PopulateFunction): [generate population function]
        fitness_function (FitnessFunction): [fitness function]
        show_function (void): [graph function]
        chromosome_to_float (ChromosomeFloatFunction): [transforms the binary list to float]
        selection_function (SelectionFunction, optional): [the selection function]. Defaults to selection_pair.
        crossover_function (CrossoverFunc, optional): [the crossover function]. Defaults to single_point_crossover.
        mutation_function (MutationFunc, optional): [the mutation function]. Defaults to mutation.
        mutation_probability (float, optional): [the mutation probability value]. Defaults to 0.
        generation_limit (int, optional): [the number of generations]. Defaults to 100.

    Returns:
        Population: [the final population]
    """
    population = populate_function()
    file.write("Initial population: \n\n")
    show_population(population)
    file.write("\n\nCP: \n\n")
    show_crossover(population)
    file.write("\n\n")

    x = []
    y = []

    for i in range(generation_limit):
        population = sorted(population, key = lambda chromosome: fitness_function(chromosome), reverse = True)
        next_generation = population[0 : 2] # elitistic selection
        for j in range(int(len(population) / 2) - 1):
            parents = selection_function(population, fitness_function)
            offspring_a, offspring_b = crossover_function(parents[0], parents[1])
            offspring_a, offspring_b = mutation_function(offspring_a, mutation_probability), mutation_function(offspring_b, mutation_probability)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
        x.append(chromosome_to_float(population[0]))
        y.append(fitness_function(population[0]))
        file.write("Best solution: " +  str(chromosome_to_float(population[0])) + '\n')
    show_function(x, y)
    return population


def main():
    dimension : int = 30
    bounds : Bounds = (-1, 2)
    parameters : Parameters = (-1, 1, 2)
    precision : int = 6
    mutation_probability : float = 0.01
    nr_generations : int = 20
    length = 2 + ceil(log2(max(abs(bounds[0] * 10 ** precision), abs(bounds[1] * 10 ** precision))))
    population = generate_population(dimension, bounds, length, precision)
    # print(population)
    # print([fitness(item, parameters, bounds, precision) for item in population])
    # show_function(bounds, parameters, precision)
    final_population = evolution(
        populate_function = partial(generate_population, dimension = dimension, bounds = bounds, length = length, precision = precision), 
        fitness_function = partial(fitness, parameters = parameters, bounds = bounds, precision = precision),
        show_function = partial(show_function, bounds = bounds, parameters = parameters, precision = precision),
        show_population = partial(show_population, parameters = parameters, bounds = bounds, precision = precision),
        show_crossover = partial(show_crossover_probability, parameters = parameters, bounds = bounds, precision = precision),
        chromosome_to_float = partial(chromosome_to_float, precision = precision),
        selection_function = selection_pair,
        crossover_function = single_point_crossover,
        mutation_function = mutation,
        mutation_probability = mutation_probability,
        generation_limit = nr_generations
        )


if __name__ == "__main__":
    main()