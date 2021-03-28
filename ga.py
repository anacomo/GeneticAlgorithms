# Comorasu Ana-Maria
# Tema 2 Algoritmi Genetici
# Grupa 234

from matplotlib import pyplot
from math import inf, log2, ceil
from numpy import arange, array
from decimal import Decimal
from typing import List, Callable, Tuple
from random import choice, choices, uniform, randint, randrange, random

# ? input and output files
input = open("input.txt", "r")
output = open("result.txt", "w")

# todo type aliases
Bounds = Tuple[int, int]
Parameters = Tuple[int, int, int]
Chromosome = List[int]
Population = List[Chromosome]

class Variables:
    def __init__(self, length: int, bounds: Bounds, parameters: Parameters, precision: int) -> None:
        self.length = length
        self.bounds = bounds
        self.parameters = parameters
        self.precision = precision

# * quadriatic function
def function(x: float, vars: Variables) -> float:
    return vars.parameters[0] * x ** 2 + vars.parameters[1] * x + vars.parameters[2]

# * function that generates chromosome
def generate_chromosome(vars: Variables) -> Chromosome:
    float_number = round(uniform(vars.bounds[0], vars.bounds[1]) - vars.bounds[0], vars.precision)
    float_number = int(float_number * 10 ** vars.precision)
    # print(float_number)
    float_number = format(float_number, '0' + str(vars.length) + 'b')
    binary_chromosome = [int(i) for i in float_number]
    # print(float_number)
    return binary_chromosome

# * function that generates population
def generate_population(dimension: int, vars: Variables) -> Population:
    return [generate_chromosome(vars) for _ in range(dimension)]

# * function that returns the float value of the chormosome
def binary_to_float(binary_chromosome: Chromosome, vars: Variables) -> float:
    binary = ''.join([str(item) for item in binary_chromosome])
    float_number = int(binary, 2)
    # print(round(float * 10 ** (- vars.precision), vars.precision))
    return round(float_number * 10 ** (- vars.precision), vars.precision)

# * find the vertex of the function
def find_max(vars: Variables) -> float:
    a, b = vars.parameters[0], vars.parameters[1]
    x_vertex = - b / 2 * a
    if x_vertex > vars.bounds[0] and x_vertex < vars.bounds[1] and a <= 0:
        return x_vertex
    else:
        if function(vars.bounds[0], vars.parameters) > function(vars.bounds[1], vars.parameters):
            return vars.bounds[0]
        else:
            return vars.bounds[1]

# * the FITNESS FUNCTION
def fitness(binary_chromosome: Chromosome, vars: Variables) -> float:
    number = binary_to_float(binary_chromosome, vars)
    if number > vars.bounds[1] - vars.bounds[0]:
        return - 1
    return function(number, vars)

# * the selection probability
def selection_prob(c: Chromosome, population: Population, vars: Variables) -> float:
    return fitness(c, vars) / sum([fitness(item, vars) for item in population])

# * the search function
def search(u: float, intervals: List[float]) -> int:
    i = 0
    step = 1
    length = len(intervals)
    while step < length:
        step *= 2
    while step:
        if i + step < length and intervals[i + step] < u:
            i += step
        step //= 2
    return i

# * the SELECTION FUNCTION
def roulette_selection(population: Population, ret_dimension: int, vars: Variables, show: bool = False) -> Population:
    intervals = [0]
    sum = 0
    for i in range(len(population)):
        sum += selection_prob(population[i], population, vars)
        intervals.append(sum)

    if show:
        output.write("\nSelection intervals: \n" + str(intervals) + "\n\n")

    intermediate_population = []
    for i in range(ret_dimension):
        u = random()
        index = search(u, intervals)

        if show: 
            output.write("u = " + str(u) + "    select " + str(index) + '\n')
        
        intermediate_population.append(population[index])

    if show:
        output.write("\n\nAfter selection: \n\n")
        for item in intermediate_population:
            output.write("X = " + str(item) + "    Value: " + str(binary_to_float(item, vars)) + "    f: " + str(fitness(item, vars)) + '\n')
        output.write("\n\n")
    return intermediate_population

# * random selection from intermediate population
def random_selection(population: Population) -> Population:
    first = choice(population)
    population.remove(first)
    second = choice(population)
    population.remove(second)
    return [first, second]

# * the CROSSOVER FUNCTION
def single_point_crossover(a: Chromosome, b: Chromosome, show: bool = False) -> Tuple[Chromosome, Chromosome]:
    length = len(a)
    if length < 2:
        return a, b
    point = randint(1, length - 1)
    offspring_a, offspring_b = a[0 : point] + b[point : ], b[0 : point] + a[point : ]
    
    if show:
        output.write("Crossover: " + str(a) + "  " + str(b) + "\n")
        output.write("Result:    " + str(offspring_a) + "  " + str(offspring_b) + '\n\n')

    return offspring_a, offspring_b

# * the mutation function
def mutation(a: Chromosome, mutation_prob: float, show: bool = False) -> Chromosome:
    index = randrange(1, len(a))
    if show: 
        output.write("Mutation from " + str(a))
    a[index] = a[index] if random() > mutation_prob else abs(a[index] - 1)
    if show:
        output.write(" to " + str(a) + '\n')
    return a

def show_function(vars: Variables) -> None:
    pyplot.subplot(2, 2, 1)
    x = arange(vars.bounds[0], vars.bounds[1], 10 ** -vars.precision)
    y = [function(i, vars) for i in x]
    pyplot.plot(x, y)

    xmax, ymax = find_max(vars), function(find_max(vars), vars)

    pyplot.scatter(array(xmax), array(ymax), marker = '*')
    pyplot.title("Quadriatic function") 

def show_elite(elite: Population, vars: Variables) -> None:
    pyplot.subplot(2, 2, 2)
    x = arange(vars.bounds[0], vars.bounds[1], 10 ** -vars.precision)
    y = [function(i, vars) for i in x]
    pyplot.plot(x, y)

    x_elite = array(elite)
    y_elite = array([function(item, vars) for item in elite])

    output.write("\n\nMaximum: \n")
    for item in elite:
        d = str(item)
        output.write("Val: " + d + "  fitness: " + str(function(item, vars)) +"\n")
        # print("de aici ", d)

    pyplot.scatter(x_elite, y_elite, color = 'hotpink')
    pyplot.title("Elites of every generation")

def show_initial_population(population: Population, vars: Variables) -> None:
    output.write("Initial Population: \n\n")
    for item in population:
        output.write("X = " + str(item) + "    Value: " + str(binary_to_float(item, vars)) + "    f: " + str(fitness(item, vars)) + '\n')
    output.write("\n\nSelection Probability\n")
    for item in population:
        output.write("Value: " + str(binary_to_float(item, vars)) + "    " + str(selection_prob(item, population, vars)) + "\n")

    pyplot.subplot(2, 2, 3)
    x = arange(vars.bounds[0], vars.bounds[1], 10 ** -vars.precision)
    y = [function(i, vars) for i in x]
    pyplot.plot(x, y)

    x_pop = array([binary_to_float(item, vars) for item in population])
    y_pop = array([function(item, vars) for item in x_pop])

    pyplot.scatter(x_pop, y_pop, color = '#88c999')
    pyplot.title("Initial Generation")

def show_final_population(population: Population, vars: Variables) -> None:
    output.write("\nFinal Population: \n\n")
    for item in population:
        output.write("X = " + str(item) + "    Value: " + str(binary_to_float(item, vars)) + "    f: " + str(fitness(item, vars)) + '\n')

    pyplot.subplot(2, 2, 4)
    x = arange(vars.bounds[0], vars.bounds[1], 10 ** -vars.precision)
    y = [function(i, vars) for i in x]
    pyplot.plot(x, y)

    x_pop = array([binary_to_float(item, vars) for item in population])
    y_pop = array([fitness(item, vars) for item in population])

    pyplot.scatter(x_pop, y_pop, color = 'purple')
    pyplot.title("Final Generation")

# * the evolution function
def evolution(vars: Variables, dimension: int, generations: int, crossover_prob: float, mutation_prob: float) -> Population:
    population = generate_population(dimension, vars)
    selected = ceil(crossover_prob * dimension / 2 * 2) # selected for crossover
    members = dimension - selected - 2                  # memnbers going to next gen
    
    show_function(vars)
    show_initial_population(population, vars)
    elite = [ ]

    for i in range(generations):
        population = sorted(population, key = lambda chromosome: fitness(chromosome, vars), reverse = True)
        # todo 1 elitistic selection
        next_generation = population[0 : 2]
        # print(population[0])
        # print(binary_to_float(population[0], vars))
        elite.append(binary_to_float(population[0], vars))

        # todo 2 copy
        # * select (1 - cp) * dimension
        next_generation += choices(population = population, k = members)

        # todo 3 crossover
        # * select (cp * dimension) members, pair them and produce offspring
        intermediate_population = roulette_selection(population, selected, vars, i == 1)
        for _ in range(selected // 2):
            parents = random_selection(intermediate_population)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1], i == 1)
            next_generation += [offspring_a, offspring_b]

        # todo 3 mutation
        next_generation = [mutation(item, mutation_prob, i == 1) for item in next_generation]
        population = next_generation

    show_elite(elite, vars)
    show_final_population(population, vars)

def main():
    dimension: int = 100
    bounds: Bounds = (0, 10)
    parameters: Parameters = (-1, 10, 10)
    precision: int = 3
    nr_generations: int = 20
    crossover_probability = 0.25
    mutation_probability = 0

    length = ceil(log2((bounds[1] - bounds[0]) * 10 ** precision))
    vars = Variables(length, bounds, parameters, precision)
    
    if function(bounds[0], vars) < 0 or function(bounds[1], vars) < 0:
        raise ValueError("The function should be positive")

    final_population = evolution(vars, dimension, nr_generations, crossover_probability, mutation_probability)

    pyplot.tight_layout()
    pyplot.show()

if __name__ == "__main__":
    main()