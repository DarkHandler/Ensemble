
import numpy as np
import random

def mutation(offspring, individualLength, lb, ub):
    """
    The mutation operator of a single individual

    Parameters
    ----------
    offspring : list
        A generated individual after the crossover
    individualLength: int
        The maximum index of the crossover
    lb: list
        lower bound limit list
    ub: list
        Upper bound limit list

    Returns
    -------
    N/A
    """
    mutationIndex = random.randint(0, individualLength - 1)
    mutationValue = random.uniform(lb, ub)#lb[mutationIndex], ub[mutationIndex])
    offspring[mutationIndex] = mutationValue
    return offspring


def elite_selection(population, fitness, elite_size):
    """
    Function to implement elite selection mechanism.
    Parameters:
        population (numpy array): Current population of solutions.
        fitness (numpy array): Fitness values of the solutions in the population.
        elite_size (int): Number of elite solutions to be selected.
    Returns:
        elite_population (numpy array): Elite solutions from the current population.
    """
    elite_indices = np.argsort(fitness)[:elite_size]
    elite_population = population[elite_indices]
    return elite_population


def update_population(population, elite_population):
    """
    Function to update the population with elite solutions.
    Parameters:
        population (numpy array): Current population of solutions.
        elite_population (numpy array): Elite solutions from the previous generation.
    Returns:
        updated_population (numpy array): Population updated with elite solutions.
    """
    #population_size = population.shape[0]
    elite_size = elite_population.shape[0]
    
    #created by me
    lb = 0
    ub = 1
    for elite_index in range(0, len(elite_population)):
        elite_population[elite_index] = mutation(elite_population[elite_index], len(elite_population[elite_index]), lb, ub) 

    updated_population = np.concatenate((elite_population, population[elite_size:]))
    return updated_population


# Example usage
population_size = 100
dim = 2
population = np.random.rand(population_size, dim) # 2-dimensional population
fitness = np.sum(population, axis=1) # fitness function
elite_size = 10
elite_population = elite_selection(population, fitness, elite_size)

updated_population = update_population(population, elite_population)

