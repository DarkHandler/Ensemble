
import random
import numpy as np



def pairSelection(population, scores, popSize):
    """
    This is used to select one pair of parents using roulette Wheel Selection mechanism

    Parameters
    ----------
    population : list
        The list of individuals
    scores : list
        The list of fitness values for each individual
    popSize: int
        Number of wolfs in a population

    Returns
    -------
    list
        parent1: The first parent individual of the pair
    list
        parent2: The second parent individual of the pair
    """

    parent1Id = rouletteWheelSelectionId(scores, popSize)
    parent1 = population[parent1Id].copy()

    parent2Id = rouletteWheelSelectionId(scores, popSize)
    parent2 = population[parent2Id].copy()

#    print(parent1, parent2)

    return parent1, parent2
    


def rouletteWheelSelectionId(scores, popSize):
    """
    A roulette Wheel Selection mechanism for selecting an individual

    Parameters
    ----------
    scores : list
        The list of fitness values for each individual
    popSize: int
        Number of wolfs in a population

    Returns
    -------
    id
        individualId: The id of the individual selected
    """

    ##reverse score because minimum value should have more chance of selection
    reverse = max(scores) + min(scores)
    reverseScores = reverse - scores.copy()
    sumScores = sum(reverseScores)
    pick = random.uniform(0, sumScores)
    current = 0
    for individualId in range(popSize):
        current += reverseScores[individualId]
        if current > pick:
            return individualId




def crossover(individualLength, parent1, parent2):
    """
    The crossover operator of a two individuals

    Parameters
    ----------
    individualLength: int
        The maximum index of the crossover
    parent1 : list
        The first parent individual of the pair
    parent2 : list
        The second parent individual of the pair

    Returns
    -------
    list
        offspring1: The first updated parent individual of the pair
    list
        offspring2: The second updated parent individual of the pair
    """

    # The point at which crossover takes place between two parents.
    crossover_point = random.randint(0, individualLength - 1)

    # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
    offspring1 = np.concatenate(
        [parent1[0:crossover_point], parent2[crossover_point:]]
    )
    # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
    offspring2 = np.concatenate(
        [parent2[0:crossover_point], parent1[crossover_point:]]
    )

    return offspring1, offspring2

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
    mutationValue = random.uniform(lb[mutationIndex], ub[mutationIndex])
    offspring[mutationIndex] = mutationValue


def methodOfGAOperators(lb, ub, population, arrayPopFitness, SearchAgents_no): #este metodo hara salto de dos en dos en las iteraciones al ser utilizado en el algoritmo GWO #scores sera el donde se guarden los fitness
    #selection, crossover, mutation
    wolf_parent1, wolf_parent2 = pairSelection(population, arrayPopFitness, SearchAgents_no) #roulette wheel

    #crossover
    crossoverLength = min(len(wolf_parent1), len(wolf_parent2)) #evaluar cuando se este resolviendo el otro problema de optimización cuando en el arreglo hay un valor 0
    wolf_parent1, wolf_parent2 = crossover(crossoverLength, wolf_parent1, wolf_parent2)
    
    #mutation probability
    mutation_prob = 0.5
    offspring1MutationProbability = random.uniform(0.0, 1.0)
    offspring2MutationProbability = random.uniform(0.0, 1.0)

    if offspring1MutationProbability < mutation_prob:
        mutation(wolf_parent1, len(wolf_parent1), lb, ub) 
    if offspring2MutationProbability < mutation_prob:
        mutation(wolf_parent2, len(wolf_parent2), lb, ub)
    
    return wolf_parent1, wolf_parent2 #se aplicaran estos cambio al lobo de la iteracion actual i y al siguiente i+1
    
def updatePopWithGAMethod(lb, ub, population, arrayPopFitness, SearchAgents_no):
    for i in range(0,SearchAgents_no, 2):
        population[i], population[(i+1)] = methodOfGAOperators(lb, ub, population, arrayPopFitness, SearchAgents_no)

