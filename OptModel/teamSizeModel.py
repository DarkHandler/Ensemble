import numpy as np

#En este trabajo se detalla el modelo de optimizacion que se utiliza para realizar el computo de:
#Optimizing the Self-Organizing Team Size Using a Genetic Algorithm in Agile Practices


#Este problema tiene una dimension dinamica de cada una de las variables, por lo que debiese modificar el funcionamiento 
# del algoritmo GWO haciendo que genere soluciones no llenas, es decir, con valores 0

#para el entendimiento en el algoritmo, 0 es una variable de una tamanio de un equipo que significara que no debe 
# tomarse encuenta para el tamanio del equipo, es decir, variable T

def teamSizeModel(search_agent): #minimization of ..
    #proceso para eliminar los valores 0 del vector A
    A = [i for i in search_agent if i != 0] #se recorre la solucion para eliminar residuos de con valor 0 creados por el modelo y evaluar el resultado

    T = len(A)
    persons_weight = 1
    team_weight = 1.514
    sum_persons = 0
    for elem in A:
        sum_persons += ( elem * (elem - 1) )
    sum_persons = (sum_persons * persons_weight) / 2

    team_value = (( T * (T - 1) ) * team_weight) / 2
    
    return (sum_persons + team_value) 


# Initialize the population/solutions
def initPopTeamSizeModel(SearchAgents_no, lb, ub, proyectSize, dim):
    Sol = []
    for i in range(0, SearchAgents_no): #por cada solucion
        solDimensionPosition = [] #array de posicion
        teamSize = np.random.randint(lb, ub) #generacion de tamaño de equipo

        if teamSize < proyectSize: #comprobar si el valor generado el menor al tamaño del proyecto para añadir
            solDimensionPosition.append(teamSize) #se agrega el valor al lobo 
            flag = 0
            sumOfAgentDimenPos = sum(solDimensionPosition) #hago la suma de los valores dentro del array del lobo
            while flag != 1: #mientras la suma no llegue al tamaño del proyecto o un poco menor dentro del dominio de valores, seguir agregando estos
                    teamSize = np.random.randint(lb, ub) 
                    summatory = sumOfAgentDimenPos + teamSize
                    if summatory > proyectSize:
                        newValue = proyectSize - sumOfAgentDimenPos
                        if newValue >= lb and newValue < ub:
                            solDimensionPosition.append(newValue)
                        flag = 1
                    elif sumOfAgentDimenPos == proyectSize:
                        flag = 1
                    else:
                        solDimensionPosition.append(teamSize)
                    sumOfAgentDimenPos = sum(solDimensionPosition)
                
            if (len(solDimensionPosition) != dim): #si falta rellenar espacio para alcanzar el tamanio de la dimension, agregar zeros
                rest = dim - len(solDimensionPosition)
                for n in range(0, rest):
                    solDimensionPosition.append(0)
        Sol.append(solDimensionPosition) 
    return Sol
        
    
# Return back the search agents that go beyond the boundaries of the search space and restricction
def checkBoundaries(search_agent, proyectSize, lb, ub):
    # Si los agentes de la poblacion poseen algun valor que no debiera fuera del rango, dejarlo en 0
    A = []
    for index in range(0, len(search_agent)):
        if search_agent[index] != 0:
            A.append(search_agent[index])
            search_agent[index] = 0
           
    for j in range(0, len(A)):
        newValue = int(np.clip(A[j], lb, ub))
        search_agent[j] = newValue
        A[j] = newValue

    #COMPRUEBO RESTRICCION - aumentando el tamanio de este cuando hace falta
    sumOfA = sum(A) # suma de los valores dentro del array
    if sumOfA < proyectSize: #añadir valores que hagan falta para que la restriccion se cumpla, es decir la suma de los valores del vector sea igual muy poco menor al tamaño del proyecto
        flag = 0
        while flag != 1: #mientras haga falta añadir valores aleatorios para agregar al vector solucion, seguir añadiendo
            teamSize = np.random.randint(lb, ub) #se genera un valor para agregar
            summatory = sumOfA + teamSize 
            if summatory > proyectSize: #se comprueba si la suma del valor al actual es mayor al tamanio del proyecto 
                newValue = proyectSize - sumOfA 
                if newValue >= lb and newValue < ub:
                    A.append(newValue)
                    indexOfNewItem = len(A)-1
                    search_agent[indexOfNewItem] = newValue
                flag = 1
            elif sumOfA == proyectSize:
                flag = 1
            else: #si aun queda espacio agrego sin parar un nuevo elemento
                A.append(teamSize)
                indexOfNewItem = len(A)-1
                search_agent[indexOfNewItem] = teamSize
            sumOfA = sum(A)

    while sumOfA > proyectSize: #COMPRUEBO RESTRICCION - disminuyendo el tamanio de este cuando hace falta
        #aqui si la restriccion no se cumple, se elimina un elemento y queda en 0
        indexOfItemDeleted = len(A)-1
        del A[-1]
        sumOfA = sum(A)
        if sumOfA < proyectSize: 
            fillValue = proyectSize - sumOfA
            if fillValue >= lb and fillValue < ub: #si este valor esta dentro de los limites, se agrega, si no se deja normal
                A.append(fillValue)
                search_agent[indexOfItemDeleted] = fillValue
            else:
                search_agent[indexOfItemDeleted] = 0
        else:
            search_agent[indexOfItemDeleted] = 0 #elemento eliminado
