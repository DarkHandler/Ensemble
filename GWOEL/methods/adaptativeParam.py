import random
import math

def updatePositionSearchAgents(a, SearchAgents_no, Max_iter, Positions, l, Alpha_pos, Delta_pos, Beta_pos, function_name):
    # Update the Position of search agents including omegas
    for i in range(0, SearchAgents_no):
        wolfSize = len(Positions[i])
        if function_name == "teamSizeModel":
            A = [n for n in Positions[i] if n != 0]
            wolfSize = len(A)
        #print(wolfSize)
        for j in range(0, wolfSize):

            r1 = random.random()  # r1 is a random number in [0,1]
            r2 = random.random()  # r2 is a random number in [0,1]

            A1 = 2 * a * r1 - a
            # Equation (3.3)
            C1 = 2 * r2
            # Equation (3.4)
            
            D_alpha = abs(C1 * Alpha_pos[j] - Positions[i][j])
            # Equation (3.5)-part 1
            X1 = Alpha_pos[j] - A1 * D_alpha
            # Equation (3.6)-part 1

            r1 = random.random()
            r2 = random.random()

            A2 = 2 * a * r1 - a
            # Equation (3.3)
            C2 = 2 * r2
            # Equation (3.4)

            D_beta = abs(C2 * Beta_pos[j] - Positions[i][j])
            # Equation (3.5)-part 2
            X2 = Beta_pos[j] - A2 * D_beta
            # Equation (3.6)-part 2

            r1 = random.random()
            r2 = random.random()

            A3 = 2 * a * r1 - a
            # Equation (3.3)
            C3 = 2 * r2
            # Equation (3.4)

            D_delta = abs(C3 * Delta_pos[j] - Positions[i][j])
            # Equation (3.5)-part 3
            X3 = Delta_pos[j] - A3 * D_delta
            # Equation (3.5)-part 3

            if function_name == "teamSizeModel":
                Positions[i][j] = int(math.fabs( (X1 + X2 + X3) / 3))  # Equation (3.7)
            else:
                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)


def linealUpdateMethod(SearchAgents_no, Max_iter, Positions, l, Alpha_pos, Delta_pos, Beta_pos, function_name):
    a = 2 - l * ((2) / Max_iter)

    # a decreases linearly from 2 to 0
    updatePositionSearchAgents(a, SearchAgents_no, Max_iter, Positions, l, Alpha_pos, Delta_pos, Beta_pos, function_name)


    

def adaptativeControlParameter_a_UpdateMethod(t, T, SearchAgents_no, Positions, Alpha_pos, Delta_pos, Beta_pos, function_name): #se pasa t es la iteracion actual, T iteracion maxima, a_s valor inicial o anterior de a_b
    a_r = 2.0
    a_t = T/2
    a_s = -0.5
    
    r5 = random.random()
    a_0 = 0.5 * r5 * (1 - t/T) * math.sin(T - t)
    a_b = a_s + ( a_r / ( 1 + math.exp(a_t * ( (2 * (t/T)) - 1)) ) ) 
    a_a = a_0 + a_b

    updatePositionSearchAgents(a_a, SearchAgents_no, T, Positions, t, Alpha_pos, Delta_pos, Beta_pos, function_name)

        