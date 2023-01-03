from sklearn.metrics import jaccard_score
from scipy.spatial import distance
import numpy as np
import math
import os

class Metrics:
  def __init__(self):
    self.div_max = -1 #valor máximo de diversidad encontrado en todo el proceso de optimización para dimension-wise.

    self.percent_exploration = -1
    self.percent_explotation = -1
    self.diversidadHamming = -1
    self.diversidad_Dice = -1
    self.diversidad_Jaccard = -1
    self.diversidad_Kulsinski = -1
    self.diversidad_Rogerstanimoto = -1
    self.diversidad_Russellrao = -1
    self.diversidad_Sokalmichener = -1
    self.diversidad_Yule = -1
    self.diversidad_Sokalsneath = -1
    self.diversidadDimensionWise = -1
    self.method = "" #"Adaptative Parameter"


  def obtenerDiversidadHammingPorPareja(self, Poblacion): #cuando el valor de esta metrica se acerca a 0 la poblacion es menos diversa
    Diversidad = 0
    Distancias = []
      
    for i in range(len(Poblacion) - 1):
      for k in range(len(Poblacion)):
        Distancia = 0

        if i < k:        
          Distancia = distance.hamming(Poblacion[i],Poblacion[k])
          Distancias.append(Distancia)
    for i in range(len(Distancias)):
      Diversidad = Diversidad + Distancias[i] 
    
    self.diversidadHamming = Diversidad
    return Diversidad

  def obtenerDiversidadDice(self, Poblacion):
    Distancias = []
    Diversidad = 0

    for j in range(len(Poblacion) -1):
      for k in range(len(Poblacion)):
        if k <= j:
          continue
        else:
          X = Poblacion[j]
          Y = Poblacion[k]

        Distancias.append(distance.dice(X,Y))

    for i in range(len(Distancias)):
      Diversidad = Diversidad + Distancias[i]

      Diversidad = Diversidad
    
    self.diversidad_Dice = Diversidad
    return Diversidad


  def obtenerDiversidadJaccard(self, Poblacion):
    Distancias = []
    Diversidad = 0

    for j in range(len(Poblacion) -1):
      for k in range(len(Poblacion)):
        if k <= j:
          continue
        else:
          X = Poblacion[j]
          Y = Poblacion[k]

        Distancias.append(distance.jaccard(X,Y))
        #Revisar diferencias
        #Distancias.append(jaccard_score(X,Y,average='binary'))
        #Distancias.append(jaccard_score(X,Y,average='micro'))
        #Distancias.append(jaccard_score(X,Y,average='macro'))
        #Distancias.append(jaccard_score(X,Y,average='weighted'))

    for i in range(len(Distancias)):
      Diversidad = Diversidad + Distancias[i]

      Diversidad = Diversidad
      
    self.diversidad_Jaccard = Diversidad
    return Diversidad


  def obtenerDiversidadKulsinski(self, Poblacion):
    Distancias = []
    Diversidad = 0

    for j in range(len(Poblacion) -1):
      for k in range(len(Poblacion)):
        if k <= j:
          continue
        else:
          X = Poblacion[j]
          Y = Poblacion[k]

        Distancias.append(distance.kulsinski(X,Y))

    for i in range(len(Distancias)):
      Diversidad = Diversidad + Distancias[i]

      Diversidad = Diversidad

    self.diversidad_Kulsinski = Diversidad      
    return Diversidad
    


  def obtenerDiversidadRogerstanimoto(self, Poblacion):
    Distancias = []
    Diversidad = 0

    for j in range(len(Poblacion) -1):
      for k in range(len(Poblacion)):
        if k <= j:
          continue
        else:
          X = Poblacion[j]
          Y = Poblacion[k]

        Distancias.append(distance.rogerstanimoto(X,Y))
    for i in range(len(Distancias)):
      Diversidad = Diversidad + Distancias[i]

      Diversidad = Diversidad
      
    self.diversidad_Rogerstanimoto = Diversidad
    return Diversidad


  def obtenerDiversidadRussellrao(self, Poblacion):
    Distancias = []
    Diversidad = 0

    for j in range(len(Poblacion) -1):
      for k in range(len(Poblacion)):
        if k <= j:
          continue
        else:
          X = Poblacion[j]
          Y = Poblacion[k]

        Distancias.append(distance.russellrao(X,Y))

    for i in range(len(Distancias)):
      Diversidad = Diversidad + Distancias[i]

      Diversidad = Diversidad
      
    self.diversidad_Russellrao = Diversidad
    return Diversidad

  def obtenerDiversidadSokalmichener(self, Poblacion):
    Distancias = []
    Diversidad = 0

    for j in range(len(Poblacion) -1):
      for k in range(len(Poblacion)):
        if k <= j:
          continue
        else:
          X = Poblacion[j]
          Y = Poblacion[k]

        Distancias.append(distance.sokalmichener(X,Y))


    for i in range(len(Distancias)):
      Diversidad = Diversidad + Distancias[i]

      Diversidad = Diversidad
      
    self.diversidad_Sokalmichener = Diversidad
    return Diversidad


  def obtenerDiversidadYule(self, Poblacion):
    Distancias = []
    Diversidad = 0

    for j in range(len(Poblacion) -1):
      for k in range(len(Poblacion)):
        if k <= j:
          continue
        else:
          X = Poblacion[j]
          Y = Poblacion[k]

        Distancias.append(distance.yule(X,Y))
        
    MatrizDeNan = np.isnan(Distancias)
    #print(f'MatrizDeNan: {MatrizDeNan}')
    for i in range(len(Distancias)):
      if MatrizDeNan[i] == False:
        Diversidad = Diversidad + Distancias[i]

      Diversidad = Diversidad
      
    self.diversidad_Yule = Diversidad
    return Diversidad


  def obtenerDiversidadSokalsneath(self, Poblacion):
    Distancias = []
    Diversidad = 0

    for j in range(len(Poblacion) -1):
      for k in range(len(Poblacion)):
        if k <= j:
          continue
        else:
          X = Poblacion[j]
          Y = Poblacion[k]

        Distancias.append(distance.sokalsneath(X,Y))

    for i in range(len(Distancias)):
      Diversidad = Diversidad + Distancias[i]

      Diversidad = Diversidad
      
    self.diversidad_Sokalsneath = Diversidad
    return Diversidad


  # METRICAS Dimension-wise 

  def obtenerDivDimensionWise(self, Pobla, n_num_agents, m_num_var): #este trabaja con el valor anterior a modo de comparacion

    Array_each_x_sub_j = np.mean(Pobla, axis=0)
    div = 0
    
    sumatoria_div = 0
    for x_sub_j in Array_each_x_sub_j:
      div_j = 0
      sumatoria_div_j = 0
      for i in range(0, n_num_agents):
        sumatoria_div_j += abs(x_sub_j - i)

      div_j = sumatoria_div_j / n_num_agents
      sumatoria_div += sumatoria_div + div_j

    div = sumatoria_div / m_num_var

    self.diversidadDimensionWise = div
    return div


  # -------- CALCULO DE TODAS LAS METRICAS ----------------

  def calculateDiversity(self, Positions, SearchAgents_no, dim, objf):
    self.diversidadHamming = self.obtenerDiversidadHammingPorPareja(Positions)

    if (objf.__name__ == "teamSizeModel"):
      self.diversidad_Dice = self.obtenerDiversidadDice(Positions) #la division por 0 mata el resultado

      self.diversidad_Rogerstanimoto = self.obtenerDiversidadRogerstanimoto(Positions) #la division por 0 mata el resultado

      self.diversidad_Sokalmichener = self.obtenerDiversidadSokalmichener(Positions) #la division por 0 mata el resultado

      self.diversidad_Sokalsneath = self.obtenerDiversidadSokalsneath(Positions) #la division por 0 mata el resultado

    self.diversidad_Jaccard = self.obtenerDiversidadJaccard(Positions)
    self.diversidad_Kulsinski = self.obtenerDiversidadKulsinski(Positions)
    self.diversidad_Russellrao = self.obtenerDiversidadRussellrao(Positions)
    self.diversidad_Yule = self.obtenerDiversidadYule(Positions)
    self.diversidadDimensionWise = self.obtenerDivDimensionWise(Positions, SearchAgents_no, dim)

    if self.div_max < self.diversidadDimensionWise or self.div_max == -1:
        self.div_max = self.diversidadDimensionWise #se asigna por mayor diversidad

    #RECORDAR QUE YA SABEMOS QUE PARAMETROS DE METRICA DE DIMENSION WISE DEBE ESTAR PARA QUE ESTE EN PORCENTAJE DE FASE DE EXPLOTACION O EXPLORACION
    self.percent_exploration = (self.diversidadDimensionWise / self.div_max) * 100
    self.percent_explotation = (math.fabs(self.diversidadDimensionWise - self.div_max) / self.div_max) * 100

    #AQUI DEPENDIENDO DEL PORCENTAJE PODRIA DECIR SI HACER BALANCE HACIA EXPLORACION O EXPLOTACION
        


  #----------- SHOW METRICS ----------------
  def showMetrics(self):
    print(f'Diversidad de Hamming pareja: {self.diversidadHamming}')
    print(f'Diversidad de Dice: {self.diversidad_Dice}')
    print(f'Diversidad de Rogerstanimoto: {self.diversidad_Rogerstanimoto}')
    print(f'Diversidad de Sokalmichener: {self.diversidad_Sokalmichener}')
    print(f'Diversidad de Sokalsneath: {self.diversidad_Sokalsneath}')
    print(f'Diversidad de Jaccard: {self.diversidad_Jaccard}')
    print(f'Diversidad de Kulsinski: {self.diversidad_Kulsinski}')
    print(f'Diversidad de Russellrao: {self.diversidad_Russellrao}')
    print(f'Diversidad de Yule: {self.diversidad_Yule}')
    print(f'Diversidad de Dim-wise: {self.diversidadDimensionWise}')

    print("porcentaje exploracion:", self.percent_exploration)
    print("porcentaje explotacion:", self.percent_explotation)



  # -------------------- STORE METRICS ---------------

  def storeMetricsIn(self, filename, l, fitness, searchAgents_no, proyectSize):
    if not os.path.exists(filename):
      header = "iteration,fitness,searchAgents_no,proyectSize,percent_exploration,percent_explotation,diversidadHamming,diversidad_Dice,diversidad_Jaccard,diversidad_Kulsinski,diversidad_Rogerstanimoto,diversidad_Russellrao,diversidad_Sokalmichener,diversidad_Yule,diversidad_Sokalsneath,diversidadDimensionWise,method\n"
      f = open(filename, "a")
      f.write(header)
    else:
      f = open(filename, "a")

    if self.percent_explotation < 60: #si es menor se debe realizar explotacion --- #para nuestro caso deberemos utilizar 60% debido a que el problema de las métricas no llegan a los 90
      method = "explotation" #explotation para que suba su porcentaje
    else:
      method = "exploration" #exploration para que baje el porcentaje de explotation acercandose a 90

    
    #format --> in one row --> iteration, fitness, searchAgents_no, proyectSize, percent_exploration, percent_explotation, diversidadHamming, diversidad_Dice, diversidad_Jaccard, 
                              #diversidad_Kulsinski, diversidad_Rogerstanimoto, diversidad_Russellrao, diversidad_Sokalmichener, diversidad_Yule, 
                              #diversidad_Sokalsneath, diversidadDimensionWise, method (label)
    f.write(str(l)+","+str(fitness)+","+str(searchAgents_no)+","+str(proyectSize)+","+str(self.percent_exploration)+","+str(self.percent_explotation)+","+str(self.diversidadHamming)+","+str(self.diversidad_Dice)+","+str(self.diversidad_Jaccard)+","+str(self.diversidad_Kulsinski)+","+str(self.diversidad_Rogerstanimoto)+","+str(self.diversidad_Russellrao)+","+str(self.diversidad_Sokalmichener)+","+str(self.diversidad_Yule)+","+str(self.diversidad_Sokalsneath)+","+str(self.diversidadDimensionWise)+","+method+"\n")
      
    f.close()



