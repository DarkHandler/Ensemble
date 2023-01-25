
#INTERNAL LIBRARIES
#from runnerSaver import run #RUNNER
import runnerSaver as rs

# Parametros de los modelos o funciones de optimizacion son:
    #limite inferior, limite superior, dimension, tamano poblacion, numero de iteraciones

# Optimizers
optimizer = ["GWO","FA","GWOEL","PSO","BA","WOA","CSA","DE"] # "PSO","BA","FA","GWO","GWOEL","WOA","CSA","DE"

#optimizer = ["GWOEL"] #experimentacion ACTUALL

# Benchmark function"
# "F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19"
# "teamSizeModel"
objectivefunc = ["teamSizeModel"]

# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns = 30

# General parameters for all optimizers
params = {"PopulationSize": 6, "Iterations": 1500}


#Experimentacion
runner = rs.RunnerSingleton()

# --- GENERATE DATASET WITH METRICS OF GWO ---
#runner.createDatasetMetricsGWO()

# ------- EXECUTE OPTIMIZATION --------
runner.run(optimizer, objectivefunc, NumOfRuns, params)

