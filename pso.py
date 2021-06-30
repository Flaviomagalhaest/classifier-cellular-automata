import random, copy
import numpy as np
import cca

####### PARAMS ############
rangeTRA = [0.1, 20]
rangeTRB = [0.1, 20]
rangeTRC = [0.001, 0.1]
rangeTRD = [0.001, 0.1]
###########################

def initPopulation(qtdPop, matrix):
   population = []
   for i in range(qtdPop):
      params = {}
      params['TRA']  = round(random.uniform(rangeTRA[0], rangeTRA[1]), 2)
      params['TRB']  = round(random.uniform(rangeTRB[0], rangeTRB[1]), 2)
      params['TRC']  = round(random.uniform(rangeTRC[0], rangeTRC[1]), 2)
      params['TRD']  = round(random.uniform(rangeTRD[0], rangeTRD[1]), 2)
      p = {}
      p['matrix']    = copy.deepcopy(matrix)
      p['params']    = params
      population.append(p)
   return population

def attPbest(population, rangeSampleCA, Y_test_ca):
   for i in range(len(population)):
      answersList = cca.weightedVote(population[i]['matrix'], rangeSampleCA)
      population[i]['score'] = cca.returnScore(Y_test_ca, answersList)

      if 'pbest' in population[i]:
         if population[i]['score'] > population[i]['pbest']['score']:
            print("individuo "+str(i)+" melhorou o pbest. score: "+str(population[i]['pbest']['score']))
            population[i]['pbest'] = copy.deepcopy(population[i])
      else:
         population[i]['pbest'] = copy.deepcopy(population[i])

def attGbest(population):
   gbest = {}
   gbest['score'] = 0
   for i in range(len(population)):
      if population[i]['score'] > gbest['score']:
         gbest = copy.deepcopy(population[i])
         print("individuo "+str(i)+" é o novo gbest. score: "+str(population[i]['score']))
   return gbest

def velocityEquation(indiv, coefAcceleration, gbest):
   if 'velocity' not in indiv:
      velocity = {}
      velocity['TRA'] = 0
      velocity['TRB'] = 0
      velocity['TRC'] = 0
      velocity['TRD'] = 0
      indiv['velocity'] = velocity
   
   #Transaction RuleS
   for p in ['TRA','TRB','TRC','TRD']:
      diffPBest = indiv['pbest']['params'][p] - indiv['params'][p]
      diffGBest = gbest['params'][p] - indiv['params'][p]
      coefPBest = random.uniform(0, coefAcceleration) * diffPBest
      coefGBest = random.uniform(0, coefAcceleration) * diffGBest
      indiv['velocity'][p] = indiv['velocity'][p] + coefPBest + coefGBest

def positionEquation(indiv):
   #Transaction RuleS
   for p in ['TRA','TRB','TRC','TRD']:
      indiv['params'][p] = indiv['params'][p] + indiv['velocity'][p]

def attPosition(population, coefAcceleration, gbest):
   for i in range(len(population)):
      velocityEquation(population[i], coefAcceleration, gbest)
      positionEquation(population[i])

def attBestResult(gbest, bestResult):
   if 'score' not in bestResult:
      bestResult = copy.deepcopy(gbest)
   if gbest['score'] > bestResult['score']:
      bestResult = copy.deepcopy(gbest)
      print('Melhor resultado encontrado até agora: '+str(bestResult['score']))
   return bestResult