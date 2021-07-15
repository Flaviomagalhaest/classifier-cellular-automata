import csv, copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from graph import BarChartRace
from collections import Counter

###### VARIABLES #######
predictRealOfProblem = {}
predictAnswerOfCA = {}
score = {}

matrixList = []
energyList = []
deadsList = []
classif = {}


###### DataFrames #######
dfClassifiers = {}
dfEnergy = {}

###### PARAMS #######
matrixSize = 5

def loadPredictCSV():
   with open('file/predict.csv', newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      rowList = []
      for row in spamreader:
         rowList.append(row)
      global predictRealOfProblem 
      global predictAnswerOfCA
      global score
      predictRealOfProblem = copy.deepcopy(rowList[0])
      predictAnswerOfCA = rowList[1]
      score = predictAnswerOfCA.pop(len(predictAnswerOfCA)-1)

def loadClassifiersCSV():
   with open('file/classifiers.csv', newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      for row in spamreader:
         c = {}
         c['name'] = row[0]
         c['score'] = row[1]
         c['predict'] = row[2:len(row)]
         classif[c['name']] = c

def loadMatrixCSV():
   with open('file/matrix.csv', newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      for row in spamreader:
         matrix = {}
         matrix['it'] = row[0]
         matrix['sample'] = row[1]
         matrix['x'] = row[2]
         matrix['y'] = row[3]

         cl = row[4:len(row)]
         m = []
         for i in range(matrixSize):
            line = []
            for j in range(matrixSize):
               line.append(cl.pop(0))
            m.append(line)
         matrix['matrix'] = m
         matrixList.append(matrix)

def loadEnergyCSV():
   with open('file/energy.csv', newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      for row in spamreader:
         iteracao = {}
         iteracao['it'] = row[0]
         iteracao['sample'] = row[1]
         iteracao['x'] = row[2]
         iteracao['y'] = row[3]

         name = [i for i in row[4:4+len(classif)]]
         value = [float(i) for i in row[4+len(classif):]]
         for n, v in zip(name, value):
            iteracao[n] = v
         energyList.append(iteracao)

def loadIterationDeadsCSV():
   with open('file/iteration_deads.csv', newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      for row in spamreader:
         iteracao = {}
         iteracao['it'] = row[0]
         iteracao['sample'] = row[1]
         iteracao['x'] = row[2]
         iteracao['y'] = row[3]
         iteracao['classifier_dead'] = row[4]
         iteracao['energy_dead'] = row[5]
         iteracao['classifier_realoc'] = row[6]
         iteracao['energy_realoc'] = row[6]
         deadsList.append(iteracao)

def drawBarChartRace():
   for item in energyList[len(energyList)-1]:
      if item != 'it' and item != 'sample' and item != 'x' and item != 'y':
         classif[item]['energy'] = energyList[len(energyList)-1][item]

   index = list(classif.keys())
   columns = ['name', 'score', 'predict', 'energy']
   dfClassifiers = pd.DataFrame(classif.values(), columns=columns, index=index)
   dfClassifiers.sort_values(by=['energy'], ascending=False)

   barChartRace = BarChartRace(matrixSize, len(predictAnswerOfCA), 10, energyList)
   barChartRace.draw()

def writeErrorsFile():
   matrixGenerate = matrixList[-1]['matrix']

   predictRealTest = predictRealOfProblem[len(predictAnswerOfCA):]
   countErrors = 0
   listErrors = []
   dfList = []
   for i in range(len(predictAnswerOfCA)):
      if predictAnswerOfCA[i] != predictRealTest[i]:
         countErrors += 1
         listErrors.append(i)
         matrixOfElementWrong = []
         for x in range(matrixSize):
            matrixOfY = []
            for y in range(matrixSize):
               c = copy.deepcopy(classif[matrixGenerate[x][y]]['predict'][i])
               c = int(c)
               e = copy.deepcopy(energyList[-1][matrixGenerate[x][y]])
               matrixOfY.append((c,e))
            matrixOfElementWrong.append(matrixOfY)
         dataFrame = pd.DataFrame(matrixOfElementWrong)
         dfList.append({'answer': predictRealTest[i], 'predictCA': predictAnswerOfCA[i], 'matrix': dataFrame.to_string(header=False, index=False)})
         a = 'a'

   with open('file/report/errors.txt', 'w', newline='') as txtfile:
      for item in dfList:
         txtfile.write('Answer correct: '+item['answer']+'\n')
         txtfile.write('Answer found: '+item['predictCA']+'\n')
         txtfile.write(item['matrix']+'\n')

def writeReportFile():
   countClassifierDeads = Counter([d['classifier_dead'] for d in deadsList])
   
   for item in energyList[len(energyList)-1]:
      if item != 'it' and item != 'sample' and item != 'x' and item != 'y':
         classif[item]['energy'] = energyList[len(energyList)-1][item]
         classif[item]['deads'] = countClassifierDeads[item]


   index = list(classif.keys())
   columns = ['name', 'score', 'deads', 'energy']
   dfClassifiers = pd.DataFrame(classif.values(), columns=columns, index=index)
   
   dfSorted = dfClassifiers.sort_values(by=['energy'], ascending=False)
   dfMatrix = pd.DataFrame(matrixList[-1]['matrix'])
   with open('file/report/report.txt', 'w', newline='') as txtfile:
      txtfile.write(dfMatrix.to_string(header=False, index=False)+'\n')
      txtfile.write(dfSorted.to_string()+'\n')
   a = 'a'

loadPredictCSV()
loadClassifiersCSV()
loadMatrixCSV()
loadEnergyCSV()
loadIterationDeadsCSV()

# writeErrorsFile()
writeReportFile()
a = 'a'