import csv, copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
from graph import BarChartRace
from collections import Counter
from datetime import datetime

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
      print('Predict CSV loaded.')

def loadClassifiersCSV():
   with open('file/classifiers.csv', newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      for row in spamreader:
         c = {}
         c['name'] = row[0]
         c['score'] = row[1]
         c['predict'] = row[2:len(row)]
         classif[c['name']] = c
      print('Classifier CSV loaded.')

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
      print('Matrix CSV loaded.')

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
      print('Energy CSV loaded.')

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
      print('Iterations Dead CSV loaded.')

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
   #Last matrix generate
   matrixGenerate = matrixList[-1]['matrix']

   #Predict of test range
   # predictRealTest = predictRealOfProblem[len(predictAnswerOfCA):]
   countErrors = 0
   listErrors = []
   dfList = []
   #loop to get every wrong of list predict generated with right answer
   for i in range(len(predictAnswerOfCA)):
      if predictAnswerOfCA[i] != predictRealOfProblem[i]:
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
         qtdCellsVotingRigth = sum([[l[0] for l in line].count(int(predictRealOfProblem[i])) for line in matrixOfElementWrong])
         qtdCellsVotingWrong = sum([[l[0] for l in line].count(int(predictAnswerOfCA[i])) for line in matrixOfElementWrong])
         dfList.append({ 'answer': predictRealOfProblem[i],
                         'predictCA': predictAnswerOfCA[i],
                         'matrix': dataFrame.to_string(header=False, index=False),
                         'qtdVotingRigth': str(qtdCellsVotingRigth),
                         'qtdVotingWrong': str(qtdCellsVotingWrong)})
   print('Errors file created.')
   

   with open('file/report/report-'+datetime.today().strftime('%Y%m%d-%H%M')+'.txt', 'a', newline='') as txtfile:
      txtfile.write('ERROS NA VOTACAO DA MATRIZ'+'\n')
      for item in dfList:         
         txtfile.write('Answer correct: '+item['answer']+' (qtd cells vote: ' + item['qtdVotingRigth'] + ')\n')
         txtfile.write('Answer found: '+item['predictCA']+' (qtd cells vote: ' + item['qtdVotingWrong'] + ')\n')
         txtfile.write(item['matrix'])
         txtfile.write('\n'+'\n')

def writeReportFile():
   countClassifierDeads = Counter([d['classifier_dead'] for d in deadsList])
   
   for item in energyList[len(energyList)-1]:
      if item != 'it' and item != 'sample' and item != 'x' and item != 'y':
         classif[item]['energy'] = energyList[len(energyList)-1][item]
         classif[item]['deads'] = countClassifierDeads[item]


   index = list(classif.keys())
   columns = ['score', 'deads', 'energy']
   dfClassifiers = pd.DataFrame(classif.values(), columns=columns, index=index)
   
   dfSorted = dfClassifiers.sort_values(by=['energy'], ascending=False)
   dfMatrix = pd.DataFrame(matrixList[-1]['matrix'])
   dfEnergy = pd.DataFrame([[classif[l]['energy'] for l in line] for line in matrixList[-1]['matrix']])
   dfScore = pd.DataFrame([[classif[l]['score'] for l in line] for line in matrixList[-1]['matrix']])
   dfDeads = pd.DataFrame([[classif[l]['deads'] for l in line] for line in matrixList[-1]['matrix']])
   with open('file/report/report-'+datetime.today().strftime('%Y%m%d-%H%M')+'.txt', 'w', newline='') as txtfile:
      txtfile.write('MATRIX DE CLASSIFICADORES'+'\n')
      txtfile.write(dfMatrix.to_string(header=False, index=False))
      txtfile.write('\n'+'\n')
      txtfile.write('ENERGIA'+'\n')
      txtfile.write(dfEnergy.to_string(header=False, index=False))
      txtfile.write('\n'+'\n')
      txtfile.write('SCORE'+'\n')
      txtfile.write(dfScore.to_string(header=False, index=False))
      txtfile.write('\n'+'\n')
      txtfile.write('DEADS'+'\n')
      txtfile.write(dfDeads.to_string(header=False, index=False))
      txtfile.write('\n'+'\n')
      txtfile.write(dfSorted.to_string())
      txtfile.write('\n'+'\n')
   print('Report file created.')

def writeDeadsFile():
   qtdIteration = int(deadsList[-1]['it']) + 1
   listQtdDeads = []
   for i in range(qtdIteration):
      qtdDeads = len([d for d in deadsList if int(d['it']) == i])
      listQtdDeads.append(qtdDeads)
   fig, ax = plt.subplots()
   ax.plot(list(range(0,qtdIteration)), listQtdDeads)
   ax.set(xlabel='Iterations', ylabel='Deads')
   ax.grid()
   plt.savefig('file/report/report-'+datetime.today().strftime('%Y%m%d-%H%M')+'.jpg')
   print('Dead File created.')

loadPredictCSV()
loadClassifiersCSV()
loadMatrixCSV()
loadEnergyCSV()
loadIterationDeadsCSV()

writeReportFile()
writeErrorsFile()
writeDeadsFile()
a = 'a'