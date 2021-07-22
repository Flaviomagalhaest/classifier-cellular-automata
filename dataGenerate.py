import copy
import csv

class DataGenerate:
   predictReal = []
   predictFinded = []      #predict, score
   classifiers = []        #name, score, predict
   
   matrix = []             #iteracao, sample, x, y, matrix,
   enegy = []              #iteracao, sample, x, y, classif energy
   status_deads = []       #iteracao, sample, classify_dead, energy, classify_new, energy
   it =     0
   sample = 0
   x =      0
   y =      0

   report = []

   def __init__(self, predictReal, classif):
      DataGenerate.predictReal = copy.deepcopy(predictReal)
      listName = copy.deepcopy([[c[1]['name'], c[1]['score']] for c in classif.items()])
      listPredict = copy.deepcopy([list(c[1]['predict']) for c in classif.items()])
      for i in range(len(listName)):
         DataGenerate.classifiers.append(listName[i] + listPredict[i])
   
   def saveStatus(matrix, classif, it=-1, sample=-1, x=-1, y=-1):
      DataGenerate.it = it
      DataGenerate.sample = sample
      DataGenerate.x = x
      DataGenerate.y = y
      
      DataGenerate.saveMatrix(matrix, it, sample, x, y)
      DataGenerate.saveEnergy(classif, it, sample, x, y)

   def saveMatrix(matrix, it, sample, x, y):
      lista = []
      lista.append(copy.deepcopy(it))
      lista.append(copy.deepcopy(sample))
      lista.append(copy.deepcopy(x))
      lista.append(copy.deepcopy(y))
      for i in range(len(matrix[0])):
         lista += copy.deepcopy([m['name'] for m in matrix[i]])
      DataGenerate.matrix.append(copy.deepcopy(lista))

   def saveEnergy(classif, it, sample, x, y):
      lista = []
      lista.append(copy.deepcopy(it))
      lista.append(copy.deepcopy(sample))
      lista.append(copy.deepcopy(x))
      lista.append(copy.deepcopy(y))
      lista += copy.deepcopy([c[1]['name'] for c in classif.items()])
      lista += copy.deepcopy([c[1]['energy'] for c in classif.items()])
      DataGenerate.enegy.append(copy.deepcopy(lista))

   def saveDeadCell(indiv, classifier, newEnergy):
      lista = []
      lista.append(copy.deepcopy(DataGenerate.it))
      lista.append(copy.deepcopy(DataGenerate.sample))
      lista.append(copy.deepcopy(DataGenerate.x))
      lista.append(copy.deepcopy(DataGenerate.y))
      lista.append(indiv['name'])
      lista.append(indiv['energy'])
      lista.append(classifier['name'])
      lista.append(newEnergy)
      DataGenerate.status_deads.append(copy.deepcopy(lista))
      a = 'a'

   def file(score, predict):
      with open('file/predict.csv', 'w', newline='') as csvfile:
         spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
         spamwriter.writerow(DataGenerate.predictReal)
         listPredict = copy.deepcopy(predict)
         listPredict.append(score)
         spamwriter.writerow(listPredict)

      with open('file/classifiers.csv', 'w', newline='') as csvfile:
         spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
         spamwriter.writerows(DataGenerate.classifiers)

      with open('file/matrix.csv', 'w', newline='') as csvfile:
         spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
         spamwriter.writerows(DataGenerate.matrix)

      with open('file/energy.csv', 'w', newline='') as csvfile:
         spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
         # spamwriter.writerows(DataGenerate.enegy)
         spamwriter.writerow(DataGenerate.enegy[-1])

      with open('file/iteration_deads.csv', 'w', newline='') as csvfile:
         spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
         spamwriter.writerows(DataGenerate.status_deads)

   def saveResult(scoreLearning, scoreInference, answerlist, matrixSize, matrix, it, distance, dataset):
      def listVoting():
         list = []
         for i in range(len(answerlist)):
            if answerlist[i] != DataGenerate.predictReal[i]:
               matrixOfElementWrong = []
               for x in range(matrixSize):
                  matrixOfY = []
                  for y in range(matrixSize):
                     # c = copy.deepcopy(classif[matrix[x][y]['name']]['predict'][i+len(answerlist)])  #Remember here that predict in full range and not only CA test range
                     c = copy.deepcopy(matrix[x][y]['predict'][i+len(answerlist)])  #Remember here that predict in full range and not only CA test range
                     c = int(c)
                     e = copy.deepcopy(matrix[x][y]['energy'])
                     matrixOfY.append((c,e))
                  matrixOfElementWrong.append(matrixOfY)
               qtdCellsVotingRigth = sum([[l[0] for l in line].count(int(DataGenerate.predictReal[i])) for line in matrixOfElementWrong])
               qtdCellsVotingWrong = sum([[l[0] for l in line].count(int(answerlist[i])) for line in matrixOfElementWrong])
               obj = {}
               obj['qtdCellsVotingRigth'] = qtdCellsVotingRigth
               obj['qtdCellsVotingWrong'] = qtdCellsVotingWrong
               list.append(copy.deepcopy(obj))
         return list
         
      
      
      cMax = max([c[1] for c in DataGenerate.classifiers])
      cMin = min([c[1] for c in DataGenerate.classifiers])
      listQtdDeads = []
      for i in range(DataGenerate.it + 1):
         qtdDeads = len([d for d in DataGenerate.status_deads if int(d[0]) == i])
         listQtdDeads.append(qtdDeads)
      voting = listVoting()
      matrixSize80Pct = int((matrixSize * matrixSize) * 0.8)
      voteMass = len([item['qtdCellsVotingWrong'] for item in voting if int(item['qtdCellsVotingWrong']) >= matrixSize80Pct])
      voteDiv = len([item['qtdCellsVotingWrong'] for item in voting if int(item['qtdCellsVotingWrong']) < matrixSize80Pct])

      list = []
      list.append(scoreLearning)
      list.append(scoreInference)
      list.append(cMax)
      list.append(cMin)
      list.append(cMax - scoreLearning)
      list.append(cMax - scoreInference)
      list.append(sum(listQtdDeads))
      list.append(sum(listQtdDeads) / len(listQtdDeads))
      list.append(max(listQtdDeads))
      list.append(min(listQtdDeads))
      list.append(len(voting))
      list.append(voteMass)
      list.append(voteDiv)
      list.append(matrixSize)
      list.append(it)
      list.append(distance)
      list.append(dataset)

      DataGenerate.report.append(copy.deepcopy(list))

   def report():
      
      a = 'a'
      pass