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

   def __init__(self, predictReal, classif):
      DataGenerate.predictReal = copy.deepcopy(predictReal)
      listName = copy.deepcopy([[c[1]['name'], c[1]['score']] for c in classif.items()])
      listPredict = copy.deepcopy([list(c[1]['predict']) for c in classif.items()])
      for i in range(len(listName)):
         DataGenerate.classifiers.append(listName[i] + listPredict[i])
      a = 'a'
   
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
         predict.append(score)
         spamwriter.writerow(predict)

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
      a = 'a'
