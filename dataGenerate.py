import copy
import csv
import pandas as pd
import sys
from datetime import datetime

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

   reportList_config1 = []
   reportList_config2 = []
   reportList_config3 = []
   reportList_config4 = []
   reportList_config5 = []
   reportList_config6 = []
   reportList_config7 = []
   reportList_config8 = []
   reportList_config9 = []
   reportList_config10 = []
   reportList_config11 = []
   reportList_config12 = []
   reportList_config13 = []
   reportList_config14 = []

   def __init__(self, predictReal, classif):
      DataGenerate.predictReal = copy.deepcopy(predictReal)
      DataGenerate.classifiers = []
      DataGenerate.matrix = []
      DataGenerate.enegy = []
      DataGenerate.status_deads = []
      
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

   def saveResult(scoreLearning, scoreInference, answerlist, matrixSize, matrix, it, distance, dataset, classif, config, time):
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

      #Saving results of some classifiers
      
      def appendClassif(list, name):
         try:
            list.append(classif[name]['score'])
         except:
               print("Erro ao salvar dados do classificador: "+name+" ", sys.exc_info()[0])

      for clf in ['Neural_Net', 'Polynomial_SVM', 'Decision_Tree_3', 'Nearest_Neighbors_3', 'Adaboost_50']:
         appendClassif(list, clf)
         
      # list.append(classif['Neural_Net']['score'])              #Neural Network
      # list.append(classif['Polynomial_SVM']['score'])          #SVM
      # list.append(classif['Decision_Tree_3']['score'])         #Decision Tree
      # list.append(classif['Nearest_Neighbors_3']['score'])     #KNN
      # list.append(classif['Adaboost_50']['score'])             #Adaboost

      list.append(time)

      if config == 'config1':
         DataGenerate.reportList_config1.append(copy.deepcopy(list))
      elif config == 'config2':
         DataGenerate.reportList_config2.append(copy.deepcopy(list))
      elif config == 'config3':
         DataGenerate.reportList_config3.append(copy.deepcopy(list))
      elif config == 'config4':
         DataGenerate.reportList_config4.append(copy.deepcopy(list))
      elif config == 'config5':
         DataGenerate.reportList_config5.append(copy.deepcopy(list))
      elif config == 'config6':
         DataGenerate.reportList_config6.append(copy.deepcopy(list))
      elif config == 'config7':
         DataGenerate.reportList_config7.append(copy.deepcopy(list))
      elif config == 'config8':
         DataGenerate.reportList_config8.append(copy.deepcopy(list))
      elif config == 'config9':
         DataGenerate.reportList_config9.append(copy.deepcopy(list))
      elif config == 'config10':
         DataGenerate.reportList_config10.append(copy.deepcopy(list))
      elif config == 'config11':
         DataGenerate.reportList_config11.append(copy.deepcopy(list))
      elif config == 'config12':
         DataGenerate.reportList_config12.append(copy.deepcopy(list))
      elif config == 'config13':
         DataGenerate.reportList_config13.append(copy.deepcopy(list))
      elif config == 'config14':
         DataGenerate.reportList_config14.append(copy.deepcopy(list))

   def report():
      columns = ['matrix_score_learning', 'matrix_score_inference', 'classif_maior_score', 'classif_menor_score', 'matrix_pct_classif_learning',
       'matrix_pct_classif_inference', 'qtd_mortes_total', 'qtd_mortes_media_por_iteracao', 'qtd_maior_mortes', 'qtd_menor_mortes', 'qtd_erros',
       'qtd_votacao_em_massa_em_erros', 'qtd_votacao_dividida_em_erros', 'matrix_tamanho', 'iteracao_nr', 'distancia', 'dataset', 'Rede Neural', 'SVM',
       'Tree', 'KNN', 'Adaboost', 'tempo']
      df_config1 = pd.DataFrame(DataGenerate.reportList_config1, columns=columns)
      df_config2 = pd.DataFrame(DataGenerate.reportList_config2, columns=columns)
      df_config3 = pd.DataFrame(DataGenerate.reportList_config3, columns=columns)
      df_config4 = pd.DataFrame(DataGenerate.reportList_config4, columns=columns)
      df_config5 = pd.DataFrame(DataGenerate.reportList_config5, columns=columns)
      df_config6 = pd.DataFrame(DataGenerate.reportList_config6, columns=columns)
      df_config7 = pd.DataFrame(DataGenerate.reportList_config7, columns=columns)
      df_config8 = pd.DataFrame(DataGenerate.reportList_config8, columns=columns)
      df_config9 = pd.DataFrame(DataGenerate.reportList_config9, columns=columns)
      df_config10 = pd.DataFrame(DataGenerate.reportList_config10, columns=columns)
      df_config11 = pd.DataFrame(DataGenerate.reportList_config11, columns=columns)
      df_config12 = pd.DataFrame(DataGenerate.reportList_config12, columns=columns)
      df_config13 = pd.DataFrame(DataGenerate.reportList_config13, columns=columns)
      df_config14 = pd.DataFrame(DataGenerate.reportList_config14, columns=columns)

      writer_config1 = pd.ExcelWriter('file/result/result-config1-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config2 = pd.ExcelWriter('file/result/result-config2-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config3 = pd.ExcelWriter('file/result/result-config3-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config4 = pd.ExcelWriter('file/result/result-config4-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config5 = pd.ExcelWriter('file/result/result-config5-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config6 = pd.ExcelWriter('file/result/result-config6-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config7 = pd.ExcelWriter('file/result/result-config7-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config8 = pd.ExcelWriter('file/result/result-config8-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config9 = pd.ExcelWriter('file/result/result-config9-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config10 = pd.ExcelWriter('file/result/result-config10-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config11 = pd.ExcelWriter('file/result/result-config11-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config12 = pd.ExcelWriter('file/result/result-config12-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config13 = pd.ExcelWriter('file/result/result-config13-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')
      writer_config14 = pd.ExcelWriter('file/result/result-config14-'+datetime.today().strftime('%Y%m%d-%H%M')+'.xlsx')

      df_config1.to_excel(writer_config1)
      df_config2.to_excel(writer_config2)
      df_config3.to_excel(writer_config3)
      df_config4.to_excel(writer_config4)
      df_config5.to_excel(writer_config5)
      df_config6.to_excel(writer_config6)
      df_config7.to_excel(writer_config7)
      df_config8.to_excel(writer_config8)
      df_config9.to_excel(writer_config9)
      df_config10.to_excel(writer_config10)
      df_config11.to_excel(writer_config11)
      df_config12.to_excel(writer_config12)
      df_config13.to_excel(writer_config13)
      df_config14.to_excel(writer_config14)

      writer_config1.save()
      writer_config2.save()
      writer_config3.save()
      writer_config4.save()
      writer_config5.save()
      writer_config6.save()
      writer_config7.save()
      writer_config8.save()
      writer_config9.save()
      writer_config10.save()
      writer_config11.save()
      writer_config12.save()
      writer_config13.save()
      writer_config14.save()