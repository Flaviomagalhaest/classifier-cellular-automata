import unittest
import cca, random, csv
from graph import Graph

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.covariance import OAS


#ARRUMAR CLASSES DOS TESTES UNITÁRIOS
class returnNeighboringClassifiersTest(unittest.TestCase):
   def setUp(self):
      self.matrixTest = [
         ["AA", "AB", "AC", "AD", "AE", "AF"],
         ["BA", "BB", "BC", "BD", "BE", "BF"],
         ["CA", "CB", "CC", "CD", "CE", "CF"],
         ["DA", "DB", "DC", "DD", "DE", "DF"],
         ["EA", "EB", "EC", "ED", "EE", "EF"],
         ["FA", "FB", "FC", "FD", "FE", "FF"]
      ]

   def test_retorna_AB_BA_BB_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 0, 0, 1, self.matrixTest)
      self.assertListEqual(["AB","BA","BB"], neighbors)

   def test_retorna_AA_AB_AC_BA_BC_CA_CB_CC_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 1, 1, 1, self.matrixTest)
      self.assertListEqual(["AA","AB","AC","BA","BC","CA","CB","CC"], neighbors)

   def test_retorna_BC_BD_BE_CC_CE_DC_DD_DE_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 2, 3, 1, self.matrixTest)
      self.assertListEqual(["BC","BD","BE","CC","CE","DC","DD","DE"], neighbors)

   def test_retorna_DD_DE_DF_ED_EF_FD_FE_FF_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 4, 4, 1, self.matrixTest)
      self.assertListEqual(["DD","DE","DF","ED","EF","FD","FE","FF"], neighbors)

   def test_retorna_AB_AC_BA_BB_BC_CA_CB_CC_DISTANCE2(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 0, 0, 2, self.matrixTest)
      self.assertListEqual(["AB","AC","BA","BB","BC","CA","CB","CC"], neighbors)

   def test_retorna_AA_AB_AC_AD_BA_BC_BD_CA_CB_CC_CD_DA_DB_DC_DD_DISTANCE2(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 1, 1, 2, self.matrixTest)
      self.assertListEqual(["AA","AB","AC","AD","BA","BC","BD","CA","CB","CC","CD","DA","DB","DC","DD"], neighbors)

   def test_retorna_ABACADAEAF_BBBCBDBEBF_CBCCCECF_DBDCDDDEDF_EBECEDEEEF_DISTANCE2(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 2, 3, 2, self.matrixTest)
      self.assertListEqual(["AB","AC","AD","AE","AF","BB","BC","BD","BE","BF","CB","CC","CE","CF","DB","DC","DD","DE","DF","EB","EC","ED","EE","EF"], neighbors)

   def test_retorna_CCCDCECF_DCDDDEDF_ECEDEF_FCFDFEFF_DISTANCE2(self):
      neighbors = cca.returnNeighboringClassifiers(6, 6, 4, 4, 2, self.matrixTest)
      self.assertListEqual(["CC","CD","CE","CF","DC","DD","DE","DF","EC","ED","EF","FC","FD","FE","FF"], neighbors)

   def test_retorna_AAABAC_BABC_CACBCC_DISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(4, 4, 1, 1, 1, self.matrixTest)
      self.assertListEqual(["AA","AB","AC","BA","BC","CA","CB","CC"], neighbors)

   def test_retorna_AAABACAD_BABCBD_CACBCCCD_DADBDCDDDISTANCE1(self):
      neighbors = cca.returnNeighboringClassifiers(4, 4, 1, 1, 2, self.matrixTest)
      self.assertListEqual(["AA","AB","AC","AD","BA","BC","BD","CA","CB","CC","CD","DA","DB","DC","DD"], neighbors)

class returNeighborsMajorityRight(unittest.TestCase):
   def setUp(self):
      self.neighbors = [
         {'name': 'vizinho1', 'score': 0.814, 'predict': [0, 0, 0], 'energy':2},
         {'name': 'vizinho2', 'score': 0.846, 'predict': [0, 0, 1], 'energy':10},
         {'name': 'vizinho3', 'score': 0.82, 'predict': [1, 0, 1], 'energy':5},
         {'name': 'vizinho4', 'score': 0.82, 'predict': [1, 1, 1], 'energy':7},
         {}, {}
      ]

   def test_retorna_true_majority_neighbors_rigth(self):
      majority = cca.neighborsMajorityRight(self.neighbors, 1, 0)
      self.assertTrue(majority)

   def test_retorna_false_majority_neighbors_wrong(self):
      majority = cca.neighborsMajorityRight(self.neighbors, 2, 0)
      self.assertFalse(majority)

   def test_retorna_false_majority_neighbors_even(self):
      majority = cca.neighborsMajorityRight(self.neighbors, 0, 1)
      self.assertFalse(majority)
   
class testLostEnergyToLive(unittest.TestCase):
   def setUp(self):
      self.matrix = [
         [
            {'name': 'vizinho11', 'score': 0.814, 'predict': [0, 0, 0], 'energy':5},
            {'name': 'vizinho12', 'score': 0.846, 'predict': [0, 0, 1], 'energy':8}],
         [
            {'name': 'vizinho23', 'score': 0.82, 'predict': [1, 0, 1], 'energy':2},
            {'name': 'vizinho24', 'score': 0.82, 'predict': [1, 1, 1], 'energy':0}]
      ]
   
   def test_one_cell_dead(self):
      cca.lostEnergyToLive(self.matrix, 1)
      self.assertEqual(4, self.matrix[0][0]['energy'])
      self.assertEqual(7, self.matrix[0][1]['energy'])
      self.assertEqual(1, self.matrix[1][0]['energy'])
      self.assertEqual(-1, self.matrix[1][1]['energy'])

class testCollectOrRelocateDeadCells(unittest.TestCase):
   def setUp(self):
      self.matrix = [
         [
            {'name': 'vizinho11', 'score': 0.814, 'predict': [0, 0, 0], 'energy':5},
            {'name': 'vizinho12', 'score': 0.846, 'predict': [0, 0, 1], 'energy':8},
            {'name': 'vizinho13', 'score': 0.82, 'predict': [1, 0, 1], 'energy':2}],
         [
            {'name': 'vizinho21', 'score': 0.82, 'predict': [1, 0, 1], 'energy':2},
            {'name': 'vizinho22', 'score': 0.82, 'predict': [1, 1, 1], 'energy':0},
            {'name': 'vizinho23', 'score': 0.82, 'predict': [1, 1, 1], 'energy':5}],
         [
            {'name': 'vizinho31', 'score': 0.82, 'predict': [1, 0, 1], 'energy':-1},
            {'name': 'vizinho32', 'score': 0.82, 'predict': [1, 1, 1], 'energy':4},
            {}]
      ]
      self.classif = {}
      self.classif['teste'] = {"name": "teste"}
      self.classif['teste2'] = {"name": "teste2"}
      self.pool = ["teste", "teste2", "teste3"]

   def test_collect_dead_cells(self):
      cca.collectOrRelocateDeadCells(self.matrix, self.pool, self.classif, False)
      self.assertEqual({}, self.matrix[1][1])
      self.assertEqual({}, self.matrix[2][0])
      self.assertListEqual(["teste", "teste2", "teste3", "vizinho22", "vizinho31"], self.pool)

   def test_relocate_dead_cells(self):
      cca.collectOrRelocateDeadCells(self.matrix, self.pool, self.classif, True)
      self.assertEqual('teste', self.matrix[1][1]['name'])
      self.assertEqual('teste2', self.matrix[2][0]['name'])
      self.assertListEqual(["teste3", "vizinho22", "vizinho31"], self.pool)

class returnMatrixOfIndividualItem(unittest.TestCase):
   def setUp(self):
      self.matrix = [
         [
            {'name': 'vizinho11', 'score': 0.814, 'predict': [0, 0, 0], 'energy':5},
            {'name': 'vizinho12', 'score': 0.846, 'predict': [0, 0, 1], 'energy':8}],
         [
            {'name': 'vizinho23', 'score': 0.82, 'predict': [1, 0, 1], 'energy':2},
            {}]
      ]

   def test_returnEnergyAndObject(self):
      list = cca.returnMatrixOfIndividualItem(self.matrix, 'energy')
      self.assertListEqual([[5,8],[2,0]], list)

class returnListOfWeightedVotes(unittest.TestCase):
   
   def setUp(self):
      self.matrix = [
         [
            {'name': 'vizinho11', 'score': 0.814, 'predict': [0, 0, 0], 'energy':15},
            {'name': 'vizinho12', 'score': 0.846, 'predict': [0, 1, 1], 'energy':15},
            {'name': 'vizinho13', 'score': 0.82, 'predict': [1, 0, 1], 'energy':5}],
         [
            {'name': 'vizinho21', 'score': 0.82, 'predict': [1, 0, 0], 'energy':5},
            {'name': 'vizinho22', 'score': 0.82, 'predict': [1, 1, 1], 'energy':3},
            {'name': 'vizinho23', 'score': 0.82, 'predict': [1, 1, 0], 'energy':3}],
         [
            {'name': 'vizinho31', 'score': 0.82, 'predict': [1, 1, 1], 'energy':5},
            {'name': 'vizinho32', 'score': 0.82, 'predict': [1, 1, 0], 'energy':5},
            {}]
      ]
   
   def test_list_of_weighted_votes(self):
      answers = cca.weightedVote(self.matrix, range(0,3))
      self.assertListEqual([0,1,2], answers)

class returnScore(unittest.TestCase):
   
   def setUp(self):
      self.samples = [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0]
   
   def test_return_100pct(self):
      list = [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0]
      score = cca.returnScore(self.samples, list)
      self.assertTrue(score == 1)

   def test_return_90pct(self):
      list = [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1]
      score = cca.returnScore(self.samples, list)
      self.assertTrue(score == 0.9)

   def test_return_70pct(self):
      list = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
      score = cca.returnScore(self.samples, list)
      self.assertTrue(score == 0.7)

   def test_return_50pct(self):
      list = [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1]
      score = cca.returnScore(self.samples, list)
      self.assertTrue(score == 0.5)

   def test_return_46pct(self):
      list = [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1]
      score = cca.returnScore(self.samples, list)
      self.assertTrue(score == 0.4666666666666667)

   def test_return_0pct(self):
      list = [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1]
      score = cca.returnScore(self.samples, list)
      self.assertTrue(score == 0)

class neighborsMajorityClassify(unittest.TestCase):
   def setUp(self):
      obj1={'name': 'QDA', 'predict': [0, 1, 1, 0, 0], 'score': 0.91, 'energy': 100}
      obj2={'name': 'Random_Forest_12_100', 'predict': [0, 1, 1, 1, 0], 'score': 0.908, 'energy': 50}
      obj3={'name': 'LinearSVC_l2', 'predict': [0, 1, 0, 1, 0], 'score': 0.904, 'energy': 200}
      obj4={'name': 'teste', 'predict': [1, 1, 0, 0, 1], 'score': 0.904, 'energy': 25}
      self.neighbors = [obj1, obj2, obj3, obj4]
   
   def test_returnValueZero(self):
      response = cca.neighborsMajorityClassify(self.neighbors, 0)
      self.assertTrue(response == 0)

   def test_returnValueOne(self):
      response = cca.neighborsMajorityClassify(self.neighbors, 1)
      self.assertTrue(response == 1)
   
   def test_returnValueZeroBecauseEnergy(self):
      response = cca.neighborsMajorityClassify(self.neighbors, 2)
      self.assertTrue(response == 0)

   def test_returnValueOneBecauseEnergy(self):
      response = cca.neighborsMajorityClassify(self.neighbors, 3)
      self.assertTrue(response == 1)

class neighborsEnergyAverage(unittest.TestCase):
   def setUp(self):
      obj1={'name': 'QDA', 'predict': [0, 1, 1, 0, 0], 'score': 0.91, 'energy': 100}
      obj2={'name': 'Random_Forest_12_100', 'predict': [0, 1, 1, 1, 0], 'score': 0.908, 'energy': 300}
      obj3={'name': 'LinearSVC_l2', 'predict': [0, 1, 0, 1, 0], 'score': 0.904, 'energy': 200}
      obj4={}
      self.neighbors = [obj1, obj2, obj3, obj4]
   
   def test_returnAverageWith3Neighbors(self):
      response = cca.neighborsEnergyAverage(self.neighbors)
      self.assertTrue(response == 200)

class neighborsMajorityEnergy(unittest.TestCase):
   def setUp(self):
      obj1={'name': 'QDA', 'predict': [0, 1, 1, 0, 0], 'score': 0.91, 'energy': 100}
      obj2={'name': 'Random_Forest_12_100', 'predict': [0, 1, 1, 1, 0], 'score': 0.908, 'energy': 300}
      obj3={'name': 'LinearSVC_l2', 'predict': [0, 1, 0, 1, 0], 'score': 0.904, 'energy': 200}
      obj4={}
      self.neighbors = [obj1, obj2, obj3, obj4]

   def test_returnEnergyToPredictZero(self):
      energyWhoVoteZero, energyWhoVoteOne, energyTotal = cca.neighborsMajorityEnergy(self.neighbors, 0)
      self.assertTrue(energyWhoVoteZero == 600)
      self.assertTrue(energyWhoVoteOne == 0)
      self.assertTrue(energyTotal == 600)

   def test_returnEnergyToPredictOne(self):
      energyWhoVoteZero, energyWhoVoteOne, energyTotal = cca.neighborsMajorityEnergy(self.neighbors, 1)
      self.assertTrue(energyWhoVoteZero == 0)
      self.assertTrue(energyWhoVoteOne == 600)
      self.assertTrue(energyTotal == 600)

   def test_returnEnergyToPredictOneAndZero(self):
      energyWhoVoteZero, energyWhoVoteOne, energyTotal = cca.neighborsMajorityEnergy(self.neighbors, 2)
      self.assertTrue(energyWhoVoteZero == 200)
      self.assertTrue(energyWhoVoteOne == 400)
      self.assertTrue(energyTotal == 600)

class weightedVoteforSample(unittest.TestCase):
   def setUp(self):
      obj1={'name': 'QDA', 'predict': [0, 1, 1, 0, 0], 'score': 0.91, 'energy': 100}
      obj2={'name': 'Random_Forest_12_100', 'predict': [0, 1, 1, 1, 1], 'score': 0.908, 'energy': 300}
      obj3={'name': 'LinearSVC_l2', 'predict': [0, 1, 0, 1, 0], 'score': 0.904, 'energy': 200}
      obj4={}
      self.neighbors1 = [obj1, obj2]
      self.neighbors2 = [obj3, obj4]
      self.matrix = [self.neighbors1, self.neighbors2]

   def test_returnVote0(self):
      response = cca.weightedVoteforSample(self.matrix, 0)
      self.assertTrue(response == 0)

   def test_returnVote1(self):
      response = cca.weightedVoteforSample(self.matrix, 1)
      self.assertTrue(response == 1)

   def test_returnVote1Divided(self):
      response = cca.weightedVoteforSample(self.matrix, 2)
      self.assertTrue(response == 1)

   def test_returnVote1Divided2(self):
      response = cca.weightedVoteforSample(self.matrix, 3)
      self.assertTrue(response == 1)

   def test_returnVote2(self):
      response = cca.weightedVoteforSample(self.matrix, 4)
      self.assertTrue(response == 2)


class testPlots(unittest.TestCase):
   def setUp(self):
      obj1={'name': 'QDA', 'predict': [0, 1, 1, 0, 0], 'score': 0.91, 'energy': 100}
      obj2={'name': 'Random_Forest_12_100', 'predict': [0, 1, 1, 1, 1], 'score': 0.908, 'energy': 300}
      obj3={'name': 'LinearSVC_l2', 'predict': [0, 1, 0, 1, 0], 'score': 0.904, 'energy': 200}
      obj4={}
      self.neighbors1 = [obj1, obj2]
      self.neighbors2 = [obj3, obj4]
      self.matrix = [self.neighbors1, self.neighbors2]

   @unittest.skip("Graphic tests")
   def test_testInteractiveMatrix(self):
      Graph()
      Graph.initMatrix(self.matrix)
      Graph.printMatrixInteractiveEnergy(self.matrix)
      Graph.printMatrixInteractiveEnergy(self.matrix)
      self.matrix[0][0]['energy'] = 700
      self.matrix[0][1]['energy'] = 900
      self.matrix[1][0]['energy'] = 50
      Graph.printMatrixInteractiveEnergy(self.matrix)
      a='a'
   
   @unittest.skip("Graphic tests")
   def test_classifierEnergyBar(self):
      data = {}
      data['QDA'] = {'name': 'QDA', 'predict': [1, 0, 1, 0, 0, 1], 'prob': [], 'score': 0.6275555555555555, 'energy': 100}
      data['LDA'] = {'name': 'LDA', 'predict': [1, 0, 1, 0, 1], 'prob': [], 'score': 0.8164444444444444, 'energy': 100}
      data['Naive_Bayes'] = {'name': 'Naive_Bayes', 'predict': [1, 0, 1, 1, 0, 1], 'prob': [], 'score': 0.8186666666666667, 'energy': 100}
      Graph()
      Graph.initBar(data)
      data['QDA']['energy'] = 3000
      Graph.printBar()
      a='a'

class testsClassifiers(unittest.TestCase):
   def test_classifier(self):
      with open('dataset/jm1.csv', newline='') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
         csvCount = 0
         jm1 = [row for nr, row in enumerate(spamreader)]
         random.shuffle(jm1)
         jm1_test = jm1[0:1000]
         jm1_train = jm1[1000:2000]

      Y_train = [j.pop(-1) for j in jm1_train]
      Y_train = [1 if x=='true' else 0 for x in Y_train]
      X_train = []
      for jt in jm1_train:
         X_train.append([float(j) for j in jt])

      Y_test = [j.pop(-1) for j in jm1_test]
      Y_test = [1 if x=='true' else 0 for x in Y_test]
      X_test = []
      for jt in jm1_test:
         X_test.append([float(j) for j in jt])

      from sklearn.linear_model import PassiveAggressiveClassifier
      import pandas as pd
      import copy

      def buildMatrix(listClassifiers):
         matrix = []
         for c1 in listClassifiers:
            line = []
            for c2 in listClassifiers:
               count = 0
               for i in range(len(c2)):
                  if c1[i] != c2 [i]:
                     count += 1
               line.append(count)
            matrix.append(copy.deepcopy(line))
         return matrix

      a000 = PassiveAggressiveClassifier()
      a001 = PassiveAggressiveClassifier()
      a002 = PassiveAggressiveClassifier()
      a003 = PassiveAggressiveClassifier()
      a004 = PassiveAggressiveClassifier()
      a005 = PassiveAggressiveClassifier()
      a006 = PassiveAggressiveClassifier()
      a007 = PassiveAggressiveClassifier()
      a008 = PassiveAggressiveClassifier()
      a009 = PassiveAggressiveClassifier()
      a010 = PassiveAggressiveClassifier()
      a011 = PassiveAggressiveClassifier()
      a012 = PassiveAggressiveClassifier()
      a013 = PassiveAggressiveClassifier()
      a014 = PassiveAggressiveClassifier()
      a015 = PassiveAggressiveClassifier()
      a016 = PassiveAggressiveClassifier()
      a017 = PassiveAggressiveClassifier()
      a018 = PassiveAggressiveClassifier()
      a019 = PassiveAggressiveClassifier()
      a020 = PassiveAggressiveClassifier()
      a021 = PassiveAggressiveClassifier()
      a022 = PassiveAggressiveClassifier()
      a023 = PassiveAggressiveClassifier()
      a024 = PassiveAggressiveClassifier()
      a025 = PassiveAggressiveClassifier()
      a026 = PassiveAggressiveClassifier()
      a027 = PassiveAggressiveClassifier()
      a028 = PassiveAggressiveClassifier()
      a029 = PassiveAggressiveClassifier()

      a000.fit(X_train, Y_train)
      a001.fit(X_train, Y_train)
      a002.fit(X_train, Y_train)
      a003.fit(X_train, Y_train)
      a004.fit(X_train, Y_train)
      a005.fit(X_train, Y_train)
      a006.fit(X_train, Y_train)
      a007.fit(X_train, Y_train)
      a008.fit(X_train, Y_train)
      a009.fit(X_train, Y_train)
      a010.fit(X_train, Y_train)
      a011.fit(X_train, Y_train)
      a012.fit(X_train, Y_train)
      a013.fit(X_train, Y_train)
      a014.fit(X_train, Y_train)
      a015.fit(X_train, Y_train)
      a016.fit(X_train, Y_train)
      a017.fit(X_train, Y_train)
      a018.fit(X_train, Y_train)
      a019.fit(X_train, Y_train)
      a020.fit(X_train, Y_train)
      a021.fit(X_train, Y_train)
      a022.fit(X_train, Y_train)
      a023.fit(X_train, Y_train)
      a024.fit(X_train, Y_train)
      a025.fit(X_train, Y_train)
      a026.fit(X_train, Y_train)
      a027.fit(X_train, Y_train)
      a028.fit(X_train, Y_train)
      a029.fit(X_train, Y_train)               
      
      list = []
      list.append(a000.predict(X_test))
      list.append(a001.predict(X_test))
      list.append(a002.predict(X_test))
      list.append(a003.predict(X_test))
      list.append(a004.predict(X_test))
      list.append(a005.predict(X_test))
      list.append(a006.predict(X_test))
      list.append(a007.predict(X_test))
      list.append(a008.predict(X_test))
      list.append(a009.predict(X_test))
      list.append(a010.predict(X_test))
      list.append(a011.predict(X_test))
      list.append(a012.predict(X_test))
      list.append(a013.predict(X_test))
      list.append(a014.predict(X_test))
      list.append(a015.predict(X_test))
      list.append(a016.predict(X_test))
      list.append(a017.predict(X_test))
      list.append(a018.predict(X_test))
      list.append(a019.predict(X_test))
      list.append(a020.predict(X_test))
      list.append(a021.predict(X_test))
      list.append(a022.predict(X_test))
      list.append(a023.predict(X_test))
      list.append(a024.predict(X_test))
      list.append(a025.predict(X_test))
      list.append(a026.predict(X_test))
      list.append(a027.predict(X_test))
      list.append(a028.predict(X_test))
      list.append(a029.predict(X_test))

      matrix = buildMatrix(list)
      print(pd.DataFrame(matrix))
      
      # print(a01.score(X_test, Y_test))
      # print(a02.score(X_test, Y_test))
      # print(a03.score(X_test, Y_test))
      # print(a04.score(X_test, Y_test))
      # print(a05.score(X_test, Y_test))
      # print(a06.score(X_test, Y_test))
      # print(a07.score(X_test, Y_test))
      # print(a08.score(X_test, Y_test)) 


      a = 'a'


if __name__ == "__main__":
   unittest.main()