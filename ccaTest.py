import unittest
import cca

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
         {'name': 'vizinho1', 'score': 0.814, 'predict': [0, 0, 0]},
         {'name': 'vizinho2', 'score': 0.846, 'predict': [0, 0, 1]},
         {'name': 'vizinho3', 'score': 0.82, 'predict': [1, 0, 1]},
         {'name': 'vizinho4', 'score': 0.82, 'predict': [1, 1, 1]},
         {}, {}
      ]

   def test_retorna_true_majority_neighbors_rigth(self):
      response = cca.neighborsMajorityRight(self.neighbors, 1, 0)
      self.assertTrue(response)

   def test_retorna_false_majority_neighbors_wrong(self):
      response = cca.neighborsMajorityRight(self.neighbors, 2, 0)
      self.assertFalse(response)

   def test_retorna_false_majority_neighbors_even(self):
      response = cca.neighborsMajorityRight(self.neighbors, 0, 1)
      self.assertFalse(response)
   
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
      cca.lostEnergyToLive(self.matrix, 1, self.pool)
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
      answers = cca.weightedVote(self.matrix, 3)
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


# class testsClassifiers(unittest.TestCase):
#    def test_classifier(self):
#       X, Y = make_classification(n_samples=1000, n_classes=2, n_features=5, n_redundant=0, random_state=1)
#       X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
#       clf1 = SGDClassifier(loss="hinge", penalty="l2")
#       clf2 = SGDClassifier(loss="log")
#       clf3 = SGDClassifier(loss="modified_huber")
#       clf4 = SGDClassifier(loss="squared_hinge")
#       clf5 = SGDClassifier(loss="perceptron")
#       clf6 = SGDClassifier(loss="squared_loss")
#       clf7 = SGDClassifier(loss='huber')
#       clf8 = SGDClassifier(loss='epsilon_insensitive')
#       # clf10 = SVC(kernel='sigmoid')
#       clf1.fit(X_train, Y_train)
#       clf2.fit(X_train, Y_train)
#       clf3.fit(X_train, Y_train)
#       clf4.fit(X_train, Y_train)
#       clf5.fit(X_train, Y_train)
#       clf6.fit(X_train, Y_train)
#       clf7.fit(X_train, Y_train)
#       clf8.fit(X_train, Y_train)
#       clf9.fit(X_train, Y_train)
#       # clf10.fit(X_train, Y_train)
#       print(clf1.score(X_test, Y_test))
#       print(clf2.score(X_test, Y_test))
#       print(clf3.score(X_test, Y_test))
#       print(clf4.score(X_test, Y_test))
#       print(clf5.score(X_test, Y_test))
#       print(clf6.score(X_test, Y_test))
#       print(clf7.score(X_test, Y_test))
#       print(clf8.score(X_test, Y_test))
#       print(clf9.score(X_test, Y_test))
#       # print(clf10.score(X_test, Y_test))
#       # print(clf1.predict(X_test)[0:10])
#       # print(clf2.predict(X_test)[0:10])
#       # print(clf3.predict(X_test)[0:10])
#       # print(clf4.predict(X_test)[0:10])


if __name__ == "__main__":
   unittest.main()