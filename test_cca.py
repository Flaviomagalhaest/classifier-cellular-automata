import copy
import csv
import random
import unittest

import pandas as pd

import cca
from classifiers import Classifiers
from graph import Graph


# ARRUMAR CLASSES DOS TESTES UNITÁRIOS
class returnNeighboringClassifiersTest(unittest.TestCase):
    def setUp(self):
        self.matrixTest = [
            ["AA", "AB", "AC", "AD", "AE", "AF"],
            ["BA", "BB", "BC", "BD", "BE", "BF"],
            ["CA", "CB", "CC", "CD", "CE", "CF"],
            ["DA", "DB", "DC", "DD", "DE", "DF"],
            ["EA", "EB", "EC", "ED", "EE", "EF"],
            ["FA", "FB", "FC", "FD", "FE", "FF"],
        ]

    def test_retorna_AB_BA_BB_DISTANCE1(self):
        neighbors = cca.returnNeighboringClassifiers(
            6,
            6,
            0,
            0,
            1,
            self.matrixTest,
        )
        self.assertListEqual(["AB", "BA", "BB"], neighbors)

    def test_retorna_AA_AB_AC_BA_BC_CA_CB_CC_DISTANCE1(self):
        neighbors = cca.returnNeighboringClassifiers(
            6,
            6,
            1,
            1,
            1,
            self.matrixTest,
        )
        self.assertListEqual(
            ["AA", "AB", "AC", "BA", "BC", "CA", "CB", "CC"], neighbors
        )

    def test_retorna_BC_BD_BE_CC_CE_DC_DD_DE_DISTANCE1(self):
        neighbors = cca.returnNeighboringClassifiers(
            6,
            6,
            2,
            3,
            1,
            self.matrixTest,
        )
        self.assertListEqual(
            ["BC", "BD", "BE", "CC", "CE", "DC", "DD", "DE"], neighbors
        )

    def test_retorna_DD_DE_DF_ED_EF_FD_FE_FF_DISTANCE1(self):
        neighbors = cca.returnNeighboringClassifiers(
            6,
            6,
            4,
            4,
            1,
            self.matrixTest,
        )
        self.assertListEqual(
            ["DD", "DE", "DF", "ED", "EF", "FD", "FE", "FF"], neighbors
        )

    def test_retorna_AB_AC_BA_BB_BC_CA_CB_CC_DISTANCE2(self):
        neighbors = cca.returnNeighboringClassifiers(
            6,
            6,
            0,
            0,
            2,
            self.matrixTest,
        )
        self.assertListEqual(
            ["AB", "AC", "BA", "BB", "BC", "CA", "CB", "CC"], neighbors
        )

    def test_retorna_AA_AB_AC_AD_BA_BC_BD_CA_CB_CC_CD_DA_DB_DC_DD_DISTANCE2(
        self,
    ):
        neighbors = cca.returnNeighboringClassifiers(
            6,
            6,
            1,
            1,
            2,
            self.matrixTest,
        )
        self.assertListEqual(
            [
                "AA",
                "AB",
                "AC",
                "AD",
                "BA",
                "BC",
                "BD",
                "CA",
                "CB",
                "CC",
                "CD",
                "DA",
                "DB",
                "DC",
                "DD",
            ],
            neighbors,
        )

    def test_ABACADAEAF_BBBCBDBEBF_CBCCCECF_DBDCDDDEDF_EBECEDEEEF_DIST2(
        self,
    ):
        neighbors = cca.returnNeighboringClassifiers(
            6,
            6,
            2,
            3,
            2,
            self.matrixTest,
        )
        self.assertListEqual(
            [
                "AB",
                "AC",
                "AD",
                "AE",
                "AF",
                "BB",
                "BC",
                "BD",
                "BE",
                "BF",
                "CB",
                "CC",
                "CE",
                "CF",
                "DB",
                "DC",
                "DD",
                "DE",
                "DF",
                "EB",
                "EC",
                "ED",
                "EE",
                "EF",
            ],
            neighbors,
        )

    def test_retorna_CCCDCECF_DCDDDEDF_ECEDEF_FCFDFEFF_DISTANCE2(self):
        neighbors = cca.returnNeighboringClassifiers(
            6,
            6,
            4,
            4,
            2,
            self.matrixTest,
        )
        self.assertListEqual(
            [
                "CC",
                "CD",
                "CE",
                "CF",
                "DC",
                "DD",
                "DE",
                "DF",
                "EC",
                "ED",
                "EF",
                "FC",
                "FD",
                "FE",
                "FF",
            ],
            neighbors,
        )

    def test_retorna_AAABAC_BABC_CACBCC_DISTANCE1(self):
        neighbors = cca.returnNeighboringClassifiers(
            4,
            4,
            1,
            1,
            1,
            self.matrixTest,
        )
        self.assertListEqual(
            ["AA", "AB", "AC", "BA", "BC", "CA", "CB", "CC"], neighbors
        )

    def test_retorna_AAABACAD_BABCBD_CACBCCCD_DADBDCDDDISTANCE1(self):
        neighbors = cca.returnNeighboringClassifiers(
            4,
            4,
            1,
            1,
            2,
            self.matrixTest,
        )
        self.assertListEqual(
            [
                "AA",
                "AB",
                "AC",
                "AD",
                "BA",
                "BC",
                "BD",
                "CA",
                "CB",
                "CC",
                "CD",
                "DA",
                "DB",
                "DC",
                "DD",
            ],
            neighbors,
        )


class returNeighborsMajorityRight(unittest.TestCase):
    def setUp(self):
        self.neighbors = [
            {
                "name": "vizinho1",
                "score": 0.814,
                "predict": [0, 0, 0],
                "energy": 2,
            },
            {
                "name": "vizinho2",
                "score": 0.846,
                "predict": [0, 0, 1],
                "energy": 10,
            },
            {
                "name": "vizinho3",
                "score": 0.82,
                "predict": [1, 0, 1],
                "energy": 5,
            },
            {
                "name": "vizinho4",
                "score": 0.82,
                "predict": [1, 1, 1],
                "energy": 7,
            },
            {},
            {},
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
                {
                    "name": "vizinho11",
                    "score": 0.814,
                    "predict": [0, 0, 0],
                    "energy": 5,
                },
                {
                    "name": "vizinho12",
                    "score": 0.846,
                    "predict": [0, 0, 1],
                    "energy": 8,
                },
            ],
            [
                {
                    "name": "vizinho23",
                    "score": 0.82,
                    "predict": [1, 0, 1],
                    "energy": 2,
                },
                {
                    "name": "vizinho24",
                    "score": 0.82,
                    "predict": [1, 1, 1],
                    "energy": 0,
                },
            ],
        ]

    def test_one_cell_dead(self):
        cca.lostEnergyToLive(self.matrix, 1)
        self.assertEqual(4, self.matrix[0][0]["energy"])
        self.assertEqual(7, self.matrix[0][1]["energy"])
        self.assertEqual(1, self.matrix[1][0]["energy"])
        self.assertEqual(-1, self.matrix[1][1]["energy"])


class testCollectOrRelocateDeadCells(unittest.TestCase):
    def setUp(self):
        self.matrix = [
            [
                {
                    "name": "vizinho11",
                    "score": 0.814,
                    "predict": [0, 0, 0],
                    "energy": 5,
                },
                {
                    "name": "vizinho12",
                    "score": 0.846,
                    "predict": [0, 0, 1],
                    "energy": 8,
                },
                {
                    "name": "vizinho13",
                    "score": 0.82,
                    "predict": [1, 0, 1],
                    "energy": 2,
                },
            ],
            [
                {
                    "name": "vizinho21",
                    "score": 0.82,
                    "predict": [1, 0, 1],
                    "energy": 2,
                },
                {
                    "name": "vizinho22",
                    "score": 0.82,
                    "predict": [1, 1, 1],
                    "energy": 0,
                },
                {
                    "name": "vizinho23",
                    "score": 0.82,
                    "predict": [1, 1, 1],
                    "energy": 5,
                },
            ],
            [
                {
                    "name": "vizinho31",
                    "score": 0.82,
                    "predict": [1, 0, 1],
                    "energy": -1,
                },
                {
                    "name": "vizinho32",
                    "score": 0.82,
                    "predict": [1, 1, 1],
                    "energy": 4,
                },
                {},
            ],
        ]
        self.classif = {}
        self.classif["teste"] = {"name": "teste"}
        self.classif["teste2"] = {"name": "teste2"}
        self.pool = ["teste", "teste2", "teste3"]

    def test_collect_dead_cells(self):
        cca.collectOrRelocateDeadCells(
            self.matrix,
            self.pool,
            self.classif,
            False,
        )
        self.assertEqual({}, self.matrix[1][1])
        self.assertEqual({}, self.matrix[2][0])
        self.assertListEqual(
            ["teste", "teste2", "teste3", "vizinho22", "vizinho31"], self.pool
        )

    def test_relocate_dead_cells(self):
        cca.collectOrRelocateDeadCells(
            self.matrix,
            self.pool,
            self.classif,
            True,
        )
        self.assertEqual("teste", self.matrix[1][1]["name"])
        self.assertEqual("teste2", self.matrix[2][0]["name"])
        self.assertListEqual(["teste3", "vizinho22", "vizinho31"], self.pool)


class returnMatrixOfIndividualItem(unittest.TestCase):
    def setUp(self):
        self.matrix = [
            [
                {
                    "name": "vizinho11",
                    "score": 0.814,
                    "predict": [0, 0, 0],
                    "energy": 5,
                },
                {
                    "name": "vizinho12",
                    "score": 0.846,
                    "predict": [0, 0, 1],
                    "energy": 8,
                },
            ],
            [
                {
                    "name": "vizinho23",
                    "score": 0.82,
                    "predict": [1, 0, 1],
                    "energy": 2,
                },
                {},
            ],
        ]

    def test_returnEnergyAndObject(self):
        list = cca.returnMatrixOfIndividualItem(self.matrix, "energy")
        self.assertListEqual([[5, 8], [2, 0]], list)


class returnListOfWeightedVotes(unittest.TestCase):
    def setUp(self):
        self.matrix = [
            [
                {
                    "name": "vizinho11",
                    "score": 0.814,
                    "predict": [0, 0, 0],
                    "energy": 15,
                },
                {
                    "name": "vizinho12",
                    "score": 0.846,
                    "predict": [0, 1, 1],
                    "energy": 15,
                },
                {
                    "name": "vizinho13",
                    "score": 0.82,
                    "predict": [1, 0, 1],
                    "energy": 5,
                },
            ],
            [
                {
                    "name": "vizinho21",
                    "score": 0.82,
                    "predict": [1, 0, 0],
                    "energy": 5,
                },
                {
                    "name": "vizinho22",
                    "score": 0.82,
                    "predict": [1, 1, 1],
                    "energy": 3,
                },
                {
                    "name": "vizinho23",
                    "score": 0.82,
                    "predict": [1, 1, 0],
                    "energy": 3,
                },
            ],
            [
                {
                    "name": "vizinho31",
                    "score": 0.82,
                    "predict": [1, 1, 1],
                    "energy": 5,
                },
                {
                    "name": "vizinho32",
                    "score": 0.82,
                    "predict": [1, 1, 0],
                    "energy": 5,
                },
                {},
            ],
        ]

    def test_list_of_weighted_votes(self):
        answers = cca.weightedVote(self.matrix, range(0, 3))
        self.assertListEqual([0, 1, 2], answers)


class returnScore(unittest.TestCase):
    def setUp(self):
        self.samples = [
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ]

    def test_return_100pct(self):
        list = [
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ]
        score = cca.returnScore(self.samples, list)
        self.assertTrue(score == 1)

    def test_return_90pct(self):
        list = [
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
        ]
        score = cca.returnScore(self.samples, list)
        self.assertTrue(score == 0.9)

    def test_return_70pct(self):
        list = [
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
        ]
        score = cca.returnScore(self.samples, list)
        self.assertTrue(score == 0.7)

    def test_return_50pct(self):
        list = [
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
        ]
        score = cca.returnScore(self.samples, list)
        self.assertTrue(score == 0.5)

    def test_return_46pct(self):
        list = [
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
        ]
        score = cca.returnScore(self.samples, list)
        self.assertTrue(score == 0.4666666666666667)

    def test_return_0pct(self):
        list = [
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
        ]
        score = cca.returnScore(self.samples, list)
        self.assertTrue(score == 0)


class neighborsMajorityClassify(unittest.TestCase):
    def setUp(self):
        obj1 = {
            "name": "QDA",
            "predict": [0, 1, 1, 0, 0],
            "score": 0.91,
            "energy": 100,
        }
        obj2 = {
            "name": "Random_Forest_12_100",
            "predict": [0, 1, 1, 1, 0],
            "score": 0.908,
            "energy": 50,
        }
        obj3 = {
            "name": "LinearSVC_l2",
            "predict": [0, 1, 0, 1, 0],
            "score": 0.904,
            "energy": 200,
        }
        obj4 = {
            "name": "teste",
            "predict": [1, 1, 0, 0, 1],
            "score": 0.904,
            "energy": 25,
        }
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
        obj1 = {
            "name": "QDA",
            "predict": [0, 1, 1, 0, 0],
            "score": 0.91,
            "energy": 100,
        }
        obj2 = {
            "name": "Random_Forest_12_100",
            "predict": [0, 1, 1, 1, 0],
            "score": 0.908,
            "energy": 300,
        }
        obj3 = {
            "name": "LinearSVC_l2",
            "predict": [0, 1, 0, 1, 0],
            "score": 0.904,
            "energy": 200,
        }
        obj4 = {}
        self.neighbors = [obj1, obj2, obj3, obj4]

    def test_returnAverageWith3Neighbors(self):
        response = cca.neighborsEnergyAverage(self.neighbors)
        self.assertTrue(response == 200)


class neighborsMajorityEnergy(unittest.TestCase):
    def setUp(self):
        obj1 = {
            "name": "QDA",
            "predict": [0, 1, 1, 0, 0],
            "score": 0.91,
            "energy": 100,
        }
        obj2 = {
            "name": "Random_Forest_12_100",
            "predict": [0, 1, 1, 1, 0],
            "score": 0.908,
            "energy": 300,
        }
        obj3 = {
            "name": "LinearSVC_l2",
            "predict": [0, 1, 0, 1, 0],
            "score": 0.904,
            "energy": 200,
        }
        obj4 = {}
        self.neighbors = [obj1, obj2, obj3, obj4]

    def test_returnEnergyToPredictZero(self):
        eWhoVoteZero, eWhoVoteOne, eTotal = cca.neighborsMajorityEnergy(
            self.neighbors, 0
        )
        self.assertTrue(eWhoVoteZero == 600)
        self.assertTrue(eWhoVoteOne == 0)
        self.assertTrue(eTotal == 600)

    def test_returnEnergyToPredictOne(self):
        eWhoVoteZero, eWhoVoteOne, eTotal = cca.neighborsMajorityEnergy(
            self.neighbors, 1
        )
        self.assertTrue(eWhoVoteZero == 0)
        self.assertTrue(eWhoVoteOne == 600)
        self.assertTrue(eTotal == 600)

    def test_returnEnergyToPredictOneAndZero(self):
        eWhoVoteZero, eWhoVoteOne, eTotal = cca.neighborsMajorityEnergy(
            self.neighbors, 2
        )
        self.assertTrue(eWhoVoteZero == 200)
        self.assertTrue(eWhoVoteOne == 400)
        self.assertTrue(eTotal == 600)


class weightedVoteforSample(unittest.TestCase):
    def setUp(self):
        obj1 = {
            "name": "QDA",
            "predict": [0, 1, 1, 0, 0],
            "score": 0.91,
            "energy": 100,
        }
        obj2 = {
            "name": "Random_Forest_12_100",
            "predict": [0, 1, 1, 1, 1],
            "score": 0.908,
            "energy": 300,
        }
        obj3 = {
            "name": "LinearSVC_l2",
            "predict": [0, 1, 0, 1, 0],
            "score": 0.904,
            "energy": 200,
        }
        obj4 = {}
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
        obj1 = {
            "name": "QDA",
            "predict": [0, 1, 1, 0, 0],
            "score": 0.91,
            "energy": 100,
        }
        obj2 = {
            "name": "Random_Forest_12_100",
            "predict": [0, 1, 1, 1, 1],
            "score": 0.908,
            "energy": 300,
        }
        obj3 = {
            "name": "LinearSVC_l2",
            "predict": [0, 1, 0, 1, 0],
            "score": 0.904,
            "energy": 200,
        }
        obj4 = {}
        self.neighbors1 = [obj1, obj2]
        self.neighbors2 = [obj3, obj4]
        self.matrix = [self.neighbors1, self.neighbors2]

    @unittest.skip("Graphic tests")
    def test_testInteractiveMatrix(self):
        Graph()
        Graph.initMatrix(self.matrix)
        Graph.printMatrixInteractiveEnergy(self.matrix)
        Graph.printMatrixInteractiveEnergy(self.matrix)
        self.matrix[0][0]["energy"] = 700
        self.matrix[0][1]["energy"] = 900
        self.matrix[1][0]["energy"] = 50
        Graph.printMatrixInteractiveEnergy(self.matrix)

    @unittest.skip("Graphic tests")
    def test_classifierEnergyBar(self):
        data = {}
        data["QDA"] = {
            "name": "QDA",
            "predict": [1, 0, 1, 0, 0, 1],
            "prob": [],
            "score": 0.6275555555555555,
            "energy": 100,
        }
        data["LDA"] = {
            "name": "LDA",
            "predict": [1, 0, 1, 0, 1],
            "prob": [],
            "score": 0.8164444444444444,
            "energy": 100,
        }
        data["Naive_Bayes"] = {
            "name": "Naive_Bayes",
            "predict": [1, 0, 1, 1, 0, 1],
            "prob": [],
            "score": 0.8186666666666667,
            "energy": 100,
        }
        Graph()
        Graph.initBar(data)
        data["QDA"]["energy"] = 3000
        Graph.printBar()


class testsClassifiers(unittest.TestCase):
    def _createBase(self, trainSamples):
        X_test = []
        X_train = []
        Y_train = []
        with open("dataset/jm1.csv", newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            jm1 = [row for nr, row in enumerate(spamreader)]
            jm1_true = [j for j in jm1 if j[21] == "true"]
            jm1_false = [j for j in jm1 if j[21] == "false"]
            random.shuffle(jm1_true)
            random.shuffle(jm1_false)
            trainPart = int(
                trainSamples / 2
            )  # Test sample divided between true answers and false answers
            jm1_train = jm1_true[:trainPart]
            jm1_train = jm1_train + jm1_false[:trainPart]
            # jm1_test = jm1_true[trainPart:trainSamples]
            # jm1_test = jm1_test + jm1_false[trainPart:trainSamples]
            jm1_test = jm1_true[trainPart:]
            jm1_test = jm1_test + jm1_false[trainPart:]
            random.shuffle(jm1_train)
            random.shuffle(jm1_test)
            jm1_test = jm1_test[:trainSamples]

        Y_train = [j.pop(-1) for j in jm1_train]
        Y_train = [1 if x == "true" else 0 for x in Y_train]

        for jt in jm1_train:
            X_train.append([float(j) for j in jt])

        Y_test = [j.pop(-1) for j in jm1_test]
        Y_test = [1 if x == "true" else 0 for x in Y_test]
        for jt in jm1_test:
            X_test.append([float(j) for j in jt])

        return X_test, X_train, Y_train, Y_test

    def _buildMatrix(self, listClassifiers):
        matrix = []
        for c1 in listClassifiers:
            line = []
            for c2 in listClassifiers:
                count = 0
                for i in range(len(c2)):
                    if c1[i] != c2[i]:
                        count += 1
                line.append(count)
            matrix.append(copy.deepcopy(line))
        return matrix

    @unittest.skip("Classifier")
    def buildMatrix(listClassifiers):
        matrix = []
        for c1 in listClassifiers:
            line = []
            for c2 in listClassifiers:
                count = 0
                for i in range(len(c2)):
                    if c1[i] != c2[i]:
                        count += 1
                line.append(count)
            matrix.append(copy.deepcopy(line))
        return matrix

    @unittest.skip("Classifier")
    def test_matriz_diff_LogisticRegression(self):
        trainSamples = 100
        with open("dataset/jm1.csv", newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            jm1 = [row for nr, row in enumerate(spamreader)]
            jm1_true = [j for j in jm1 if j[21] == "true"]
            jm1_false = [j for j in jm1 if j[21] == "false"]
            random.shuffle(jm1_true)
            random.shuffle(jm1_false)
            trainPart = int(
                trainSamples / 2
            )  # Test sample divided between true answers and false answers
            jm1_train = jm1_true[:trainPart]
            jm1_train = jm1_train + jm1_false[:trainPart]
            # jm1_test = jm1_true[trainPart:trainSamples]
            # jm1_test = jm1_test + jm1_false[trainPart:trainSamples]
            jm1_test = jm1_true[trainPart:]
            jm1_test = jm1_test + jm1_false[trainPart:]
            random.shuffle(jm1_train)
            random.shuffle(jm1_test)
            jm1_test = jm1_test[:1000]

        Y_train = [j.pop(-1) for j in jm1_train]
        Y_train = [1 if x == "true" else 0 for x in Y_train]
        X_train = []
        for jt in jm1_train:
            X_train.append([float(j) for j in jt])

        Y_test = [j.pop(-1) for j in jm1_test]
        Y_test = [1 if x == "true" else 0 for x in Y_test]
        X_test = []
        for jt in jm1_test:
            X_test.append([float(j) for j in jt])

        import copy

        import pandas as pd
        from sklearn.linear_model import LogisticRegression

        def buildMatrix(listClassifiers):
            matrix = []
            for c1 in listClassifiers:
                line = []
                for c2 in listClassifiers:
                    count = 0
                    for i in range(len(c2)):
                        if c1[i] != c2[i]:
                            count += 1
                    line.append(count)
                matrix.append(copy.deepcopy(line))
            return matrix

        # classifList = []
        list = []

        c1 = LogisticRegression(random_state=0)
        c2 = LogisticRegression(C=0.5)
        c3 = LogisticRegression(C=0.1)
        c4 = LogisticRegression(C=0.05)

        c5 = LogisticRegression(solver="newton-cg", random_state=0)
        c6 = LogisticRegression(solver="newton-cg", C=0.5)
        c7 = LogisticRegression(solver="newton-cg", C=0.1)
        c8 = LogisticRegression(solver="newton-cg", C=0.05)
        c9 = LogisticRegression(
            penalty="none",
            solver="newton-cg",
            random_state=0,
        )

        c10 = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            random_state=0,
        )
        c11 = LogisticRegression(penalty="l2", solver="liblinear", C=0.5)
        c12 = LogisticRegression(penalty="l2", solver="liblinear", C=0.1)
        c13 = LogisticRegression(penalty="l2", solver="liblinear", C=0.05)

        c14 = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=0,
        )
        c15 = LogisticRegression(penalty="l1", solver="liblinear", C=0.5)
        c16 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1)
        c17 = LogisticRegression(penalty="l1", solver="liblinear", C=0.05)

        classifList = [
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
            c10,
            c11,
            c12,
            c13,
            c14,
            c15,
            c16,
            c17,
        ]
        listScore = []

        # for c in classif[1]:
        for c in classifList:
            c.fit(X_train, Y_train)
            list.append(copy.deepcopy(c.predict(X_test)))
            listScore.append(copy.deepcopy(c.score(X_test, Y_test)))

        matrix = buildMatrix(list)
        print(pd.DataFrame(matrix))
        print(listScore)

    @unittest.skip("Classifier")
    def test_matriz_diff_PassiveAgressive(self):

        ClassifiersClass = Classifiers()
        classif = ClassifiersClass.getPassiveAgressive()
        listScore = []

        X_test, X_train, Y_train, Y_test = self._createBase(100)
        list = []

        for c in classif[1]:
            c.fit(X_train, Y_train)
            list.append(copy.deepcopy(c.predict(X_test)))
            listScore.append(copy.deepcopy(c.score(X_test, Y_test)))

        matrix = self._buildMatrix(list)
        print(pd.DataFrame(matrix))
        print(listScore)
        pass

    @unittest.skip("Classifier")
    def test_matriz_diff_Ridge(self):

        ClassifiersClass = Classifiers()
        classif = ClassifiersClass.geRidget()

        X_test, X_train, Y_train, Y_test = self._createBase(200)
        list = []
        listScore = []

        for c in classif[1]:
            # for c in classifiers:
            c.fit(X_train, Y_train)
            list.append(copy.deepcopy(c.predict(X_test)))
            listScore.append(copy.deepcopy(c.score(X_test, Y_test)))

        matrix = self._buildMatrix(list)
        print(pd.DataFrame(matrix))
        print(listScore)
        pass

    @unittest.skip("Classifier")
    def test_matriz_diff_SGD(self):

        X_test, X_train, Y_train, Y_test = self._createBase(200)
        list = []
        listScore = []

        from sklearn.linear_model import SGDClassifier

        c0 = SGDClassifier(loss="hinge", penalty="elasticnet")
        c1 = SGDClassifier(loss="log")
        c2 = SGDClassifier(loss="modified_huber")
        c3 = SGDClassifier(loss="squared_hinge")
        c4 = SGDClassifier(loss="perceptron")
        c5 = SGDClassifier(loss="huber")
        c6 = SGDClassifier(loss="epsilon_insensitive")
        c7 = SGDClassifier(loss="squared_loss")

        classifiers = [c0, c1, c2, c3, c4, c5, c6, c7]

        # for c in classif[1]:
        for c in classifiers:
            c.fit(X_train, Y_train)
            list.append(copy.deepcopy(c.predict(X_test)))
            listScore.append(copy.deepcopy(c.score(X_test, Y_test)))

        matrix = self._buildMatrix(list)
        print(pd.DataFrame(matrix))
        print(listScore)
        pass

    @unittest.skip("Classifier")
    def test_matriz_diff_Adaboost(self):

        X_test, X_train, Y_train, Y_test = self._createBase(200)
        list = []
        listScore = []

        from sklearn.ensemble import AdaBoostClassifier

        c0 = AdaBoostClassifier(n_estimators=50)
        c1 = AdaBoostClassifier(n_estimators=500)
        c2 = AdaBoostClassifier(n_estimators=1000)
        c3 = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
        c4 = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
        c5 = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1)
        c6 = AdaBoostClassifier(n_estimators=50, learning_rate=0.5)
        c7 = AdaBoostClassifier(n_estimators=500, learning_rate=0.5)
        c8 = AdaBoostClassifier(n_estimators=1000, learning_rate=0.5)
        c9 = AdaBoostClassifier(n_estimators=50, random_state=5)
        c10 = AdaBoostClassifier(n_estimators=500, random_state=5)
        c11 = AdaBoostClassifier(n_estimators=1000, random_state=5)
        c12 = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=0.1,
            random_state=5,
        )
        c13 = AdaBoostClassifier(
            n_estimators=500,
            learning_rate=0.1,
            random_state=5,
        )
        c14 = AdaBoostClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            random_state=5,
        )
        c15 = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=0.5,
            random_state=5,
        )
        c16 = AdaBoostClassifier(
            n_estimators=500,
            learning_rate=0.5,
            random_state=5,
        )
        c17 = AdaBoostClassifier(
            n_estimators=1000,
            learning_rate=0.5,
            random_state=5,
        )

        classifiers = [c0, c1, c2, c3, c4, c5, c6, c7, c8]
        classifiers = classifiers + [
            c9,
            c10,
            c11,
            c12,
            c13,
            c14,
            c15,
            c16,
            c17,
        ]

        # for c in classif[1]:
        for c in classifiers:
            c.fit(X_train, Y_train)
            list.append(copy.deepcopy(c.predict(X_test)))
            listScore.append(copy.deepcopy(c.score(X_test, Y_test)))

        matrix = self._buildMatrix(list)
        print(pd.DataFrame(matrix))
        print(listScore)
        pass

    @unittest.skip("Classifier")
    def test_matriz_diff_Bagging(self):

        ClassifiersClass = Classifiers()
        classif = ClassifiersClass.getEnsembleBagging()

        X_test, X_train, Y_train, Y_test = self._createBase(200)
        list = []
        listScore = []

        from sklearn.ensemble import BaggingClassifier

        c0 = BaggingClassifier()
        c1 = BaggingClassifier(n_estimators=50)
        c2 = BaggingClassifier(n_estimators=250)
        c3 = BaggingClassifier(n_estimators=500)
        c4 = BaggingClassifier(max_samples=0.1)
        c5 = BaggingClassifier(n_estimators=50, max_samples=0.1)
        c6 = BaggingClassifier(n_estimators=250, max_samples=0.1)
        c7 = BaggingClassifier(n_estimators=500, max_samples=0.1)
        c8 = BaggingClassifier(bootstrap=False)
        c9 = BaggingClassifier(n_estimators=50, bootstrap=False)
        c10 = BaggingClassifier(n_estimators=250, bootstrap=False)
        c11 = BaggingClassifier(n_estimators=500, bootstrap=False)
        c12 = BaggingClassifier(max_samples=0.1, bootstrap=False)
        c13 = BaggingClassifier(
            n_estimators=50,
            max_samples=0.1,
            bootstrap=False,
        )
        c14 = BaggingClassifier(
            n_estimators=250,
            max_samples=0.1,
            bootstrap=False,
        )
        c15 = BaggingClassifier(
            n_estimators=500,
            max_samples=0.1,
            bootstrap=False,
        )
        c16 = BaggingClassifier(bootstrap_features=True)
        c17 = BaggingClassifier(n_estimators=50, bootstrap_features=True)
        c18 = BaggingClassifier(n_estimators=250, bootstrap_features=True)
        c19 = BaggingClassifier(n_estimators=500, bootstrap_features=True)
        c20 = BaggingClassifier(max_samples=0.1, bootstrap_features=True)
        c21 = BaggingClassifier(
            n_estimators=50, max_samples=0.1, bootstrap_features=True
        )
        c22 = BaggingClassifier(
            n_estimators=250, max_samples=0.1, bootstrap_features=True
        )
        c23 = BaggingClassifier(
            n_estimators=500, max_samples=0.1, bootstrap_features=True
        )
        c24 = BaggingClassifier(bootstrap=False, bootstrap_features=True)
        c25 = BaggingClassifier(
            n_estimators=50, bootstrap=False, bootstrap_features=True
        )
        c26 = BaggingClassifier(
            n_estimators=250, bootstrap=False, bootstrap_features=True
        )
        c27 = BaggingClassifier(
            n_estimators=500, bootstrap=False, bootstrap_features=True
        )
        c28 = BaggingClassifier(
            max_samples=0.1, bootstrap=False, bootstrap_features=True
        )
        c29 = BaggingClassifier(
            n_estimators=50,
            max_samples=0.1,
            bootstrap=False,
            bootstrap_features=True,
        )
        c30 = BaggingClassifier(
            n_estimators=250,
            max_samples=0.1,
            bootstrap=False,
            bootstrap_features=True,
        )
        c31 = BaggingClassifier(
            n_estimators=500,
            max_samples=0.1,
            bootstrap=False,
            bootstrap_features=True,
        )

        classifiers = [c0, c1, c2, c3, c4, c5, c6, c7]
        classifiers = classifiers + [c8, c9, c10, c11, c12, c13, c14, c15]
        classifiers = classifiers + [
            c16,
            c17,
            c18,
            c19,
            c20,
            c21,
            c22,
            c23,
            c24,
            c25,
            c26,
            c27,
            c28,
            c29,
            c30,
            c31,
        ]

        for c in classif[1]:
            # for c in classifiers:
            c.fit(X_train, Y_train)
            list.append(copy.deepcopy(c.predict(X_test)))
            listScore.append(copy.deepcopy(c.score(X_test, Y_test)))

        matrix = self._buildMatrix(list)
        print(pd.DataFrame(matrix))
        print(listScore)
        pass

    @unittest.skip("Classifier")
    def test_matriz_diff_RandomForest(self):

        # ClassifiersClass = Classifiers()
        # classif = ClassifiersClass.getEnsembleAdaboost()

        X_test, X_train, Y_train, Y_test = self._createBase(200)
        list = []
        listScore = []

        from sklearn.ensemble import RandomForestClassifier

        c0 = RandomForestClassifier()
        c1 = RandomForestClassifier(n_estimators=50)
        c2 = RandomForestClassifier(n_estimators=500)
        c3 = RandomForestClassifier(criterion="entropy")
        c4 = RandomForestClassifier(n_estimators=50, criterion="entropy")
        c5 = RandomForestClassifier(n_estimators=500, criterion="entropy")
        c6 = RandomForestClassifier(min_samples_split=5)
        c7 = RandomForestClassifier(n_estimators=50, min_samples_split=5)
        c8 = RandomForestClassifier(n_estimators=500, min_samples_split=5)
        c9 = RandomForestClassifier(criterion="entropy", min_samples_split=5)
        c10 = RandomForestClassifier(
            n_estimators=50, criterion="entropy", min_samples_split=5
        )
        c11 = RandomForestClassifier(
            n_estimators=500, criterion="entropy", min_samples_split=5
        )
        c12 = RandomForestClassifier(bootstrap=False)
        c13 = RandomForestClassifier(n_estimators=50, bootstrap=False)
        c14 = RandomForestClassifier(n_estimators=500, bootstrap=False)
        c15 = RandomForestClassifier(criterion="entropy", bootstrap=False)
        c16 = RandomForestClassifier(
            n_estimators=50, criterion="entropy", bootstrap=False
        )
        c17 = RandomForestClassifier(
            n_estimators=500, criterion="entropy", bootstrap=False
        )
        c18 = RandomForestClassifier(min_samples_split=5, bootstrap=False)
        c19 = RandomForestClassifier(
            n_estimators=50, min_samples_split=5, bootstrap=False
        )
        c20 = RandomForestClassifier(
            n_estimators=500, min_samples_split=5, bootstrap=False
        )
        c21 = RandomForestClassifier(
            criterion="entropy", min_samples_split=5, bootstrap=False
        )
        c22 = RandomForestClassifier(
            n_estimators=50,
            criterion="entropy",
            min_samples_split=5,
            bootstrap=False,
        )
        c23 = RandomForestClassifier(
            n_estimators=500,
            criterion="entropy",
            min_samples_split=5,
            bootstrap=False,
        )

        classifiers = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
        classifiers = classifiers + [
            c12,
            c13,
            c14,
            c15,
            c16,
            c17,
            c18,
            c19,
            c20,
            c21,
            c22,
            c23,
        ]
        # for c in classif[1]:
        for c in classifiers:
            c.fit(X_train, Y_train)
            list.append(copy.deepcopy(c.predict(X_test)))
            listScore.append(copy.deepcopy(c.score(X_test, Y_test)))

        matrix = self._buildMatrix(list)
        print(pd.DataFrame(matrix))
        print(listScore)
        pass

    def test_matriz_diff_BernoulliNB(self):

        X_test, X_train, Y_train, Y_test = self._createBase(200)
        list = []
        listScore = []

        from sklearn.naive_bayes import BernoulliNB

        c0 = BernoulliNB()
        c1 = BernoulliNB(binarize=10)
        c2 = BernoulliNB(alpha=5)
        c3 = BernoulliNB(alpha=0.05)
        # c4 = RidgeClassifier(fit_intercept=False,solver='svd')
        # c5 = RidgeClassifier(alpha=2.5,fit_intercept=False,solver='svd')
        # c6 = RidgeClassifier(alpha=5,fit_intercept=False,solver='svd')
        # c7 = RidgeClassifier(alpha=0.5,fit_intercept=False,solver='svd')
        # c8 = RidgeClassifier(solver='sparse_cg')
        # c9 = RidgeClassifier(solver='lsqr')
        # c10 = RidgeClassifier(solver='sag')

        classifiers = [c0, c1, c2, c3]
        # classifiers = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]

        # for c in classif[1]:
        for c in classifiers:
            c.fit(X_train, Y_train)
            list.append(copy.deepcopy(c.predict(X_test)))
            listScore.append(copy.deepcopy(c.score(X_test, Y_test)))

        matrix = self._buildMatrix(list)
        print(pd.DataFrame(matrix))
        print(listScore)
        pass

    def test_allEstimators(self):
        from sklearn.utils import all_estimators

        estimators = all_estimators(type_filter="classifier")
        with open("classifiers.txt", "w", newline="\n") as txtfile:
            txtfile.writelines([c[0] + "\n" for c in estimators])

    def test_allEstimators_proba(self):
        from sklearn.utils import all_estimators

        proba = []
        estimators = all_estimators(type_filter="classifier")
        with open("classifiers2.txt", "w", newline="\n") as txtfile:
            for name, class_ in estimators:
                if hasattr(class_, "predict_proba"):
                    proba.append(name)
            txtfile.writelines([c + "\n" for c in proba])

    def test_classif_proba(self):
        ClassifiersClass = Classifiers()
        names, classifiers = ClassifiersClass.getAll(ensembleFlag=True)

        temProba = []
        nTem = []
        for classif in classifiers:
            if hasattr(classif, "predict_proba"):
                temProba.append(classif)
            else:
                nTem.append(classif)


if __name__ == "__main__":
    unittest.main()
