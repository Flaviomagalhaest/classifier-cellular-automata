import pandas as pd
import matplotlib.pyplot as plt
import random, cca, copy, pso

from classifiers import Classifiers
from params import Params

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

####### PARAMS ############
params = Params.get()
energyInit              = params['energyInit']
nrCells                 = params['nrCells']
t                       = params['t']
distance                = params['distance']
sample                  = params['sample']
liveEnergy              = params['liveEnergy']
cellRealocation         = params['cellRealocation']
totalSamples            = params['totalSamples']
testSamples              = params['testSamples']
rangeSampleCA           = params['rangeSampleCA']
###########################

####### TEST Sample ############
X, Y = make_classification(n_samples=totalSamples, n_classes=2, n_features=100, n_redundant=0, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testSamples)
X_test_cf = list(X_test[0:500])     #Sample test to train Celullar automata
Y_test_cf = list(Y_test[0:500])     #Sample test to train Celullar automata
X_test_ca = list(X_test[500:1000])   #Sample test to validate Celullar automata
Y_test_ca = list(Y_test[500:1000])   #Sample test to validate Celullar automata

####### CLASSIFIERS ############
ClassifiersClass = Classifiers()
names, classifiers = ClassifiersClass.getAll(ensembleFlag=True)

classif = {}
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    print("Clasificador "+name+" treinado.")
    c = {}
    c['name'] = name
    c['predict'] = clf.predict(X_test)
    c['prob'] = clf.predict_proba(X_test)
    # c['confidence'] = clf.decision_function(X_test)
    # c['confAvg'], c['confAvgWhenWrong'], c['confAvgWhenRight'] =  cca.confidenceInClassification(c['predict'], Y_test, c['confidence'])
    c['score'] = clf.score(X_test_ca, Y_test_ca)
    c['energy'] = energyInit
    classif[name] = c

#shuffling classifiers for the pool
poolClassif = list(classif.keys())
random.shuffle(poolClassif)

#building matrix of first celullar automata
matrix = []
for m in range(nrCells):
    matrix.append(cca.returnMatrixline(classif, poolClassif, nrCells))
matrixOrigin = copy.deepcopy(matrix)

params['TRA'] = 2
params['TRB'] = 4
params['TRC'] = 0.05
params['TRD'] = 0.025


cca.algorithmCCA(matrix, Y_test_cf, nrCells, distance, poolClassif, classif, params, t, True)
cca.printMatrix(matrix)
answersList = cca.weightedVote(matrix, rangeSampleCA)
score = cca.returnScore(Y_test_ca, answersList)
print([{classif[c]['name']: classif[c]['score']} for c in classif])
print("Maior score encontrado: " + str(max([classif[c]['score'] for c in classif])))
print("Menor score encontrado: " + str(min([classif[c]['score'] for c in classif])))
print(score)

answersList2 = cca.weightedVote2(matrix, rangeSampleCA)
score2 = cca.returnScore(Y_test_ca, answersList2)
print(score2)

params['TRA'] = 0
params['TRB'] = 0
params['TRC'] = 0.6
params['TRD'] = 0.8

answersListInference = cca.inferenceAlgorithm(matrix, nrCells, distance, params, rangeSampleCA, 100)
score3 = cca.returnScore(Y_test_ca, answersListInference)
print(score3)

# cca.restartEnergyMatrix(matrix, energyInit)
# cca.algorithmCCA(matrix, Y_test_cf, nrCells, distance, poolClassif, classif, params, t, False)
# cca.printMatrix(matrix)
# answersList = cca.weightedVote(matrix, rangeSampleCA)
# score4 = cca.returnScore(Y_test_ca, answersList)
# print(score4)

# ############ PSO ###############
# # params ###
# qtdPop = 10
# iteration = 50
# coefAcceleration = 1
# ############
# bestResult = {}
# population = pso.initPopulation(qtdPop, matrix)
# for i in range(iteration):
#     for j in range(len(population)):
#         print("treinando matriz para o individuo "+str(j)+" iteracao: "+str(i))
#         cca.algorithmCCA(population[j]['matrix'], Y_test_cf, nrCells, distance, poolClassif, classif, population[j]['params'], t, True)
#     pso.attPbest(population, rangeSampleCA, Y_test_ca)
#     gbest = pso.attGbest(population)
#     bestResult = pso.attBestResult(gbest, bestResult)
#     pso.attPosition(population, coefAcceleration, gbest)
# print([{classif[c]['name']: classif[c]['score']} for c in classif])
# print("Maior score encontrado: " + str(max([classif[c]['score'] for c in classif])))
# print("Menor score encontrado: " + str(min([classif[c]['score'] for c in classif])))
# print(bestResult['params'])
# print(bestResult['score'])
# a = "a"