import pandas as pd
import matplotlib.pyplot as plt
import random, cca, copy, pso

from classifiers import Classifiers
from paramsPSO import Params

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
X_test_cf = list(X_test[0:50])     #Sample test to train Celullar automata
Y_test_cf = list(Y_test[0:50])     #Sample test to train Celullar automata
X_test_ca = list(X_test[50:100])   #Sample test to validate Celullar automata
Y_test_ca = list(Y_test[50:100])   #Sample test to validate Celullar automata

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

############ PSO ###############
# params ###
qtdPop = 20
iteration = 50
coefAcceleration = 1
############
bestResult = {}
population = pso.initPopulation(qtdPop, matrix)
try:
    for i in range(iteration):
        for j in range(len(population)):
            print("matriz de inferencia para o individuo "+str(j)+" iteracao: "+str(i))
            answersListInference = cca.inferenceAlgorithm(population[j]['matrix'], nrCells, distance, population[j]['params'], rangeSampleCA, 100)
            scoreInference = cca.returnScore(Y_test_ca, answersListInference)
            pso.attPbest(population[j],j, scoreInference)
        gbest = pso.attGbest(population)
        bestResult = pso.attBestResult(gbest, bestResult)
        pso.attPosition(population, coefAcceleration, gbest)
finally: 
    print([{classif[c]['name']: classif[c]['score']} for c in classif])
    print("Maior score encontrado: " + str(max([classif[c]['score'] for c in classif])))
    print("Menor score encontrado: " + str(min([classif[c]['score'] for c in classif])))
    print(bestResult['params'])
    print(bestResult['score'])
    a = "a"

# try:
#     for i in range(iteration):
#         for j in range(len(population)):
#             print("treinando matriz para o individuo "+str(j)+" iteracao: "+str(i))
#             cca.algorithmCCA(population[j]['matrix'], Y_test_cf, nrCells, distance, poolClassif, classif, population[j]['params'], t, True)
#         pso.attPbest(population, rangeSampleCA, Y_test_ca)
#         gbest = pso.attGbest(population)
#         bestResult = pso.attBestResult(gbest, bestResult)
#         pso.attPosition(population, coefAcceleration, gbest)
# finally: 
#     print([{classif[c]['name']: classif[c]['score']} for c in classif])
#     print("Maior score encontrado: " + str(max([classif[c]['score'] for c in classif])))
#     print("Menor score encontrado: " + str(min([classif[c]['score'] for c in classif])))
#     print(bestResult['params'])
#     print(bestResult['score'])
#     a = "a"