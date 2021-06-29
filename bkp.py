import pandas as pd
import matplotlib.pyplot as plt
import random, cca, copy

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
energyInit      = params['energyInit']
nrCells         = params['nrCells']
t               = params['t']
distance        = params['distance']
sample          = params['sample']
liveEnergy      = params['liveEnergy']
cellRealocation = params['cellRealocation']
totalSamples    = params['totalSamples']
sampleSize      = params['sampleSize']
rangeSampleCA   = params['rangeSampleCA']
###########################

####### TEST Sample ############
X, Y = make_classification(n_samples=totalSamples, n_classes=2, n_features=5, n_redundant=0, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=sampleSize)
X_test_cf = list(X_test[0:400])     #Sample test to train Celullar automata
Y_test_cf = list(Y_test[0:400])     #Sample test to train Celullar automata
X_test_ca = list(X_test[400:800])   #Sample test to validate Celullar automata
Y_test_ca = list(Y_test[400:800])   #Sample test to validate Celullar automata

####### CLASSIFIERS ############
ClassifiersClass = Classifiers()
names, classifiers = ClassifiersClass.getAll(ensembleFlag=True)

classif = {}
for name, clf in zip(names, classifiers):
    print("Treinando "+name)
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

#training iteration
for x in range (0, t):
    for sample in range(len(Y_test_cf)):
        #get each cells of matrix
        for i in range(nrCells):
            for j in range(nrCells):
                neighbors = []
                #neighbors of current cell
                neighbors = cca.returnNeighboringClassifiers(nrCells, nrCells, i, j, distance, matrix)
                
                #return of classifier of neighbors. True if majority right.
                majorityNeighborsClassifier, averageNeighborsEnergy = cca.neighborsMajorityRight(neighbors, sample, Y_test_cf[sample])
                
                #value of sample classified
                if 'predict' in matrix[i][j]:
                    cellSample = matrix[i][j]['predict'][sample]
                    currentEnergy = copy.deepcopy(matrix[i][j]['energy'])
                    if cellSample == Y_test_cf[sample]:
                        #Classifier is right
                        if (majorityNeighborsClassifier):
                            matrix[i][j]['energy'] = cca.transactionRuleA(currentEnergy, averageNeighborsEnergy)
                        else:
                            matrix[i][j]['energy'] = cca.transactionRuleB(currentEnergy, averageNeighborsEnergy)
                    else:
                        #Classifier is wrong
                        if (majorityNeighborsClassifier):
                            matrix[i][j]['energy'] = cca.transactionRuleC(currentEnergy, averageNeighborsEnergy)
                        else:
                            matrix[i][j]['energy'] = cca.transactionRuleD(currentEnergy, averageNeighborsEnergy)
                    a = 'a'
                cca.collectOrRelocateDeadCells(matrix, poolClassif, classif, cellRealocation, averageNeighborsEnergy)
                a = 'a'
        # cca.lostEnergyToLive(matrix, liveEnergy)
        # cca.printMatrix(matrix)
        # cca.collectOrRelocateDeadCells(matrix, poolClassif, classif, cellRealocation, averageNeighborsEnergy)

    # cca.printMatrix(matrix)
    print(str(x) + ": "+str(cca.returnMatrixOfIndividualItem(matrix, 'energy')))
cca.printMatrix(matrix)
answersList = cca.weightedVote(matrix, rangeSampleCA)
score = cca.returnScore(Y_test_ca, answersList)
print([[{l['name']: l['score']} if 'energy' in l else 0 for l in m] for m in matrixOrigin])
print("Maior score encontrado: " + str(max([max([l['score'] for l in m]) for m in matrixOrigin])))
print("Menor score encontrado: " + str(min([min([l['score'] for l in m]) for m in matrixOrigin])))
print(score)
a = "a"