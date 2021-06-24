import pandas as pd
import matplotlib.pyplot as plt
import random, cca

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

####### TEST MASS ############
X, Y = make_classification(n_samples=1000, n_classes=2, n_features=5, n_redundant=0, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

####### CLASSIFIERS ############
ClassifiersClass = Classifiers()
names, classifiers = ClassifiersClass.getAll(ensembleFlag=False)

####### PARAMS ############
params = Params.get()
energyInit      = params['energyInit']
nrCells         = params['nrCells']
t               = params['t']
distance        = params['distance']
sample          = params['sample']
liveEnergy      = params['liveEnergy']
cellRealocation = params['cellRealocation']
###########################

classif = {}
for name, clf in zip(names, classifiers):
    print("Treinando "+name)
    clf.fit(X_train, Y_train)
    print("Clasificador "+name+" treinado.")
    c = {}
    c['name'] = name
    c['score'] = clf.score(X_test, Y_test)
    c['predict'] = clf.predict(X_test)
    c['energy'] = energyInit
    classif[name] = c

#shuffling classifiers for the pool
poolClassif = list(classif.keys())
random.shuffle(poolClassif)

#building matrix of first celullar automata
matrix = [
    cca.returnMatrixline(classif, poolClassif, nrCells),
    cca.returnMatrixline(classif, poolClassif, nrCells),
    cca.returnMatrixline(classif, poolClassif, nrCells),
    cca.returnMatrixline(classif, poolClassif, nrCells),
    cca.returnMatrixline(classif, poolClassif, nrCells),
]
matrixOrigin = list(matrix)
print(cca.returnMatrixOfIndividualItem(matrix, 'score'))

#training iteration
for x in range (0, t):
    for sample in range(len(Y_test)):
        #get each cells of matrix
        for i in range(nrCells):
            for j in range(nrCells):
                neighbors = []
                #neighbors of current cell
                neighbors = cca.returnNeighboringClassifiers(nrCells, nrCells, i, j, distance, matrix)
                
                #return of classifier of neighbors. True if majority right.
                majorityNeighborsClassifier = cca.neighborsMajorityRight(neighbors, sample, Y_test[sample])
                
                #value of sample classified
                if 'predict' in matrix[i][j]:
                    cellSample = matrix[i][j]['predict'][sample]
                    currentEnergy = matrix[i][j]['energy']
                    if cellSample == Y_test[sample]:
                        #Classifier is right
                        if (majorityNeighborsClassifier):
                            matrix[i][j]['energy'] = currentEnergy + 2
                        else:
                            matrix[i][j]['energy'] = currentEnergy + 4
                    else:
                        #Classifier is wrong
                        if (majorityNeighborsClassifier):
                            matrix[i][j]['energy'] = currentEnergy - 4
                        else:
                            matrix[i][j]['energy'] = currentEnergy - 2
        cca.lostEnergyToLive(matrix, liveEnergy, poolClassif)
        cca.collectOrRelocateDeadCells(matrix, poolClassif, classif, cellRealocation)

        # if cellRealocation:
        a = "a"
    # print(cca.returnMatrixOfIndividualItem(matrix, 'energy'))
        # cca.printMatrix(matrix)
cca.printMatrix(matrix)
answersList = cca.weightedVote(matrix, len(Y_test))
score = cca.returnScore(Y_test, answersList)
print(score)
a = "a"
