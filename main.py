import pandas as pd
import matplotlib.pyplot as plt
import random, cca, copy, pso, csv

from classifiers import Classifiers
from params import Params
from graph import Graph
from dataGenerate import DataGenerate

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
###########################

# ####### TEST Sample ############
# X, Y = make_classification(n_samples=totalSamples, n_classes=2, n_features=100, n_redundant=0, random_state=1)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testSamples)

####### SAMPLE #################
with open('dataset/jm1.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    csvCount = 0
    jm1 = [row for nr, row in enumerate(spamreader)]
    random.shuffle(jm1)
    jm1_test = jm1[0:testSamples]
    jm1_train = jm1[testSamples:totalSamples]

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


divTest = int(testSamples/2)
rangeSampleCA  = range(divTest, testSamples)
X_test_cf = list(X_test[0:divTest])     #Sample test to train Celullar automata
Y_test_cf = list(Y_test[0:divTest])     #Sample test to train Celullar automata
X_test_ca = list(X_test[divTest:testSamples])   #Sample test to validate Celullar automata
Y_test_ca = list(Y_test[divTest:testSamples])   #Sample test to validate Celullar automata


####### CLASSIFIERS ############
ClassifiersClass = Classifiers()
names, classifiers = ClassifiersClass.getAll(ensembleFlag=True)

classif = {}
for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(X_train, Y_train)
    print("Clasificador "+name+" treinado.")
    c = {}
    c['name'] = name
    c['predict'] = clf.predict(X_test)
    # c['prob'] = clf.predict_proba(X_test)
    # c['confidence'] = clf.decision_function(X_test)
    # c['confAvg'], c['confAvgWhenWrong'], c['confAvgWhenRight'] =  cca.confidenceInClassification(c['predict'], Y_test, c['confidence'])
    # c['score'] = clf.score(X_test_ca, Y_test_ca)
    c['score'] = clf.score(X_test, Y_test)
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

###### plot config ######
# plt.ion()
# fig, ax = plt.subplots()
# cca.printMatrixInteractive(matrix, fig, ax)
# Graph(matrix)
# Graph.initMatrix(matrix)
DataGenerate(Y_test_ca, classif)
#########################

params['TRA'] = 2
params['TRB'] = 4
params['TRC'] = 0.05
params['TRD'] = 0.025
# params['TRC'] = 4
# params['TRD'] = 2

DataGenerate.saveStatus(matrix, classif)
cca.algorithmCCA(matrix, Y_test_cf, nrCells, distance, poolClassif, classif, params, t, True)
# Graph.printMatrixInteractiveEnergy(matrix, 'energy')
answersList = cca.weightedVote(matrix, rangeSampleCA)
score = cca.returnScore(Y_test_ca, answersList)
DataGenerate.file(score, answersList)
print([{classif[c]['name']: classif[c]['score']} for c in classif])
print("Maior score encontrado: " + str(max([classif[c]['score'] for c in classif])))
print("Menor score encontrado: " + str(min([classif[c]['score'] for c in classif])))
print(score)

answersList2 = cca.weightedVote2(matrix, rangeSampleCA)
score2 = cca.returnScore(Y_test_ca, answersList2)
print(score2)

# params['TRA'] = 0
# params['TRD'] = 0.6

# answersListInference = cca.inferenceAlgorithm(matrix, nrCells, distance, params, rangeSampleCA, 100)
# score3 = cca.returnScore(Y_test_ca, answersListInference)
# print(score3)