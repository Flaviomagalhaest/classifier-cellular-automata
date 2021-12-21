import pandas as pd
import matplotlib.pyplot as plt
import random, cca, copy, csv, sys

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
testSamples             = params['testSamples']
trainSamples            = params['trainSamples']
totalSamples            = testSamples + trainSamples
database = 'jm1'
###########################

def datasetSkLearn():
    ####### TEST Sample ############
    X, Y = make_classification(n_samples=totalSamples, n_classes=2, n_features=100, n_redundant=0, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testSamples)
    return X_train, X_test, Y_train, Y_test

def datasetJM1(multipleTrain=False):
    ####### SAMPLE #################
    with open('dataset/jm1.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        jm1 = [row for nr, row in enumerate(spamreader)]
        jm1_true = [j for j in jm1 if j[21] == 'true']
        jm1_false = [j for j in jm1 if j[21] == 'false']
        random.shuffle(jm1_true)
        random.shuffle(jm1_false)
        trainPart = int(trainSamples/2)            #Test sample divided between true answers and false answers
        jm1_train = jm1_true[:trainPart]
        jm1_train = jm1_train + jm1_false[:trainPart]
        jm1_test = jm1_true[trainPart:]
        jm1_test = jm1_test + jm1_false[trainPart:]
        random.shuffle(jm1_train)
        random.shuffle(jm1_test)
        jm1_test = jm1_test[:testSamples]
        # jm1_train = jm1[testSamples:totalSamples]

    Y_train = [j.pop(-1) for j in jm1_train]
    Y_train = [1 if x=='true' else 0 for x in Y_train]
    X_train = []
    for jt in jm1_train:
        X_train.append([float(j) for j in jt])

    if multipleTrain:
        div = 4
        dataDiv = int((totalSamples - testSamples) / div)
        Y_train = [Y_train[x:x+dataDiv] for x in range(0, len(jm1_train), dataDiv)]
        X_train = [X_train[x:x+dataDiv] for x in range(0, len(jm1_train), dataDiv)]

    Y_test = [j.pop(-1) for j in jm1_test]
    Y_test = [1 if x=='true' else 0 for x in Y_test]
    X_test = []
    for jt in jm1_test:
        X_test.append([float(j) for j in jt])
    return X_train, X_test, Y_train, Y_test

def dataset():
    if database == 'jm1':
        X_train, X_test, Y_train, Y_test = datasetJM1(False)
    else:
        X_train, X_test, Y_train, Y_test = datasetSkLearn()    

    divTest = int(testSamples/2)
    rangeSampleCA  = range(divTest, testSamples)
    X_test_cf = list(X_test[0:divTest])     #Sample test to train Celullar automata
    Y_test_cf = list(Y_test[0:divTest])     #Sample test to train Celullar automata
    X_test_ca = list(X_test[divTest:testSamples])   #Sample test to validate Celullar automata
    Y_test_ca = list(Y_test[divTest:testSamples])   #Sample test to validate Celullar automata
    return X_train, X_test, Y_train, Y_test, X_test_cf, Y_test_cf, X_test_ca, Y_test_ca, rangeSampleCA

def trainClassif(X_train, Y_train, X_test, Y_test):
    def fit(clf, X_train, Y_train):
        if isinstance(X_train[0][0], list):
            nr = random.randint(0, 3)
            clf.fit(X_train[nr], Y_train[nr])
        else:
            clf.fit(X_train, Y_train)

    ####### CLASSIFIERS ############
    ClassifiersClass = Classifiers()
    names, classifiers = ClassifiersClass.getAll(ensembleFlag=True)

    classif = {}
    for name, clf in zip(names, classifiers):
        try:
            fit(clf, X_train, Y_train)
            print("Clasificador "+name+" treinado.")
            c = {}
            c['name'] = name
            c['predict'] = clf.predict(X_test)
            # c['prob'] = clf.predict_proba(X_test)
            # c['confidence'] = clf.decision_function(X_test)
            # c['confAvg'], c['confAvgWhenWrong'], c['confAvgWhenRight'] =  cca.confidenceInClassification(c['predict'], Y_test, c['confidence'])
            # c['score'] = clf.score(X_test_ca, Y_test_ca)
            c['score'] = clf.score(X_test_ca, Y_test_ca)
            c['energy'] = energyInit
            classif[name] = c
        except:
            print("Unexpected error:", sys.exc_info()[0])
        finally:
            continue
    
    #shuffling classifiers for the pool
    poolClassif = list(classif.keys())
    random.shuffle(poolClassif)
    return poolClassif, classif

def buildMatrix(classif, poolClassif):
    #building matrix of first celullar automata
    matrix = []
    for m in range(nrCells):
        matrix.append(cca.returnMatrixline(classif, poolClassif, nrCells))
    return matrix

def buildPool(classif):
    massVote = []
    for i in range(testSamples):
        element = [classif[c]['predict'][0] for c in classif]
        massVote.append(max(set(element), key = element.count))
    for c in classif:
        count = 0
        for j in range(testSamples):
            if classif[c]['predict'][j] != massVote[j]:
                count += 1
        classif[c]['qVariety'] = count
    teste = pd.DataFrame(classif.values())
    a = 'a'

for repeat in range(30):
    X_train, X_test, Y_train, Y_test, X_test_cf, Y_test_cf, X_test_ca, Y_test_ca, rangeSampleCA = dataset()
    poolClassif, classif = trainClassif(X_train, Y_train, X_test, Y_test)
    
    matrix = buildMatrix(classif, poolClassif)
    # buildPool(classif)
    DataGenerate(Y_test_ca, classif)
    params['TRA'] = 2
    params['TRB'] = 4
    # params['TRC'] = 0.05
    # params['TRD'] = 0.025
    params['TRC'] = 0.025
    params['TRD'] = 0.012

    DataGenerate.saveStatus(matrix, classif)
    cca.algorithmCCA(matrix, Y_test_cf, nrCells, distance, poolClassif, classif, params, t, True)
    # Graph.printMatrixInteractiveEnergy(matrix, 'energy')
    answersList = cca.weightedVote(matrix, rangeSampleCA)
    score = cca.returnScore(Y_test_ca, answersList)
    DataGenerate.file(score, answersList)
    print("Maior score encontrado: " + str(max([classif[c]['score'] for c in classif])))
    print("Menor score encontrado: " + str(min([classif[c]['score'] for c in classif])))
    print(score)

    answersListInference = cca.inferenceAlgorithm(matrix, nrCells, distance, params, rangeSampleCA, t)
    score2 = cca.returnScore(Y_test_ca, answersListInference)
    print(score2)
    print('Salvando resultados da '+str(repeat)+' repeticao')
    DataGenerate.saveResult(score, score2, answersList, nrCells, matrix, t, distance, database)
DataGenerate.report()