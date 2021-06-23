import pandas as pd
import matplotlib.pyplot as plt
import random, cca

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

X, Y = make_classification(n_samples=1000, n_classes=2, n_features=5, n_redundant=0, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

names = ["Nearest_Neighbors_3", "Nearest_Neighbors_4", "Nearest_Neighbors_5", "Nearest_Neighbors_7", "Linear_SVM", "Polynomial_SVM",
         "RBF_SVM", "SIGMOID_SVM", "OVO_SVM", "Gaussian_Process","Gradient_Boosting", "Decision_Tree_3", "Decision_Tree_5", "Extra_Trees_10_2",
         "Extra_Trees_30_2", "Extra_Trees_10_4", "Random_Forest_12_100", "Random_Forest_15_100", "Random_Forest_5_300",
         "Random_Forest_7_300", "Neural_Net", "AdaBoost_50", "AdaBoost_100", "AdaBoost_150", "Naive_Bayes", "QDA", "SGD_hinge", "SGD_log",
         "SGD_modified_huber","SGD_squared_hinge", "SGD_perceptron", "SGD_squared_loss", "SGD_huber", "SGD_epsilon_insensitive", "LDA", "Ridget",
         "Linear_NuSVC", "Polynomial_NuSVC", "RBF_NuSVC", "SIGMOID_NuSVC", "OVO_NuSVC", "LinearSVC", "LinearSVC_l2"]

classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(4),
    KNeighborsClassifier(5),
    KNeighborsClassifier(7),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
    SVC(kernel='sigmoid'),
    SVC(decision_function_shape='ovo'),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    DecisionTreeClassifier(max_depth=3),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
    ExtraTreesClassifier(n_estimators=30, min_samples_split=2),
    ExtraTreesClassifier(n_estimators=10, min_samples_split=4),
    RandomForestClassifier(max_depth=12, n_estimators=100),
    RandomForestClassifier(max_depth=15, n_estimators=100),
    RandomForestClassifier(max_depth=5, n_estimators=300),
    RandomForestClassifier(max_depth=7, n_estimators=300),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=50),
    AdaBoostClassifier(n_estimators=100),
    AdaBoostClassifier(n_estimators=150),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="hinge", penalty="l2"),
    SGDClassifier(loss="log"),
    SGDClassifier(loss="modified_huber"),
    SGDClassifier(loss="squared_hinge"),
    SGDClassifier(loss="perceptron"),
    SGDClassifier(loss="squared_loss"),
    SGDClassifier(loss='huber'),
    SGDClassifier(loss='epsilon_insensitive'),
    LinearDiscriminantAnalysis(),
    RidgeClassifier(),
    NuSVC(kernel="linear"),
    NuSVC(kernel="poly", degree=3),
    NuSVC(kernel="rbf",gamma=2),
    NuSVC(kernel='sigmoid'),
    NuSVC(decision_function_shape='ovo'),
    LinearSVC(),
    LinearSVC(penalty='l2', loss='hinge')
    ]

####### PARAMS ############
energyInit = 5              #Initial energy of cells
nrCells = 5                 #Size of matrix (ex. nrCells = 5 -> matrix 5x5)
t = 100                      #Number of iteractions
distance = 1                #Euclidean distance of matrix 
sample = 0
liveEnergy = 1              #Value of energy that cell wil lost per iteration
cellRealocation = False     #Flag to realocate new classifier in dead cells
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
