import pandas as pd
import random, cca

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

X, Y = make_classification(n_samples=1000, n_classes=2, n_features=5, n_redundant=0, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

names = ["Nearest_Neighbors_3", "Nearest_Neighbors_4", "Nearest_Neighbors_5", "Nearest_Neighbors_7", "Linear_SVM", "Polynomial_SVM", "RBF_SVM", "Gaussian_Process",
         "Gradient_Boosting", "Decision_Tree_3", "Decision_Tree_5", "Extra_Trees_10_2", "Extra_Trees_30_2", "Extra_Trees_10_4", "Random_Forest_12_100", "Random_Forest_15_100", "Random_Forest_5_300",
         "Random_Forest_7_300", "Neural_Net", "AdaBoost_50", "AdaBoost_100", "AdaBoost_150",
         "Naive_Bayes", "QDA", "SGD"]

classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(4),
    KNeighborsClassifier(5),
    KNeighborsClassifier(7),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
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
    SGDClassifier(loss="hinge", penalty="l2")]

classif = {}

for name, clf in zip(names, classifiers):
    print("Treinando "+name)
    clf.fit(X_train, Y_train)
    print("Clasificador "+name+" treinado.")
    c = {}
    c['name'] = name
    c['score'] = clf.score(X_test, Y_test)
    c['predict'] = clf.predict(X_test)
    classif[name] = c

#shuffling classifiers for the pool
keysClassif = list(classif.keys())
random.shuffle(keysClassif)

#building matrix of first celullar automata
nrCells = 5
matrix = [
    cca.returnMatrixline(classif, keysClassif, nrCells),
    cca.returnMatrixline(classif, keysClassif, nrCells),
    cca.returnMatrixline(classif, keysClassif, nrCells),
    cca.returnMatrixline(classif, keysClassif, nrCells),
    cca.returnMatrixline(classif, keysClassif, nrCells),
]


#training iteration
t = 1000
# for i in range(t):
distance = 1
sample = 0
#get each cells of matrix
for i in range(nrCells):
   for j in range(nrCells):
      neighbors = []
      #neighbors of current cell
      neighbors = cca.returnNeighboringClassifiers(nrCells, nrCells, i, j, distance, matrix)
      #value of sample classified
      cellSample = matrix[i][j]['predict'][sample]
      
      
      
      if cellSample == Y_test[sample]:
         #algoritmo caso certo
         b = 1
      else:
         #algoritmo caso errado
         b = 1
      a = "a"

a = "a"