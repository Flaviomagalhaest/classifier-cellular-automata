from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

class Classifiers:
   def getDiscriminantAnalysis(self, names=[], classifiers=[]):
      names.append("QDA")
      names.append("LDA")
      classifiers.append(QuadraticDiscriminantAnalysis())
      classifiers.append(LinearDiscriminantAnalysis())
      return names, classifiers

   def getEnsemble(self, names=[], classifiers=[]):
      names.append("Random_Forest_12_100")
      names.append("Random_Forest_15_100")
      names.append("Random_Forest_5_300")
      names.append("Random_Forest_7_300")
      names.append("AdaBoost_50")
      names.append("AdaBoost_100")
      names.append("AdaBoost_150")
      names.append("Extra_Trees_10_2")
      names.append("Extra_Trees_30_2")
      names.append("Extra_Trees_10_4")
      names.append("Gradient_Boosting")
      classifiers.append(RandomForestClassifier(max_depth=12, n_estimators=100))
      classifiers.append(RandomForestClassifier(max_depth=15, n_estimators=100))
      classifiers.append(RandomForestClassifier(max_depth=5, n_estimators=300))
      classifiers.append(RandomForestClassifier(max_depth=7, n_estimators=300))
      classifiers.append(AdaBoostClassifier(n_estimators=50))
      classifiers.append(AdaBoostClassifier(n_estimators=100))
      classifiers.append(AdaBoostClassifier(n_estimators=150))
      classifiers.append(ExtraTreesClassifier(n_estimators=10, min_samples_split=2))
      classifiers.append(ExtraTreesClassifier(n_estimators=30, min_samples_split=2))
      classifiers.append(ExtraTreesClassifier(n_estimators=10, min_samples_split=4))
      classifiers.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0))
      return names, classifiers

   def getGaussian(self, names=[], classifiers=[]):
      names.append("Gaussian_Process")
      classifiers.append(GaussianProcessClassifier(1.0 * RBF(1.0)))
      return names, classifiers

   def getLinearModel(self, names=[], classifiers=[]):
      names.append("SGD_hinge")
      names.append("SGD_log")
      names.append("SGD_modified_huber")
      names.append("SGD_squared_hinge")
      names.append("SGD_perceptron")
      names.append("SGD_huber")
      names.append("SGD_epsilon_insensitive")
      names.append("SGD_squared_loss")
      names.append("Ridget")
      classifiers.append(SGDClassifier(loss="hinge", penalty="l2"))
      classifiers.append(SGDClassifier(loss="log"))
      classifiers.append(SGDClassifier(loss="modified_huber"))
      classifiers.append(SGDClassifier(loss="squared_hinge"))
      classifiers.append(SGDClassifier(loss="perceptron"))
      classifiers.append(SGDClassifier(loss='huber'))
      classifiers.append(SGDClassifier(loss='epsilon_insensitive'))
      classifiers.append(SGDClassifier(loss="squared_loss"))
      classifiers.append(RidgeClassifier())
      return names, classifiers

   def getNaiveBayers(self, names=[], classifiers=[]):
      names.append("Naive_Bayes")
      classifiers.append(GaussianNB())
      return names, classifiers

   def getNeuralNetwork(self, names=[], classifiers=[]):
      names.append("Neural_Net")
      classifiers.append(MLPClassifier(alpha=1, max_iter=1000))
      return names, classifiers

   def getNeighbors(self, names=[], classifiers=[]):
      names.append("Nearest_Neighbors_3")
      names.append("Nearest_Neighbors_4")
      names.append("Nearest_Neighbors_5")
      names.append("Nearest_Neighbors_7")
      classifiers.append(KNeighborsClassifier(3))
      classifiers.append(KNeighborsClassifier(4))
      classifiers.append(KNeighborsClassifier(5))
      classifiers.append(KNeighborsClassifier(7))
      return names, classifiers

   def getSVM(self, names=[], classifiers=[]):
      names.append("Linear_SVM")
      names.append("Polynomial_SVM")
      names.append("RBF_SVM")
      names.append("SIGMOID_SVM")
      names.append("OVO_SVM")
      names.append("Linear_NuSVC")
      names.append("Polynomial_NuSVC")
      names.append("RBF_NuSVC")
      names.append("SIGMOID_NuSVC")
      names.append("OVO_NuSVC")
      names.append("LinearSVC")
      names.append("LinearSVC_l2")

      classifiers.append(SVC(kernel="linear", C=0.025, probability=True))
      classifiers.append(SVC(kernel="poly", degree=3, C=0.025, probability=True))
      classifiers.append(SVC(kernel="rbf", C=1, gamma=2, probability=True))
      classifiers.append(SVC(kernel='sigmoid', probability=True))
      classifiers.append(SVC(decision_function_shape='ovo', probability=True))
      classifiers.append(NuSVC(kernel="linear", probability=True))
      classifiers.append(NuSVC(kernel="poly", degree=3, probability=True))
      classifiers.append(NuSVC(kernel="rbf",gamma=2, probability=True))
      classifiers.append(NuSVC(kernel='sigmoid', probability=True))
      classifiers.append(NuSVC(decision_function_shape='ovo', probability=True))
      # classifiers.append(LinearSVC())
      # classifiers.append(LinearSVC(penalty='l2', loss='hinge'))
      return names, classifiers
   
   def getTree(self, names=[], classifiers=[]):
      names.append("Decision_Tree_3")
      names.append("Decision_Tree_5")
      classifiers.append(DecisionTreeClassifier(max_depth=3))
      classifiers.append(DecisionTreeClassifier(max_depth=5))
      return names, classifiers

   def getAll(self, ensembleFlag=False):
      names = []
      classifiers = []

      names, classifiers = self.getDiscriminantAnalysis(names, classifiers)
      names, classifiers = self.getGaussian(names, classifiers)
      # names, classifiers = self.getLinearModel(names, classifiers)
      names, classifiers = self.getNaiveBayers(names, classifiers)
      names, classifiers = self.getNeuralNetwork(names, classifiers)
      names, classifiers = self.getNeighbors(names, classifiers)
      names, classifiers = self.getSVM(names, classifiers)
      names, classifiers = self.getTree(names, classifiers)

      if ensembleFlag: names, classifiers = self.getEnsemble(names, classifiers)
      return names, classifiers