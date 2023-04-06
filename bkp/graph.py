import matplotlib.pyplot as plt
import cca, copy
import pandas as pd
import matplotlib.animation as animation

class Graph:
   #matrix
   figMatrix = ''
   axMatrix = ''
   imMatrix = {}
   matrixGraph=[]
   
   #bar
   figBar = ''
   axBar = ''
   bar = ''
   classif = {}
   textBar = []
   recBar = []


   def __init__(self):
      plt.ion()
   
   def initBar(data):
      Graph.figBar, Graph.axBar = plt.subplots()
      Graph.printBar(data)
      

   def printBar(data={}):
      if Graph.classif == {}:
         Graph.classif = data
         for i, d in enumerate(Graph.classif):
            Graph.textBar.append(Graph.axBar.text(data[d]['energy'], i, data[d]['energy'], ha='left'))
         names = list(data.keys())
         energies = [d['energy'] for d in data.values()]
         Graph.bar = Graph.axBar.barh(names, energies)
      else:
         for i, d in enumerate(Graph.classif):
            Graph.textBar[i] = Graph.classif[d]['energy']
            rect = Graph.bar[i]
            rect.set_width(Graph.classif[d]['energy'])
      Graph.figBar.canvas.flush_events()

   def initMatrix(matrix):
      Graph.figMatrix, Graph.axMatrix = plt.subplots()
      Graph.imMatrix = Graph.axMatrix.imshow(cca.returnMatrixOfIndividualItem(matrix, 'energy'))
      Graph.printMatrixInteractiveEnergy(matrix)
   
   #Print interactive matrix of energies
   def printMatrixInteractive(energyMatrix):
      Graph.imMatrix.set_data(energyMatrix)
      Graph.imMatrix.autoscale()
      matrixSize = len(energyMatrix[0])

      if Graph.matrixGraph == []:
         for i in range(matrixSize):
            matrixGraphLine = []
            for j in range(matrixSize):
                  text = Graph.axMatrix.text(j, i, energyMatrix[i][j], ha="center", va="center", color="w")
                  matrixGraphLine.append(text)
            Graph.matrixGraph.append(matrixGraphLine)
      else:
         for i in range(matrixSize):
            for j in range(matrixSize):
               Graph.matrixGraph[i][j].set_text(energyMatrix[i][j])

      Graph.figMatrix.tight_layout()
      Graph.figMatrix.canvas.flush_events()

   def printMatrixInteractiveEnergy(matrix):
      matrixEnergy = cca.returnMatrixOfIndividualItem(matrix, 'energy')
      Graph.printMatrixInteractive(matrixEnergy)

class BarChartRace():

   def __init__(self, matrixSize, sampleSize, iterationSize, energyList):
      self.it = 0
      self.sample = 0
      self.x = 0
      self.y = 0
      self.matrixSize = matrixSize
      self.sampleSize = sampleSize
      self.energyList = energyList
      self.iterationSize = iterationSize

   def draw(self):
      it = self.it
      sample = self.sample
      matrixSize = self.matrixSize
      sampleSize = self.sampleSize
      energyList = self.energyList
      iterationSize = self.iterationSize

      def barChartRace(frame):
         it = frame[0]
         sample = frame[1]
         # x = frame[2]
         # y = frame[3]
         
         dff = dfEnergy[
               dfEnergy['it'].eq(str(it))
            ][
               dfEnergy['sample'].eq(str(sample))
            ][
               dfEnergy['x'].eq(str(0))
            ][
               dfEnergy['y'].eq(str(0))]
         ax.clear()
         names = [c for c in dff.columns if c not in ['it', 'sample', 'x', 'y']]
         values = dff.values[0][4:]
         ax.barh(names, values)
         # updateParams()
         a = 'a'
         # for i, (value, name) in enumerate(zip(values, names)):
         #    ax.text(value, i,     name,           size=14, weight=600, ha='right', va='bottom')
         #    ax.text(value, i,     value,  size=14, ha='left',  va='center')
      
      # def frames():
      #    lista = []
      #    for i in range(0, iterationSize):
      #       for s in range(0, sampleSize):
      #          l = [i, s]
      #          lista.append(copy.deepcopy(l))
      #    return lista
      


      for iteracao in range (0, iterationSize):
         def frames():
            lista = []
            for i in range(iteracao, iteracao+1):
               for s in range(0, sampleSize):
                  l = [i, s]
                  lista.append(copy.deepcopy(l))
            return lista
         frames = frames()
         fig, ax = plt.subplots(figsize=(30, 15))
         columns = list(energyList[0].keys())
         dfEnergy = pd.DataFrame(energyList, columns=columns)
         animator = animation.FuncAnimation(fig, barChartRace, frames=frames)
         # plt.show()
         print('comecando a gravação da animação. iteracao '+str(iteracao))
         writergif = animation.PillowWriter(fps=15)
         animator.save('file/filename'+str(iteracao)+'.gif',writer=writergif)
         a = 'a'

      # animation.FuncAnimation(fig, draw_barchart, frames=range(1968, 2019))