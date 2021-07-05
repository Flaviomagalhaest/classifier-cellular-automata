import matplotlib.pyplot as plt
import cca, copy

class Graph:
   fig = ''
   ax = ''
   matrixGraph=[]
   im = {}
   
   def __init__(self, matrix):
      plt.ion()
      Graph.fig, Graph.ax = plt.subplots()
      Graph.im = Graph.ax.imshow(cca.returnMatrixOfIndividualItem(matrix, 'energy'))
   
   #Print interactive matrix of energies
   def printMatrixInteractive(energyMatrix):
      Graph.im.set_data(energyMatrix)
      Graph.im.autoscale()
      matrixSize = len(energyMatrix[0])

      if Graph.matrixGraph == []:
         for i in range(matrixSize):
            matrixGraphLine = []
            for j in range(matrixSize):
                  text = Graph.ax.text(j, i, energyMatrix[i][j], ha="center", va="center", color="w")
                  matrixGraphLine.append(text)
            Graph.matrixGraph.append(matrixGraphLine)
      else:
         for i in range(matrixSize):
            for j in range(matrixSize):
               Graph.matrixGraph[i][j].set_text(energyMatrix[i][j])

      Graph.fig.tight_layout()
      Graph.fig.canvas.flush_events()

   def printMatrixInteractiveEnergy(matrix):
      matrixEnergy = cca.returnMatrixOfIndividualItem(matrix, 'energy')
      Graph.printMatrixInteractive(matrixEnergy)