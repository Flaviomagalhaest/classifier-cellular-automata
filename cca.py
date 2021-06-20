from numpy import true_divide
import matplotlib.pyplot as plt

#Return a line of matrix
def returnMatrixline(clf, keys, nrCells):
    returnList = []
    for x in range(nrCells):
        returnList.append(clf[keys.pop(0)])
    return returnList

#Return all neighbors of a determined cell
def returnNeighboringClassifiers(totalX, totalY, x, y, distance, neighbors):
    neighboringClassifiers = []
    for i in range(x-distance, x+distance+1):
        if i < 0: continue
        if i >= totalX: break
        for j in range(y-distance, y+distance+1):
            if j < 0: continue
            if j >= totalY: break
            if (i == x and j == y):
                continue
            else:
                neighboringClassifiers.append(neighbors[i][j])
    return neighboringClassifiers

#Return true if most neighbors got it right. false if wrong.
def neighborsMajorityRight(neighbors, sampleIndex, answer):
    sumRight = 0
    for i in range(len(neighbors)):
        sampleCell = neighbors[i]['predict'][sampleIndex]
        if sampleCell == answer:
            sumRight += 1
    if sumRight > (len(neighbors)/2):
        return True
    else: return False

def returnMatrixOfIndividualItem(matrix, item):
    return [[l[item] for l in m] for m in matrix]

#Print matrix of energies in console
def printMatrix(matrix):
    energyMatrix = returnMatrixOfIndividualItem(matrix, 'energy')
    fig, ax = plt.subplots()
    im = ax.imshow(energyMatrix)
    matrixSize = len(matrix[0])
    for i in range(matrixSize):
        for j in range(matrixSize):
            text = ax.text(j, i, energyMatrix[i][j],
                        ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()
a = "a"