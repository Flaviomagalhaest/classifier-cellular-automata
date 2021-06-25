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
    totalNeighbors = len(neighbors)
    for i in range(len(neighbors)):
        if ('predict' in neighbors[i]):          
            sampleCell = neighbors[i]['predict'][sampleIndex]
            if sampleCell == answer:
                sumRight += 1
        else: totalNeighbors -= 1
    if sumRight > (totalNeighbors/2):
        return True
    else: return False

#Method to subtract energy from live cell
def lostEnergyToLive(matrix, liveEnergy, pool=[]):
    matrixLength = len(matrix[0])
    for i in range(matrixLength):
        for j in range(matrixLength):
            if 'energy' in matrix[i][j]:
                matrix[i][j]['energy'] = matrix[i][j]['energy'] - liveEnergy


#Method to fill the empty spaces of matrix (from dead cells)
def collectOrRelocateDeadCells(matrix, pool=[], classifiers={}, cellRealocation=False):
    matrixLength = len(matrix[0])
    for i in range(matrixLength):
        for j in range(matrixLength):
            if ('energy' in matrix[i][j]) and (matrix[i][j]['energy'] <= 0):
                print("Classifier "+matrix[i][j]['name']+" died. "+pool[0]+" took the place.")
                pool.append(matrix[i][j]['name'])
                if cellRealocation:
                    matrix[i][j] = classifiers[pool.pop(0)]
                else: matrix[i][j] = {}

#Return list of answers of matrix using weighted vote for each samples
def weightedVote(matrix, rangeSampleCA):
    answers = []
    matrixLength = len(matrix[0])
    for x in rangeSampleCA:
        voteOne = 0
        voteZero = 0
        for i in range(matrixLength):
            for j in range(matrixLength):
                if 'predict' in matrix[i][j]:
                    if matrix[i][j]['predict'][x] == 0:
                        voteZero += matrix[i][j]['energy']
                    if matrix[i][j]['predict'][x] == 1:
                        voteOne += matrix[i][j]['energy']
        if voteOne > voteZero:
            answers.append(1)
        elif voteOne < voteZero:
            answers.append(0)
        else: answers.append(2)
    return answers

#Compare list of samples with the list generated
def returnScore(samples, generatedList):
    total = len(samples)
    hitSum = 0
    for i in range(total):
        if samples[i] == generatedList[i]:
            hitSum += 1
    return hitSum/total
        


def returnMatrixOfIndividualItem(matrix, item):
    return [[l[item] if 'energy' in l else 0 for l in m] for m in matrix]

#Print matrix of energies in console
def printMatrix(matrix):
    energyMatrix = returnMatrixOfIndividualItem(matrix, 'energy')
    fig, ax = plt.subplots()
    im = ax.imshow(energyMatrix)
    matrixSize = len(matrix[0])
    for i in range(matrixSize):
        for j in range(matrixSize):
            text = ax.text(j, i, energyMatrix[i][j], ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()
a = "a"