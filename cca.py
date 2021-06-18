#Return a line of matrix
from numpy import true_divide


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
def neighborsMajorityRigth(neighbors, sample, answer):
    sumRight = 0
    for i in range(len(neighbors)):
        sampleCell = neighbors[0]['predict'][sample]
        if sampleCell == answer:
            sumRight += 1
    if sumRight >= (len(neighbors)/2):
        return True
    else: return False
        
a = "a"