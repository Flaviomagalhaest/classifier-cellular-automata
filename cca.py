from numpy import true_divide
import matplotlib.pyplot as plt
import copy

#Return a line of matrix
def returnMatrixline(clf, keys, nrCells):
    returnList = []
    for x in range(nrCells):
        returnList.append(copy.deepcopy(clf[keys.pop(0)]))
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
                if neighbors[i][j] != {}:
                    neighboringClassifiers.append(neighbors[i][j])
    return neighboringClassifiers

#Return true if most neighbors got it right. false if wrong. false if even.
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

def neighborsMajorityEnergy(neighbors, sampleIndex):
    # energyWhoVoteZero = sum([c['energy'] for c in neighbors if 'predict' in c and c['predict'][sampleIndex] == 0])
    # energyWhoVoteOne = sum([c['energy'] for c in neighbors if 'predict' in c and c['predict'][sampleIndex] == 1])
    energyWhoVoteZero = sum([c['energy']*c['prob'][sampleIndex][0] for c in neighbors if 'predict' in c and c['predict'][sampleIndex] == 0])
    energyWhoVoteOne = sum([c['energy']*c['prob'][sampleIndex][1] for c in neighbors if 'predict' in c and c['predict'][sampleIndex] == 1])
    energyTotal = sum([c['energy'] for c in neighbors if 'predict' in c])
    return energyWhoVoteZero, energyWhoVoteOne, energyTotal

#Return average of neighbors's energy
def neighborsEnergyAverage(neighbors):
    energyTotal = sum([c['energy'] for c in neighbors if 'energy' in c])
    qtdNeighbors = len([c['energy'] for c in neighbors if 'energy' in c])

    if (energyTotal > 0) and (qtdNeighbors > 0):
        return energyTotal/qtdNeighbors
    else:
        return 0

#Return what classifies the majority neighbors choose.
def neighborsMajorityClassify(neighbors, sampleIndex):
    voteZero = ([c['predict'][sampleIndex] for c in neighbors]).count(0)
    voteOne = ([c['predict'][sampleIndex] for c in neighbors]).count(1)
    energyZero, energyOne, energyTotal = neighborsMajorityEnergy(neighbors, sampleIndex)
    if voteOne > voteZero:
        return 1
    elif voteOne < voteZero:
        return 0
    else:
        if energyZero >= energyOne:
            return 0
        else:
            return 1

#Method to subtract energy from live cell
def lostEnergyToLive(matrix, liveEnergy):
    matrixLength = len(matrix[0])
    for i in range(matrixLength):
        for j in range(matrixLength):
            if 'energy' in matrix[i][j]:
                matrix[i][j]['energy'] = round(matrix[i][j]['energy'] - liveEnergy, 2)


#Method to fill the empty spaces of matrix (from dead cells)
def collectOrRelocateDeadCells(matrix, pool=[], classifiers={}, cellRealocation=False, initEnergy=100):
    matrixLength = len(matrix[0])
    for i in range(matrixLength):
        for j in range(matrixLength):
            if ('energy' in matrix[i][j]) and (matrix[i][j]['energy'] <= 0):
                pool.append(matrix[i][j]['name'])
                if cellRealocation:
                    # print("Classifier "+matrix[i][j]['name']+" died. "+pool[0]+" took the place.")
                    matrix[i][j] = copy.deepcopy(classifiers[pool.pop(0)])
                    matrix[i][j]['energy'] = initEnergy
                else: 
                    # print("cell is dead.")
                    matrix[i][j] = {}

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

#Return list of answers of matrix using weighted vote for each samples
def weightedVote2(matrix, rangeSampleCA):
    answers = []
    matrixLength = len(matrix[0])
    for x in rangeSampleCA:
        voteOne = 0
        voteZero = 0
        OneEnergy = 0
        ZeroEnergy = 0
        for i in range(matrixLength):
            for j in range(matrixLength):
                if 'predict' in matrix[i][j]:
                    if matrix[i][j]['predict'][x] == 0:
                        voteZero += 1
                        ZeroEnergy += matrix[i][j]['energy']
                    if matrix[i][j]['predict'][x] == 1:
                        voteOne += 1
                        OneEnergy += matrix[i][j]['energy']
        if voteOne > voteZero:
            answers.append(1)
        elif voteOne < voteZero:
            answers.append(0)
        else: 
            if OneEnergy > ZeroEnergy:
                answers.append(1)
            elif OneEnergy < ZeroEnergy:
                answers.append(0)
            else: answers.append(2)
    return answers

def weightedVoteforSample(matrix, sample):
    matrixLength = len(matrix[0])
    energyWhoVoteZero = 0
    energyWhoVoteOne = 0
    for i in range(matrixLength):
        energyZero = 0
        energyOne = 0
        energyZero, energyOne, energyTotal = neighborsMajorityEnergy(matrix[i], sample)
        energyWhoVoteZero += energyZero
        energyWhoVoteOne += energyOne
    if energyWhoVoteZero > energyWhoVoteOne:
        return 0 
    elif energyWhoVoteZero < energyWhoVoteOne:
        return 1
    else: return 2

def weightedVoteforSample2(matrix, sample):
    matrixLength = len(matrix[0])
    energyWhoVoteZero = 0
    energyWhoVoteOne = 0
    for i in range(matrixLength):
        energyZero = 0
        energyOne = 0
        energyZero, energyOne, energyTotal = neighborsMajorityEnergy(matrix[i], sample)
        energyWhoVoteZero += energyZero
        energyWhoVoteOne += energyOne
    if energyWhoVoteZero > energyWhoVoteOne:
        return 0 
    elif energyWhoVoteZero < energyWhoVoteOne:
        return 1
    else: return 2


#Compare list of samples with the list generated
def returnScore(samples, generatedList):
    total = len(samples)
    hitSum = 0
    for i in range(total):
        if samples[i] == generatedList[i]:
            hitSum += 1
    return hitSum/total
        
def transactionRuleA(currentEnergy, averageNeighbors, x):
    return round(currentEnergy + x, 2)
    # return round(currentEnergy + (averageNeighbors * 0.001),1)

def transactionRuleB(currentEnergy, averageNeighbors, x):
    return round(currentEnergy + x, 2)
    # return round(currentEnergy + (averageNeighbors * 0.005),1)

def transactionRuleC(currentEnergy, averageNeighbors, x):
    # return round(currentEnergy + x, 2)
    return round(currentEnergy - (averageNeighbors * x),2)

def transactionRuleD(currentEnergy, averageNeighbors, x):
    # return round(currentEnergy + x, 2)
    return round(currentEnergy - (averageNeighbors * x),2)

def restartEnergyMatrix(matrix, energy=100):
    matrixLength = len(matrix[0])
    for i in range(matrixLength):
        for j in range(matrixLength):
            matrix[i][j]['energy'] = energy

def algorithmCCA(matrix, Y_test_cf, nrCells, distance, pool, classif, params, qtdIteration=10, learning=True):
    #training iteration
    for x in range (0, qtdIteration):
        for sample in range(len(Y_test_cf)):
            #get each cells of matrix
            for i in range(nrCells):
                for j in range(nrCells):
                    neighbors = []
                    #neighbors of current cell
                    neighbors = returnNeighboringClassifiers(nrCells, nrCells, i, j, distance, matrix)
                    
                    #return of classifier of neighbors. True if majority right.
                    majorityNeighborsClassifier = neighborsMajorityRight(neighbors, sample, Y_test_cf[sample])
                    averageNeighborsEnergy = neighborsEnergyAverage(neighbors)
                    
                    #value of sample classified
                    if 'predict' in matrix[i][j]:
                        cellSample = matrix[i][j]['predict'][sample]
                        currentEnergy = copy.deepcopy(matrix[i][j]['energy'])
                        if cellSample == Y_test_cf[sample]:
                            #Classifier is right
                            if (majorityNeighborsClassifier):
                                matrix[i][j]['energy'] = transactionRuleA(currentEnergy, averageNeighborsEnergy, params['TRA'])
                            else:
                                matrix[i][j]['energy'] = transactionRuleB(currentEnergy, averageNeighborsEnergy, params['TRB'])
                        else:
                            #Classifier is wrong
                            if (majorityNeighborsClassifier):
                                matrix[i][j]['energy'] = transactionRuleC(currentEnergy, averageNeighborsEnergy, params['TRC'])
                            else:
                                matrix[i][j]['energy'] = transactionRuleD(currentEnergy, averageNeighborsEnergy, params['TRD'])
                        a = 'a'
                    collectOrRelocateDeadCells(matrix, pool, classif, learning, averageNeighborsEnergy)
        if x == 9 or x == 99 or x==999:
            printMatrix(matrix)
        print("iteracao "+str(x))

def inferenceAlgorithm(matrix, nrCells, distance, params, rangeSampleCA, qtdIteration=100):
    classification = []
    # matrixSample = copy.deepcopy(matrix)
    for sample in rangeSampleCA:
        matrixSample = copy.deepcopy(matrix)
        for x in range(qtdIteration):
            for i in range(nrCells):
                for j in range(nrCells):
                    if 'predict' in matrixSample[i][j]:
                        neighbors = []
                        #neighbors of current cell
                        neighbors = returnNeighboringClassifiers(nrCells, nrCells, i, j, distance, matrixSample)
                        neighborsVote = neighborsMajorityClassify(neighbors, sample)
                        averageNeighborsEnergy = neighborsEnergyAverage(neighbors)
                        cellSample = matrixSample[i][j]['predict'][sample]
                        if cellSample == neighborsVote:
                            matrixSample[i][j]['energy'] = transactionRuleA(matrixSample[i][j]['energy'], averageNeighborsEnergy, params['TRA'])
                        else:
                            matrixSample[i][j]['energy'] = transactionRuleD(matrixSample[i][j]['energy'], averageNeighborsEnergy, params['TRD'])
                        if matrixSample[i][j]['energy'] <= 0:
                            matrixSample[i][j] = {}
                    # collectOrRelocateDeadCells(matrixSample)
            a = 'a'
        # print("sample "+str(sample))
        classification.append(weightedVoteforSample(matrixSample, sample))
        # printMatrix(matrixSample)
    return classification


#################### UTILS ##############################

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

def returnIndexOfDifferenceInLists(list1, list2):
    listIndex = []
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            listIndex.append(i)
    return listIndex

def returnIndexOfEqualsInLists(list1, list2):
    listIndex = []
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            listIndex.append(i)
    return listIndex

def confidenceInClassification(predict, answers, confidences):
    listEqual = returnIndexOfEqualsInLists(predict, answers)
    listDiff = returnIndexOfDifferenceInLists(predict, answers)

    confAvg = sum([abs(l) for l in confidences]) / len(confidences)
    confAvgWhenWrong = sum([abs(confidences[l]) for l in listDiff]) / len(listDiff)
    confAvgWhenRight = sum([abs(confidences[l]) for l in listEqual]) / len(listEqual)

    return confAvg, confAvgWhenWrong, confAvgWhenRight
