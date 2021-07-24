class Params:
   def get():
      obj = {}
      obj['energyInit'] = 100                     #Initial energy of cells
      obj['nrCells'] = 7                        #Size of matrix (ex. nrCells = 5 -> matrix 5x5)
      obj['t'] = 10                            #Number of iteractions
      obj['distance'] = 2                       #Euclidean distance of matrix 
      obj['sample'] = 0          
      obj['liveEnergy'] = 1                     #Value of energy that cell wil lost per iteration
      obj['cellRealocation'] = True             #Flag to realocate new classifier in dead cells
      obj['testSamples'] = 2000            #Size of samples used in testing
      obj['trainSamples'] = 500            #Size of samples used in testing
      return obj  