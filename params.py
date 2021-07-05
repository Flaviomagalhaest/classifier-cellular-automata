class Params:
   def get():
      obj = {}
      obj['energyInit'] = 100                     #Initial energy of cells
      obj['nrCells'] = 5                        #Size of matrix (ex. nrCells = 5 -> matrix 5x5)
      obj['t'] = 100                            #Number of iteractions
      obj['distance'] = 1                       #Euclidean distance of matrix 
      obj['sample'] = 0          
      obj['liveEnergy'] = 1                     #Value of energy that cell wil lost per iteration
      obj['cellRealocation'] = True             #Flag to realocate new classifier in dead cells
      obj['totalSamples'] = 5000                #Total of samples used in training e testing
      obj['testSamples'] = 4500            #Size of samples used in testing
      obj['rangeSampleCA'] = range(500,1000)     #Range of test samples used to test CCA
      return obj  