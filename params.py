class Params:
   def get():
      obj = {}
      #Initial energy of cells
      obj['energyInit'] = 5
      #Size of matrix (ex. nrCells = 5 -> matrix 5x5)
      obj['nrCells'] = 5
      #Number of iteractions
      obj['t'] = 100
      #Euclidean distance of matrix 
      obj['distance'] = 1
      obj['sample'] = 0
      #Value of energy that cell wil lost per iteration
      obj['liveEnergy'] = 1
      #Flag to realocate new classifier in dead cells
      obj['cellRealocation'] = False
      
      return obj  