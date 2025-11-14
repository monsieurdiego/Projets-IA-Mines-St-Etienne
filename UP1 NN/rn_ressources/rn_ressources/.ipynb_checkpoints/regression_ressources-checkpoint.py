# Regression

import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import warnings
import csv

"""
Computes all polynomials indices of given order in given dimension

Examples :

inds = polyIndices( ordermax, dimension )

inds = [-1,0,-1,0,1] -> 1*x0*1*x0*x1 = x0^2 x1
inds = [-1,-1,-1,3,3] -> 1*1*1*x3*x3 = x3^2

inds = polyIndices( FATordermax, FATdimension, 1e6 ) # If you get the warning (But good luck inverting the resulting matrix)
"""
def polyIndices( ordermax, dimension, safety=1e3 ):

   # First initialize with '[-1,-1,...,-1] -> 1 (order 0 monomial)'
   inds = -1 * np.ones( (1,ordermax), dtype=int )
   sze = 1
   
   for order in range(1,ordermax+1):
      addi = -1 * np.ones( (1,ordermax-order), dtype=int ) # This one will simply be added to ind0 to get the right size
      ind0 = np.zeros( (1,order), dtype=int ) # First of this order: all in first term x0^order.
      
      continueThis = True
      
      while continueThis: 
         ind1 = np.concatenate( (ind0,addi) , axis=1 )
         inds = np.concatenate( (inds,ind1) , axis=0 ) # Store the previously-computed indices
         
         # Now, we will increment on ind0 : [0,0,0,0] -> [0,0,0,1] -> ... -> [0,0,0,dimension-1] -> [0,0,1,1] -> ... -> [0,0,1,dimension-1] -> etc
         ind0[0,order-1] = ind0[0,order-1] + 1
         cursor = order-1  # We'll see if no index is superior to dimension-1. Start with the rightmost
         
         goLeft = (cursor > 0) # While this cursor has not reached the leftmost term, we continue to go left
         while goLeft:
            goLeft = False
            if ind0[0,cursor] > dimension-1: # Need to change decade Ex: [0,0,dimension] -> [0,1,1] Or: [0,dimension,dimension] -> [1,1,1]
               ind0[0,cursor-1] = ind0[0,cursor-1] + 1
               for i in range(cursor,order): ind0[0,i] = ind0[0,cursor-1]
               goLeft = (cursor > 1) # Extreme case cursor=1 requires to quit computing the monomial and is handeled above
               
            cursor = cursor-1 # Put the cursor one increment left
            
         if ind0[0,0] > dimension-1: # This whole monomial doesn't fit in the required dimension
            continueThis = False
            
         sze += 1
         
         if sze > safety: # Test on sze is plain safety. Please remember that if you want a really big polynomial
            warnings.warn("Trying to build a bigger polynomial than expected. If you expect indeed you polynomial to be huge, increase the third argument.")
            break
         
   return inds
   
"""
Computes all polynomials of specified dimension, which maximal order is specified

Examples: 

poly = polyPowers( ordermax, dimension )
poly = polyPowers( FATordermax, FATdimension, 1e6 ) # If you get the warning (But good luck actually using the resulting polynomial)

Output is of the shape:

poly = [[1,0,0] # First monomial : x0
        [0,1,0] # Second monomial : x1
        ...]
        
Each line gives a monomial. Examples of lines:
[5,3,6,8] -> x0^5 * x1^3 * x2^6 * x3^8
[0,0,2,1] -> x2^2 * x3^1
[0,0,0,0] -> 1
"""
def polyPowers( ordermax, dimension, safety=1e3 ):
   
   sze    = 1 # Stores the number of monomial in the polynomial
   powers = np.zeros( (0,dimension), dtype=int )
   
   powcur = np.zeros( (1,dimension), dtype=int ) # First monomial is [0,0,0,...,0] -> 1
   powers = np.concatenate( (powers,powcur) , axis=0 ) # Dont do simply powers=powcur bcause numpy would just egalize the pointers :(

   computeMore = True # Do we need to compute one monomial more?
   
   while computeMore:
      powcur[0,0] += 1 # increment first index
      
      # While the order is too high, 
      for i in range(dimension-1):
      
         if np.sum( powcur, axis=None ) <= ordermax:
            break # The monomial is admissible
            
         powcur[0,i] = 0
         powcur[0,i+1] += 1
         
      # If the momomial is admissible, add it, else, end the process
      if np.sum( powcur, axis=None ) <= ordermax:
         powers = np.concatenate( (powers,powcur) , axis=0 )
      else:
         computeMore = False
         
      sze += 1
         
      # Safety check
      if sze > safety: # Test on sze is plain safety. Please remember that if you want a really big polynomial
         warnings.warn("Trying to build a bigger polynomial than expected. If you expect indeed you polynomial to be huge, increase the third argument.")
         break
   
   return powers
   
"""
Read a single microstructure and adapts its format

Examples :

x = readMicrostruct("./hypercube/m25.csv") # Unix systems
x = readMicrostruct("hypercube\m25.csv")# Windows (to be confirmed)
"""
def readSingleMicrostruct( adress ):
   
   with open(adress, newline='') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      
      xt = np.zeros( (1,3) )   # Temporary row storing space
      x3 = np.zeros( (0,3) ) # This will store the microstructure under the triplet format
      
      i = 0
      for row in csvreader:
         j = 0
         for nb in row:
            xt[0,j] = float(nb)
            j += 1
            
         x3 = np.concatenate( (x3,xt), 0 )
         i  += 1

   x = x3.transpose().reshape((3*i,1)) # This will only work if the matrix has the right size
   return(x)
   
"""
Read a series of imput and output data
"""
def readData( adress ):
   xpath = adress + "VfRminRmaxepsi.csv"
   ypath = adress + "gamma.csv"
   
   with open(xpath, newline='') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      x  = np.zeros( (0,4) )
      xt = np.zeros( (1,4) )
      
      i = 0
      for row in csvreader:
         j = 0
         for nb in row:
            xt[0,j] = float(nb)
            j += 1
         x = np.concatenate( (x,xt), 0 )
         i += 1
         
   with open(ypath, newline='') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      yt = np.zeros( (1,1) )
      y  = np.zeros( (0,1) )
      
      i = 0
      for row in csvreader:
         j = 0
         for nb in row:
            yt[0,j] = float(nb)
            j += 1
         y = np.concatenate( (y,yt), 0 )
         i += 1
      
   return x, y
   
"""
Plot a microstructure
"""
def plotMicrostruct( x, nfig ):
   plt.figure(nfig)
   nrows = np.shape(x)[0]//3
   print(nrows)
   
   ax = plt.gca()
   for i in range(nrows):
      cir = plt.Circle( (x[i], x[i+nrows]), x[i+2*nrows], color='r' )
      ax.add_patch(cir)

   ax.set_aspect('equal')
   plt.xlim([0, 1.])
   plt.ylim([0, 1.])
   plt.show() # block=False
   
"""
Creates a random sample for cross-validation

Example :

Xv, Xd, Yv, Yd = crossValSplit( X, Y, .2 ) # 20% of the database ends up in Xc (rounded below), the validation samples and 80% of it ends up in Xd, from which the model will be built
"""
def crossValSplit( X, Y, qtt, setSeed=False ):

   nsamples = np.shape(X)[0]
   if np.shape(Y)[0] != nsamples: # check size of Y agrees
      sys.stderr.write("Not same number of input and output vectors in the database\n")
   
   ncross = int(np.floor(nsamples*qtt)) # Nb of samples used for cross validation
   
   mylist = list(range(nsamples))        # List 0,1,2...

   if setSeed: # Set random seed if requested (to make stuff deterministic)
      rnd.seed(0)
      
   crlist = rnd.sample(mylist,ncross)        # Extract random indices
   dalist = list(set(mylist) - set(crlist))  # List (set) substraction
   
   Xv = X[crlist,:]
   Xd = X[dalist,:]
   Yv = Y[crlist,:]
   Yd = Y[dalist,:]
   
   return Xv, Xd, Yv, Yd
  
"""
Computation on one line for the first parameter, and evaluation of the model

Examples:
modelPlotter( model, [True,True,True], x, y, 0 ) plots the model for top values (1)
=> shortcut: modelPlotter( model, True x, y, 0 )
modelPlotter( model, [False,False,False], x, y, 0 ) plots the model for bottom values (0)
=> shortcut: modelPlotter( model, False x, y, 0 )
modelPlotter( model, [False,True,False], x, y, 0 ) plots the model for different values
=> no shortcut

5th argument and above are directly sent to the model
"""
def modelPlotter( model, isTop, x, y, nfig, *args, **kwargs ):
   xmax = np.max( x, axis=0 ) # Find the max and min of each component
   xmin = np.min( x, axis=0 )
   tol = .0001 * ( xmax-xmin )

   npts = 50
   nsamples = np.shape(x)[0]
   x0range = np.linspace(xmin[0],xmax[0],npts)
   Xval = np.zeros( (npts, 4) )
   Xval[:,0] = x0range

   if not hasattr(isTop, '__iter__'): # Shortcut
      isTop = [isTop,isTop,isTop]

   # Attribute the values
   xval = xmin.tolist()
   for i in range(1,4):
      if isTop[i-1]:
         xval[i] = xmax[i]

   Xval[:,1] = xval[1] * np.ones((npts))
   Xval[:,2] = xval[2] * np.ones((npts))
   Xval[:,3] = xval[3] * np.ones((npts))

   ind1 = np.where( np.abs(x[:,1] - xval[1] * np.ones((nsamples))) <= tol[1] )
   ind2 = np.where( np.abs(x[:,2] - xval[2] * np.ones((nsamples))) <= tol[2] )
   ind3 = np.where( np.abs(x[:,3] - xval[3] * np.ones((nsamples))) <= tol[3] )
   
   ind = np.intersect1d( ind1, np.intersect1d(ind2,ind3) )
   
   Xcheck = x[ind,0]
   Ycheck = y[ind,:]

   Yval = model( Xval, *args, **kwargs )
   
   plt.figure(nfig)
   plt.plot(x0range,Yval,color='r',label='Model predictions')
   plt.plot(Xcheck,Ycheck,color='b',marker='*',linestyle='None',label='True values')
   plt.legend()
   plt.xlabel('Fiber Volume Fraction')
   plt.ylabel('Permeability (mm^2)')
   plt.title("Rmin = "+str(xval[1])+", Rmax = "+str(xval[2])+", epsi = "+str(xval[3]))
   plt.show()
  
   
if __name__ == "__main__" : # Draft zone
   print("use the `import` command to use those functions into your own script.")
   
