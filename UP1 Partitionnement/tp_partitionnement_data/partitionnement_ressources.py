# Clustering

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import platform
import sys
import csv
         
"""
Generate sample vectors

Example :

centers = np.array([[0,5,-30],[0,5,0]])
sigmas  = np.array([[1,1,1]])
X = generateSamples2D( 100, centers, sigmas )
"""
def generateSamples2D( nsamples, centers, sigmas, setSeed=False ):
   Cshape = np.shape(centers)
   Sshape = np.shape(sigmas)
   
   if Cshape[0] != 2:
      sys.stderr.write("centers don't have 2 dofs\n")
      return
      
   if Sshape[0] != 1:
      sys.stderr.write("sigmas has too many rows\n")
      return
      
   if Cshape[1] != Sshape[1]:
      sys.stderr.write("there should be as many centers as sigmas\n")
      return
      
   nclust = Cshape[1]

   if setSeed: # Set random seed if requested (to make stuff deterministic)
      np.random.seed(0)
      rnd.seed(0)
   
   X = np.zeros((2,nsamples))
   
   for i in range(nsamples):
      clust = rnd.randrange(nclust) # Affect to a random cluster
      X[0,i] = np.random.normal( centers[0,clust] , sigmas[0,clust] )
      X[1,i] = np.random.normal( centers[1,clust] , sigmas[0,clust] )
      
   return X

"""
Compute mean and standard deviation of a set of vectors grouped in a numpy array
"""
def statVectors(X):
   Xshape = np.shape(X)
   ndofs  = Xshape[0]
   nvects = Xshape[1]
   
   Xmoy = np.sum( X, 1 ) / nvects # The sum
   Xvar = np.zeros( ndofs )
   for i in range(nvects):
      Xvar = Xvar + np.square(X[:,i]-Xmoy)
   Xvar = Xvar / (nvects-1)
   Xstd = np.sqrt(Xvar)
      
   return Xmoy, Xstd

"""
Initialization method for k-means: randomly pick one point for each cluster among database
"""
def pickInitialization( X, K, setSeed=False ):
   Xshape = np.shape(X)
   ndofs = Xshape[0]
   nvect = Xshape[1]
   
   if K > nvect:
      sys.stderr.write("requested number of clusters is too high for the database\n")
      return
      
   if setSeed: # Set random seed if requested (to make stuff deterministic)
      rnd.seed(0)
      
   avaliables = list(range(nvect)) # List of pickable indices
   b = np.zeros( (ndofs,K) ) # Set of barycenters
   
   for k in range(K):
      myInd = rnd.choice(avaliables)
      avaliables.remove(myInd)
      b[:,k] = X[:,myInd]
      
   return b
      
"""
Initialization method for k-means: randomly pick coordinates from the statistical properties of the database
"""
def statInitialization( X, K, setSeed=False ):
   Xshape = np.shape(X)
   ndofs = Xshape[0]
   nvect = Xshape[1]
   
   Xmoy, Xstd = statVectors(X) # Random initialization
   b = np.zeros( (ndofs,K) ) # Set of barycenters
   
   if setSeed: # Set random seed if requested (to make stuff deterministic)
      np.random.seed(0)

   for k in range(K):
      for i in range(ndofs):
         #b[i,k] = np.random.normal( Xmoy[i] , Xstd[i] ) # TODO: enable option here?
         b[i,k] = np.random.uniform( Xmoy[i]-Xstd[i] , Xmoy[i]+Xstd[i] )

   return b

"""
Initialization method for k-means: randomly affect a cluster to each data point, and compute their barycenters
"""
def pointInitialization( X, K, setSeed=False ):
   Xshape = np.shape(X)
   ndofs = Xshape[0]
   nvect = Xshape[1]
   
   if K > nvect:
      sys.stderr.write("requested number of clusters is too high for the database\n")
      return
      
   if setSeed: # Set random seed if requested (to make stuff deterministic)
      np.random.seed(0)
      
   ense1 = np.arange(K)
   np.random.shuffle(ense1) # This is to be sure any cluster has at least one vector
   ense2 = np.random.randint( K, size=nvect-K )
   ense = np.concatenate((ense1,ense2),axis=0)
   
   # Compute barycenter of each cluster
   b = np.zeros( (ndofs,K) )
   bsum = np.zeros( (ndofs,K) ) # This will store the sum
   card = np.zeros( K ) # This will store the number of vectors in a given subset
   for n in range(nvect): # Compute sums
      bsum[:,ense[n]] += X[:,n]
      card[ense[n]] += 1
   for k in range(K):
      b[:,k] = bsum[:,k]/card[k]
   
   return b

"""
Plot clusters in 2D
"""
def plot2DClusters( X, ense, b, nfig ):
   plt.figure(nfig)
   plt.scatter(X[0,:],X[1,:],c=ense) # Plot points
   plt.scatter(b[0,:],b[1,:],marker='*',c='grey') # Plot barycenters
   ax = plt.gca()
   ax.set_aspect('equal')
   plt.show() # block=False

"""
Read a single microstructure and adapts its format

Examples :

x = readMicrostruct( './microstruct_gaussian/circles_gaussian1.csv' ) # Unix systems
x = readMicrostruct( 'microstruct_gaussian\circles_gaussian1.csv' ) # Windows (to be confirmed)
"""
def readSingleMicrostruct( adress ):
   li = 0 # Current row
   xt = np.zeros( (1,3) )   # Temporary row storing space
   x3 = np.zeros( (100,3) ) # This will store the microstructure under the triplet format
   
   with open(adress, newline='') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      
      i = 0
      for row in csvreader:
         j = 0
         for nb in row:
            xt[0,j] = float(nb)
            j += 1
            
         if ( (xt[0,0] >= -.5) and (xt[0,0] <= .5) and (xt[0,1] >= -.5) and (xt[0,1] <= .5) ):
            x3[li,:] = xt  # Only account for those that are inside (periodicity)
            li += 1
            
         i  += 1

   x = x3.transpose().reshape((300,1)) # This will only work if the matrix has the right size
   return(x)

"""
Read all microstructures

Examples :

X = readMicrostructures( adress, ordered ) # ordered = True or False
X = readMicrostructures( '.', False ) # Read in the current directory (Unix systems)
X = readMicrostructures( '', False ) # Read in the current directory (Windows (to be confirmed))
"""
def readMicrostructures( adress, ordered ):

   # Build paths
   paths = []
   
   if platform.system() == 'Windows': # Look at me, I'm Bill Gates and I don't write paths the same way normal people do!
      if ordered:
         paths.append( adress + '\microstruct_sparse_ordered\circles_sparse' )
         paths.append( adress + '\microstruct_gaussian_ordered\circles_gaussian' )
         paths.append( adress + '\microstruct_heaps_ordered\circles_heaps' )
      else:
         paths.append( adress + '\microstruct_sparse\circles_sparse' )
         paths.append( adress + '\microstruct_gaussian\circles_gaussian' )
         paths.append( adress + '\microstruct_heaps\circles_heaps' )
   else:
      if ordered:
         paths.append( adress + '/microstruct_sparse_ordered/circles_sparse' )
         paths.append( adress + '/microstruct_gaussian_ordered/circles_gaussian' )
         paths.append( adress + '/microstruct_heaps_ordered/circles_heaps' )
      else:
         paths.append( adress + '/microstruct_sparse/circles_sparse' )
         paths.append( adress + '/microstruct_gaussian/circles_gaussian' )
         paths.append( adress + '/microstruct_heaps/circles_heaps' )

   X = np.zeros((300,0))

   for i in range(90): # TODO: adapt to the number of files in the folder
      for j in range(3):
         myadress = paths[j] + str(i+1) + '.csv'
         x = readSingleMicrostruct(myadress)
         X = np.concatenate( (X,x), axis=1 )

   return X
   
"""
Plot a microstructure
"""
def plotMicrostruct( x, nfig ):
   plt.figure(nfig)
   nrows = 100
   
   ax = plt.gca()
   for i in range(nrows):
      cir = plt.Circle( (x[i], x[i+100]), x[i+200], color='r' )
      ax.add_patch(cir)

   ax.set_aspect('equal')
   plt.xlim([-.5, .5])
   plt.ylim([-.5, .5])
   plt.show() # block=False
   
"""
Compute list of distances to closest neighbour
"""
def computeDistances(xyr):
   nrows = 100
   distances = np.zeros((100,1))
   
   x = xyr[0:100]     # Extract data
   y = xyr[100:200]
   r = xyr[200:300]
   
   # Generate periodicity stuff. TODO: code it less ugly
   xcat = x
   xcat = np.concatenate( (xcat, x-np.ones(100)) )
   xcat = np.concatenate( (xcat, x+np.ones(100)) )
   xcat = np.concatenate( (xcat, x-np.ones(100)) )
   xcat = np.concatenate( (xcat, x+np.ones(100)) )
   xcat = np.concatenate( (xcat, x) )
   xcat = np.concatenate( (xcat, x) )
   xcat = np.concatenate( (xcat, x-np.ones(100)) )
   xcat = np.concatenate( (xcat, x+np.ones(100)) )
   
   ycat = y
   ycat = np.concatenate( (ycat, y-np.ones(100)) )
   ycat = np.concatenate( (ycat, y-np.ones(100)) )
   ycat = np.concatenate( (ycat, y+np.ones(100)) )
   ycat = np.concatenate( (ycat, y+np.ones(100)) )
   ycat = np.concatenate( (ycat, y-np.ones(100)) )
   ycat = np.concatenate( (ycat, y+np.ones(100)) )
   ycat = np.concatenate( (ycat, y) )
   ycat = np.concatenate( (ycat, y) )
   
   rcat = r
   rcat = np.concatenate( (rcat, r) )
   rcat = np.concatenate( (rcat, r) )
   rcat = np.concatenate( (rcat, r) )
   rcat = np.concatenate( (rcat, r) )
   rcat = np.concatenate( (rcat, r) )
   rcat = np.concatenate( (rcat, r) )
   rcat = np.concatenate( (rcat, r) )
   rcat = np.concatenate( (rcat, r) )
   
   for i in range(nrows):
      dx = x[i]*np.ones(900) - xcat
      dy = y[i]*np.ones(900) - ycat
      ri = r[i]
      
      # All the distances to all other circles
      dist = np.sqrt( np.square(dx) + np.square(dy) ) - rcat - ri*np.ones(900)
      
      # At this point, the only negative values correspond to the cricle to itself (9 times because of periodicity).
      themax = np.max(dist)
      dist = np.where( dist<0, (themax+1)*np.ones(900), dist ) # /!\hacky shit: I replace negative values by a huge value.
      
      distances[i] = np.min(dist)
      
   return distances
      
   
if __name__ == "__main__" : # Draft zone
   print("use the `import` command to use those functions into your own script.")
   
