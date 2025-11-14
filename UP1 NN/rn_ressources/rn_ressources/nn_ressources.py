import regression_ressources as regs
import numpy as np
import matplotlib.pyplot as plt
import sys


"""
Backpropagation in a NN to get the gradient

/!\ This implementation is ludicrously sub-optimal, but its easier to understand. /!\
=> Refer to `BackPropagateNN_Block` for a more efficient implementation

x: imput as a multi-vector
y: output as a multi-vector
Wlist: list of matrices for affine steps
lambd: coefficient of sigmoid # /!\ Don't use "lambda" as avariable
alphalist: intermediary quantities
betalist: intermediary quantities
"""
def BackPropagateNN( x, y, Wlist, lambd, alphalist, betalist, lastLayerLinear=False ):
   sze = np.shape(x)[1] # Nb of vectors in the multi-vector x
   if (sze != np.shape(y)[1]):
      sys.stderr.write("Not same number of input and output vectors in the database\n")
      
   nlayers = len(Wlist)
      
   G = [ np.zeros(np.shape(Wlist[i])) for i in range(nlayers) ] # Initialize gradient
      
   for n in range(sze): # Loop on all points of the database
   
      beta_n = []
      alpha_n = []
      for i in range(nlayers):
         betaInter  = betalist[i][:,n] # Extract vector
         alphaInter = alphalist[i][:,n]
         beta_n.append(betaInter[:,None])  # betaInter[:,None] is a vodoo command to transform vector into row array
         alpha_n.append(alphaInter[:,None])
         
      x_n = x[:,n]
      y_n = y[:,n]
      
      G_n = BackPropagateNNSingle( x_n, y_n, Wlist, lambd, alpha_n, beta_n, lastLayerLinear )
      
      G = [ (G[i] + G_n[i]) for i in range(nlayers) ] # Increment gradient
      
   return G

"""
Backpropagation for a single vector
"""
def BackPropagateNNSingle( x, y, Wlist, lambd, alphalist, betalist, lastLayerLinear=False ):
   nlayers = len(Wlist)
   
   z = betalist[-1]
   delta = z-y
   
   G = []
   
   for s in range( nlayers-1, -1, -1 ):
      alpha = alphalist[s]
      
      if s == 0:
         beta = x[:,None] # \beta_0 (again with the vodoo trick that transforms vectors into arrays)
      else:
         beta = betalist[s-1]
         
      beta = np.concatenate( (beta, [[1]]), axis=0 ) # Add affine part
      W = Wlist[s]
      
      if lastLayerLinear and s == nlayers-1:
         gprim = np.ones(np.shape(alpha)) # Linear activation function
      else:
         elam = np.exp(-lambd*alpha)
         gprim = (lambd*elam)/np.square(1+elam)
         
      e = delta * gprim # Term-to-term product
      
      g = e @ beta.T # This is rank-one a matrix : one row times one line
      G.append(g)
      
      delta = W.T @ e
      delta = np.delete( delta, -1 , axis=0 ) # Remove last line: no retropropagation on the bias neuron

   G.reverse() # Because it was backwards
   
   return G

"""
Block Backpropagation in a NN to get the gradient. Efficient version

x: imput as a multi-vector
y: output as a multi-vector
Wlist: list of matrices for affine steps
lambd: coefficient of sigmoid # /!\ Don't use "lambda" as avariable
alphalist: intermediary quantities
betalist: intermediary quantities
"""
def BackPropagateNN_Block( x, y, Wlist, lambd, alphalist, betalist, lastLayerLinear=False ):
   nlayers = len(Wlist)
   
   z = betalist[-1]
   delta = z-y
   
   G = []
   ono = np.ones((1,np.shape(x)[1])) # A group of ones for affine function
   
   for s in range( nlayers-1, -1, -1 ):
      alpha = alphalist[s]
      
      if s == 0:
         beta = x # \beta_0
      else:
         beta = betalist[s-1]
         
      beta = np.concatenate( (beta, ono), axis=0 ) # Add affine part
      W = Wlist[s]
      
      if lastLayerLinear and s == nlayers-1:
         gprim = np.ones(np.shape(alpha)) # Linear activation function
      else:
         elam = np.exp(-lambd*alpha)
         gprim = (lambd*elam)/np.square(1+elam)
         
      e = delta * gprim
      g = e @ beta.T
      G.append(g)
      
      delta = W.T @ e
      delta = np.delete( delta, -1 , axis=0 ) # Remove last line: no retropropagation on the bias neuron

   G.reverse() # Because it was backwards
   
   return G

"""
Initialisation of a NN with uniform weights

sinput : shape of the input layer
myshape : list (or 1D array) of shapes of internal  and last layers
"""
def InitializeNN_ones( sinput, myshape ):
   nlayers = len(myshape)
   Wlist = []
   s1 = sinput # Layer input size
   for s in range(nlayers):
      s2 = myshape[s]                       # Layer output size
      W = 1/(s1+1) * np.ones( (s2,s1+1) )   # Output of each layer must be of magnitude 1
      Wlist.append(W)
      s1 = s2  # For next layer
      
   return Wlist
   
"""
Initialisation of a NN with random weights

sinput : shape of the input layer
myshape : list (or 1D array) of shapes of internal  and last layers
setSeed : optional parameter for debugging mainly
"""
def InitializeNN_rand( sinput, myshape, setSeed=False  ):

   if setSeed: # Set random seed if requested (to make stuff deterministic)
      np.random.seed(0)

   nlayers = len(myshape)
   Wlist = []
   s1 = sinput # Layer input size
   for s in range(nlayers):
      s2 = myshape[s] # Layer output size
      W = 1/(s1+1) * np.random.uniform( 0., 1., (s2,s1+1) )   # Output of each layer must be of magnitude 1
      Wlist.append(W)
      s1 = s2  # For next layer
      
   return Wlist
   
"""
Normalizes the data between 0 and 1 by an affine transform
"""
def normalize_data( x ):
   xmin = np.amin(x,0)
   xmax = np.amax(x,0)
      
   XminC = xmin[..., None]
   XmaxC = xmax[..., None]

   x = x - (np.ones((np.shape(x)[0],1)) @ XminC.T)
   x = x / (np.ones((np.shape(x)[0],1)) @ (XmaxC.T - XminC.T))
   
   return x, xmin, xmax
   

if __name__ == "__main__" : # Draft zone

   import tensorflow as tf

   #print(tf.__version__)

   x, y = regs.readData( "./hypercube/" )

   x, xmin, xmax = normalize_data(x)
   y, ymin, ymax = normalize_data(y)

   Xv, Xd, Yv, Yd = regs.crossValSplit( x, y, .2 )

   # BEGIN TENSORFLOW
   # Adapted from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb
   model = tf.keras.models.Sequential([
     tf.keras.layers.InputLayer(input_shape=(4, )), 
     tf.keras.layers.Dense(2, activation='sigmoid'), 
     tf.keras.layers.Dense(1, activation='sigmoid') 
     #tf.keras.layers.Dense(1)                     
   ])
   
   loss_fn = tf.keras.losses.MeanSquaredError() 
   
   model.compile(optimizer='adam',               # adam = a variant of stochastic gradient
              loss=loss_fn,
              metrics=['accuracy'])
              
   model.fit(Xd, Yd, epochs=5, verbose=0)
   
   model.evaluate(Xv, Yv)
   # END TENSORFLOW
   
   #modelPlotter( model, True, x, y, 0 )
   
   """
   Forward propagation function (non optimized).
   Dispatchs multi-vectors into single vectors and Calls in FwdPropagateNNSingle (to code)
   """
   def FwdPropagateNN( x, Wlist, lambd, lastLayerLinear=False ):

      sx = np.shape(x)
      sze = sx[1] # Nb of vectors in the multi-vector x

      alphalist = []
      betalist  = []

      nlayers = len(Wlist)
      shapes = [ sx[0] ]
      for s in range(nlayers):
         shapes.append( np.shape(Wlist[s])[0] )  # Retrieve sizes of intermediate vectors
         alphalist.append( np.zeros((shapes[-1],sze)) ) # Initialize zero arrays (to be filled later)
         betalist.append( np.zeros((shapes[-1],sze)) ) # Idem for beta

      z = np.zeros((shapes[-1],sze)) # Initialize Output multi-vector
      
      for n in range(sze): # Loop on all points of the database
         x_n = x[:,n]
         z_n, alphalist_n, betalist_n = FwdPropagateNNSingle( x_n, Wlist, lambd, lastLayerLinear )
         
         # Put them into the packed-up data
         z[:,n] = z_n
         for s in range(nlayers):
            alphalist[s][:,n] = alphalist_n[s]
            betalist[s][:,n]  = betalist_n[s]
         
      return z, alphalist, betalist
   
