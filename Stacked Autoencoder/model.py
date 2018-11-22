import numpy as np
import matplotlib.pyplot as plt
import pdb


class Model():

    def softmax_cross_entropy_loss(self, Z, Y=np.array([])):
        cache={} 
        n,m = Y.shape
        mask = range(m)
        A = np.exp(Z-np.max(Z))/(np.sum(np.exp(Z-np.max(Z)),axis=0)).reshape(1,m)  ## total is n,m
        cache["A"]=A
        loss = -np.log(A[Y.astype(int),mask])
        loss= np.sum(loss)/m 
        return A, cache, loss

    def softmax_cross_entropy_loss_der(self, Y, cache):
        n,m = Y.shape
        mask = range(m)
        dZ= cache["A"]
        dZ[Y.astype(int),mask]-=1
        dZ/=m
        return dZ

    def initialize_2layer_weights(self,n_in, n_h, n_fin):

        parameters = {}
        parameters["W1"] = (np.random.randn(n_in,n_h)*(np.sqrt(2.0/(n_in*n_h)))).T
        parameters["b1"] = np.zeros(n_h).reshape((n_h,1))
        parameters["W2"] = (np.random.randn(n_h,n_fin)*(np.sqrt(2.0/(n_h*n_fin)))).T
        parameters["b2"] = np.zeros(n_fin).reshape((n_fin,1))

        return parameters


    def initialize_multilayer_weights(self, net_dims):
        numLayers = len(net_dims)
        parameters = {}
        for l in range(numLayers-1):
            parameters["W"+str(l+1)] = np.random.randn(net_dims[l], net_dims[l+1])*(np.sqrt(2.0/(net_dims[l] * net_dims[l+1]))) #
            parameters["b"+str(l+1)] = np.zeros(net_dims[l+1]).reshape((net_dims[l+1],1))
        return parameters

    def tanh(self,Z):
        
        A = np.tanh(Z)
        cache = {}
        cache["Z"] = Z
        return A, cache

    def tanh_der(self,dA, cache):
        
        Z = cache["Z"]
        dZ = dA * (1-np.tanh(Z)**2)
        return dZ

    def sigmoid(self,Z):
        
        A = 1/(1+np.exp(-Z))
        cache = {}
        cache["Z"] = Z
        return A, cache

    def sigmoid_der(self,dA, cache):
        

        Z = cache["Z"]
        A_, cache_= sigmoid(Z)
        dZ= dA * A_ *(1-A_)

        return dZ

    def relu(self,Z):
        A = np.maximum(0,Z)
        cache = {}
        cache["Z"] = Z
        return A, cache

    def relu_der(self,dA, cache):
        dZ = np.array(dA, copy=True)
        Z = cache["Z"]
        dZ[Z<0] = 0
        return dZ

    def linear_forward(self,A, W, b):
        Z=W.dot(A)+b
        cache = {}
        cache["A"] = A
        return Z, cache

    def layer_forward(self,A_prev, W, b, activation):
        Z, lin_cache = linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, act_cache = sigmoid(Z)
        elif activation == "tanh":
            A, act_cache = tanh(Z)
        
        cache = {}
        cache["lin_cache"] = lin_cache
        cache["act_cache"] = act_cache

        return A, cache

    def crossEntropy(self,A2,X):

        obs = X.shape[0]
        pred = A2

        cost= -X*np.log(pred) - (1-X)*np.log(1-pred)
        cost= cost.sum()/obs

        return cost

    def MSE(self,A2,X):

        obs = X.shape[0]

        cost= (1/2)*(A2-X)*(A2-X)
        cost= cost.sum()/obs

        return cost

    def linear_backward(self,dZ, cache, W, b):
        A =cache["A"]
        dA_prev = np.dot(W.T,dZ)
        dW = np.dot(dZ,A.T)
        db = np.sum(dZ,axis=1,keepdims= True)
        return dA_prev, dW, db

    def layer_backward(self,dA, cache, W, b, activation):
        lin_cache = cache["lin_cache"]
        act_cache = cache["act_cache"]

        if activation == "sigmoid":
            dZ = sigmoid_der(dA, act_cache)
        elif activation == "tanh":
            dZ = tanh_der(dA, act_cache)
        dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
        return dA_prev, dW, db

class Noise():
    def SaltAndPepper(self, image, rate=0.3):
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
          
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
        
    def GaussianNoise(self, X, sd=0.5):
        X += numpy.random.normal(0, sd, X.shape)
        return X
        
    def MaskingNoise(self, X, rate=0.5):
        mask = (numpy.random.uniform(0,1, X.shape)<rate).astype("i4")
        X = mask*X
        return X