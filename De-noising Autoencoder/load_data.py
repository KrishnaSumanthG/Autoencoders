import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import argparse

class myDataset():
	def __init__(self, args):
		self.datapath = args.dataPath	
		self.noTraining = args.noTraining
		self.noValidation = args.noValidation
		self.noTesting = args.noTesting
		self.labelRange = args.labelRange
		self.noTrPerClass = args.noTrPerClass
		self.noValPerClass = args.noValPerClass
		self.noTsPerClass = args.noTsPerClass
		self.batchSize = args.batchSize


		fd = open(os.path.join(self.dataPath, 'train-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		print(loaded[:16])
		self.trData = loaded[16:].reshape((60000, 28*28)).astype(int)

		fd = open(os.path.join(self.dataPath, 'train-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		self.trLabels = loaded[8:].reshape((60000)).astype(float)

		fd = open(os.path.join(self.dataPath, 't10k-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		self.tsData = loaded[16:].reshape((10000, 28*28)).astype(int)

		fd = open(os.path.join(self.dataPath, 't10k-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		self.tsLabels = loaded[8:].reshape((10000)).astype(float)

		self.trData = self.trData/255.
		self.tsData = self.tsData/255.		

	    self.tsX = np.zeros((self.noTesting, 28*28))
    	self.trX = np.zeros((self.noTraining - self.noValidation, 28*28))
    	self.valX= np.zeros((noValSamples, 28*28))
    	self.tsY = np.zeros(self.noTesting)
    	self.trY = np.zeros(self.noTraining - self.noValidation)
    	self.valY = np.zeros(self.noValidation)


		count = 0

		for ll in labelRange:

		    idl = np.where(self.trLabels == ll)
		    
		    idl1 = idl[0][: (self.noTrPerClass-self.noValPerClass)]
		    idl2 = idl[0][(self.noTrPerClass-self.noValPerClass):self.noTrPerClass]
		    
		    idx1 = list(range(count*(self.noTrPerClass-self.noValPerClass), (count+1)*(self.noTrPerClass-self.noValPerClass)))
		    idx2 = list(range(count*(self.noValPerClass), (count+1)*self.noValPerClass))
		    
		    self.trX[idx1, :] = self.trData[idl1, :]
		    self.trY[idx1] = self.trLabels[idl1]
		    
		    # Val data

		    self.valX[idx2, :] = self.trData[idl2, :]
		    self.valY[idx2] = self.trLabels[idl2]

		    # Test data

		    idl = np.where(self.tsLabels == ll)
			idl = idl[0][: self.noTsPerClass]
			idx = list(range(count*self.noTsPerClass, (count+1)*self.noTsPerClass))
			self.tsX[idx, :] = self.tsData[idl, :]
			self.tsY[idx] = self.tsLabels[idl]
			count += 1

		# np.random.seed(1)
		test_idx = np.random.permutation(tsX.shape[0])
		self.tsX = self.tsX[test_idx,:]
		self.tsY = self.tsY[test_idx]

		self.trY = self.trY.reshape(1, -1)
		self.valY = self.valY.reshape(1, -1)
		self.tsY = self.tsY.reshape(1, -1)




	def getTrMiniBatch(self):
		idx = list(range(self.trX.shape[0]))
		minibatch = idx[:self.batchSize]
		return trX[minibatch, :], trY[minibatch, :]

	def getValMiniBatch(self):
		idx = list(range(self.valX.shape[0]))
		minibatch = idx[:self.batchSize]
		return valX[minibatch, :], valY[minibatch, :]

	def getTsMiniBatch(self):
		idx = list(range(self.tsX.shape[0]))
		minibatch = idx[:self.batchSize]
		return tsX[minibatch, :], tsY[minibatch, :]




