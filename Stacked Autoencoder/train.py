import numpy as np
from load_data import myDataset
from model import Model, Noise
import matplotlib.pyplot as plt
import pdb
import argparse

def train(X, X_val, net_dims, epochs=200, learningRate=0.1,costEstimate="MSE"):

    batchSize=int(args.batchSize)
    n_in, n_h, n_fin = net_dims
    model=Model()
    noise = Noise()
    parameters = model.initialize_2layer_weights(n_in, n_h, n_fin)

    costs = []
    costs_ = []
    
    for ii in range(epochs):
        noBatches = int(X.shape[0]/batchSize)
        for jj in range(noBatches):

            XTrBatch, _ = myDataset.getTrMiniBatch()
            noisyXTrBatch = noise.GaussianNoise(XTrBatch, sd=0.3)

            
            W1,b1 = parameters["W1"],parameters["b1"]
            W2,b2 = parameters["W2"],parameters["b2"]

            A1,cache1= model.layer_forward(noisyXTrBatch, W1, b1, "relu")
            A2,cache2 = model.layer_forward(A1, W2, b2, "sigmoid")

            if costEstimate == "MSE":
                cost = model.MSE(A2, XTrBatch)
                dA= (A2-XTrBatch)/XTrBatch.shape[0]
            else:
                cost = model.crossEntropy(A2, XTrBatch)
                dA= ((A2 - X)/ (A2*(1.0 - A2)))/XTrBatch.shape[0]

            dA_prev2, dW2, db2 = model.layer_backward(dA, cache2, W2, b2, "sigmoid")
            dA_prev1, dW1, db1 = model.layer_backward(dA_prev2, cache1, W1, b1, "relu")

            parameters['W2']+=-learningRate*dW2
            parameters['b2']+=-learningRate*db2
            parameters['W1']+=-learningRate*dW1
            parameters['b1']+=-learningRate*db1


            if jj % 100 == 0:
                XValBatch , _ = myDataset.getValMiniBatch()
                noisyXValBatch = noise.GaussianNoise(XValBatch, sd=0.3)
                costs.append(cost)
                A1_,cache1_= model.layer_forward(noisyXValBatch, W1, b1, "relu")
                A2_,cache2_ = model.layer_forward(A1_, W2, b2, "sigmoid")
                if costEstimate == "MSE":
                    cost_ = model.MSE(A2_, XValBatch)
                else:
                    cost_ = model.crossEntropy(A2_, XValBatch)
                costs_.append(cost_)

            if jj % 100 == 0:
                print("Cost at iteration %i is: %f" %(jj, cost))
        

    return costs,costs_, parameters

def main(args):
    data = myDataset(args)
    train_data, val_data, test_data = data.getTrData(), data.getValData(), data.getTsData()

    m, n_in = train_data.shape
    n_fin = n_in
    n_h = 1000
    net_dims = [n_in, n_h, n_fin]

    args.learningRate = args.learningRate
    epochs = args.epochs
    plot_costs=[]
    plot_costs_=[]
    test_acc=[]
    net_dims = [n_in, n_h, n_fin]

    costs,costs_, parameters = train(train_data, val_data, net_dims, \
        epochs=epochs, learningRate=args.learningRate)
    plot_costs.append(costs)
    plot_costs_.append(costs_)
    
    
    it =  list(range(0,num_iterations,10))
    plt.plot(it, costs, label='train')
    plt.plot(it, costs_, label='val')
    plt.title('Train_cost and Val_cost vs iterations')
    plt.xlabel('iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='./data',
                        help="data path")
    parser.add_argument('--noTraining', default='60000',
                        help="number of training examples")
    parser.add_argument('--noValidation', default='10000',
                        help="number of validation examples")
    parser.add_argument('--noTesting', default='10000',
                        help="number of testing examples")
    parser.add_argument('--batchSize', default='64',
                        help="batch size")
    parser.add_argument('--learningRate', default='0.0001',
                        help="initial learning rate")
    # parser.add_argument('--cost', default='MSE',
    #                     help="MSE or crossEntropy")
    parser.add_argument('--labelRange', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="all the labels in the output")
    parser.add_argument('--noTrPerClass', default='6000',
                        help="number of training examples per class")
    parser.add_argument('--noValPerClass', default='1000',
                        help="number of validation examples per class")
    parser.add_argument('--noTsPerClass', default='1000',
                        help="number of testing example per class")
    parser.add_argument('--epochs', default=100,
                        help="number of epochs")
    parser.add_argument('--costEstimate', default='MSE',
                        help="Loss function")

    args = parser.parse_args()

    # args.id += '--batchSize' + str(args.batchSize)
    # args.id += '--learningRate' + str(args.learningRate)
    # args.id += '--noTraining' + str(args.noTraining)
    # args.id += '--noTesting' + str(args.noTesting)
    # args.id += '--cost' + str(args.cost)
    # args.id += '--dataPath' + str(args.dataPath)
    # args.id += '--noValidation' + str(args.noValidation)
    
    main(args)




