import numpy as np
from load_data import myDataset
from model import Model, Noise
import matplotlib.pyplot as plt
import pdb
import argparse

def train(X, X_val, net_dims, epochs=2000, learningRate=0.1,costEstimate="MSE", decayRate = 0.5):
    learningRate=float(args.learningRate)
    epochs=int(args.epochs)
    decayRate = float(args.decayRate)
    batchSize=int(args.batchSize)
    n_in, n_h, n_fin = net_dims
    data=myDataset(args)
    model=Model()
    noise = Noise()
    parameters = model.initialize_2layer_weights(n_in, n_h, n_fin)

    costs = []
    costs_ = []
    
    for ii in range(epochs):
        noBatches = int(X.shape[0]/batchSize)
        learningRate = learningRate*(1/(1+decayRate*ii))
        print(learningRate)
        print("Epoch: ",ii )
        for jj in range(noBatches):

            XTrBatch= data.getTrMiniBatch()
            noisyXTrBatch = noise.GaussianNoise(XTrBatch, sd=0.3)
            
            W1,b1 = parameters["W1"],parameters["b1"]
            W2,b2 = parameters["W2"],parameters["b2"]

            A1,cache1= model.layer_forward(noisyXTrBatch, W1, b1, "relu")
            A2,cache2 = model.layer_forward(A1, W2, b2, "sigmoid")

            if costEstimate == "MSE":
                cost = model.MSE(A2, XTrBatch)
                dA= (A2-XTrBatch)/(XTrBatch.shape[1])
            else:
                cost = model.crossEntropy(A2, XTrBatch)
                dA= ((A2 - X)/ (A2*(1.0 - A2)))/XTrBatch.shape[1]

            dA_prev2, dW2, db2 = model.layer_backward(dA, cache2, W2, b2, "sigmoid")
            dA_prev1, dW1, db1 = model.layer_backward(dA_prev2, cache1, W1, b1, "relu")

            parameters['W2']+=-learningRate*dW2
            parameters['b2']+=-learningRate*db2
            parameters['W1']+=-learningRate*dW1
            parameters['b1']+=-learningRate*db1

            if jj % 50 == 0:
                print("Cost at iteration %i is: %f" %(jj*50, cost))
        
        XValBatch = data.getValMiniBatch()
        noisyXValBatch = noise.GaussianNoise(XValBatch, sd=0.3)
        costs.append(cost)
        A1_,cache1_= model.layer_forward(noisyXValBatch, W1, b1, "relu")
        A2_,cache2_ = model.layer_forward(A1_, W2, b2, "sigmoid")
        if costEstimate == "MSE":
            cost_ = model.MSE(A2_, XValBatch)
        else:
            cost_ = model.crossEntropy(A2_, XValBatch)
        costs_.append(cost_)

    return costs,costs_, parameters

def predict(parameters):
    data=myDataset(args)
    model=Model()
    noise = Noise()
    XTsBatch= data.getTsMiniBatch()
    noisyXTsBatch = noise.GaussianNoise(XTsBatch, sd=0.3)
    #print(noisyXTrBatch.shape)

    W1,b1 = parameters["W1"],parameters["b1"]
    W2,b2 = parameters["W2"],parameters["b2"]

    A1,cache1= model.layer_forward(noisyXTsBatch, W1, b1, "relu")
    A2,cache2 = model.layer_forward(A1, W2, b2, "sigmoid")

    fig1=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns*rows +1):
        pixels1 = A2[:,i]
        img = pixels1.reshape((28, 28))
        fig1.add_subplot(rows, columns, i)
        plt.imshow(img)
    
    fig2=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns*rows +1):
        pixels2 = XTsBatch[:,i]
        img = pixels2.reshape((28, 28))
        fig2.add_subplot(rows, columns, i)
        plt.imshow(img)

    fig3=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns*rows +1):
        pixels3 = noisyXTsBatch[:,i]
        img = pixels3.reshape((28, 28))
        fig3.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def autoencoder(parameters, inputdata):
    model=Model()
    W1,b1 = parameters["W1"],parameters["b1"]
    A1,cache1= model.layer_forward(inputdata.T, W1, b1, "relu")
    return A1

def softmaxTr(net_dims, X, X_val, Y, Yval):
    learningRate=float(args.learningRate)
    epochs=int(args.epochs)
    decayRate = float(args.decayRate)
    batchSize=int(args.batchSize)
    model=Model()
    A0=X
    A0_=X_val
    costs = []
    costs_ = []
    n_in, n_h, n_fin = net_dims
    parameters = model.initialize_2layer_weights(n_in, n_h, n_fin)
    W1,b1 = parameters["W1"],parameters["b1"]
    for ii in range(epochs):
        noBatches = int(X.shape[0]/batchSize)
        learningRate = learningRate*(1/(1+decayRate*ii))
        print(learningRate)
        print("Epoch: ",ii )
        for jj in range(noBatches):
            W1,b1 = parameters["W1"],parameters["b1"]
            AL,cache1=model.layer_forward(A0, W1, b1, "linear")
            A,cache2,cost = model.softmax_cross_entropy_loss(AL, Y)
            dZ = model.softmax_cross_entropy_loss_der(Y, cache2)
            dA_prev, dW1, db1 = model.layer_backward(dZ, cache2, W1, b1, "linear")
            parameters['W1']+=-learningRate*dW1
            parameters['b1']+=-learningRate*db1

            if jj % 50 == 0:
                print("Cost at iteration %i is: %f" %(jj*50, cost))

        costs.append(cost)
        AL_,cache1_=model.layer_forward(A0_, W1, b1, "linear")
        A_,cache2_,cost_ = model.softmax_cross_entropy_loss(AL_, Yval)
        costs_.append(cost_)

    return costs,costs_, parameters

def multi_layer_network(X, Y,X_val, Y_val, net_dims, num_iterations=100, learning_rate=0.2, decay_rate=0.01):

    parameters = initialize_multilayer_weights(net_dims)
    A0 = X
    A0_ = X_val
    costs = []
    costs_ = []
    for ii in range(num_iterations):

        AL,cache1 = multi_layer_forward(A0, parameters)
        A,cache2,cost = softmax_cross_entropy_loss(AL, Y)

        # Backward Prop
        dZ = softmax_cross_entropy_loss_der(Y, cache2)
        grads = multi_layer_backward(dZ, cache1, parameters)
        parameters, alpha = update_parameters(parameters, grads, num_iterations, learning_rate, decay_rate=0.0)
        
        if ii % 10 == 0:
            costs.append(cost)
            AL_,cache1_ = multi_layer_forward(A0_, parameters)
            A_,cache2_,cost_ = softmax_cross_entropy_loss(AL_, Y_val)
            costs_.append(cost_)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, alpha))
    
    return costs,costs_, parameters

def main(args):
    data = myDataset(args)
    train_data, val_data, test_data = data.getTrData(), data.getValData(), data.getTsData()

    m,n_in = train_data.shape
    n_fin = n_in
    n_h1 = 500
    net_dims = [n_in, n_h1, n_fin]

    args.learningRate = args.learningRate
    epochs = int(args.epochs)
    plot_costs=[]
    plot_costs_=[]
    test_acc=[]

    costs,costs_, parameters = train(train_data, val_data, net_dims, \
        epochs=epochs, learningRate=args.learningRate, decayRate = args.decayRate)
    plot_costs.append(costs)
    plot_costs_.append(costs_)
    
    it =  list(range(0,epochs))
    plt.plot(it, costs, label='train')
    plt.plot(it, costs_, label='val')
    plt.title('Train_cost and Val_cost vs iterations')
    plt.xlabel('iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

    inputnextTr1=autoencoder(parameters,train_data)
    inputnextVal1=autoencoder(parameters,val_data)

    m,n_in = inputnextTr1.shape
    n_fin = n_in
    n_h2 = 100
    net_dims = [n_in, n_h2, n_fin]

    plot_costs=[]
    plot_costs_=[]
    test_acc=[]

    costs,costs_, parameters = train(inputnextTr1, inputnextVal1, net_dims, \
        epochs=epochs, learningRate=args.learningRate, decayRate = args.decayRate)
    plot_costs.append(costs)
    plot_costs_.append(costs_)
    
    it =  list(range(0,epochs))
    plt.plot(it, costs, label='train')
    plt.plot(it, costs_, label='val')
    plt.title('Train_cost and Val_cost vs iterations')
    plt.xlabel('iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

    inputnextTr2=autoencoder(parameters,inputnextTr1)
    inputnextVal2=autoencoder(parameters,inputnextVal1)
    ##net_dims == ??
    costs,costs_, parameters = softmaxTr(net_dims,inputnextTr2,inputnextVal2, train_labels, val_labels)

    net_dims = [784,n_h1,n_h2,10]
    print("Network dimensions are:" + str(net_dims))

    train_data, val_data, test_data = data.getTrData(), data.getValData(), data.getTsData()

    costs, costs_,parameters = model.multi_layer_network(train_data, train_label,val_data, val_label, net_dims, \
            num_iterations=num_iterations, learning_rate=learningRate)

    # compute the accuracy for training set and testing set
    train_Pred = model.classify(train_data, parameters)
    test_Pred = model.classify(test_data, parameters)

    trAcc = model.accuracy(train_Pred, train_label)
    teAcc = model.accuracy(test_Pred, test_label)
    plot_acc.append(teAcc)
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Validation cost is{0:0.3f} %".format(costs_[-1]))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

    ## change multi_layer_neural net
    ## check dims

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
    parser.add_argument('--decayRate', default='0.5',
                        help="decay rate for learning rate")

    args = parser.parse_args()

    # args.id += '--batchSize' + str(args.batchSize)
    # args.id += '--learningRate' + str(args.learningRate)
    # args.id += '--noTraining' + str(args.noTraining)
    # args.id += '--noTesting' + str(args.noTesting)
    # args.id += '--cost' + str(args.cost)
    # args.id += '--dataPath' + str(args.dataPath)
    # args.id += '--noValidation' + str(args.noValidation)
    
    main(args)