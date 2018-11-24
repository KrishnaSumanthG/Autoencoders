import numpy as np
from load_data import myDataset
from model import Model, Noise
import matplotlib.pyplot as plt
import pdb
import argparse

def train(X, Y, X_val, Y_val, net_dims, epochs=2000, learningRate=0.1,costEstimate="MSE", decayRate = 0.5):
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
    print("shapes")
    print(X.shape)
    print(X_val.shape)
    
    for ii in range(epochs):
        noBatches = int(X.shape[0]/batchSize)
        learningRate = learningRate*(1/(1+decayRate*ii))
        print(learningRate)
        print("Epoch: ",ii )
        for jj in range(noBatches):

            XTrBatch, YTrBatch= data.getTrMiniBatch(X, Y)

            W1,b1 = parameters["W1"],parameters["b1"]
            W2,b2 = parameters["W2"],parameters["b2"]

            A1,cache1= model.layer_forward(XTrBatch, W1, b1, "relu")
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
        
        XValBatch, YValBatch = data.getValMiniBatch(X_val, Y_val)
        costs.append(cost)
        A1_,cache1_= model.layer_forward(XValBatch, W1, b1, "relu")
        A2_,cache2_ = model.layer_forward(A1_, W2, b2, "sigmoid")
        if costEstimate == "MSE":
            cost_ = model.MSE(A2_, XValBatch)
        else:
            cost_ = model.crossEntropy(A2_, XValBatch)
        costs_.append(cost_)

    return costs,costs_, parameters

## Uncomment for visualization

# def predict(parameters):
#     data=myDataset(args)
#     model=Model()
#     noise = Noise()
#     XTsBatch, YTsBatch= data.getTsMiniBatch()
#     noisyXTsBatch = noise.GaussianNoise(XTsBatch, sd=0.3)
#     #print(noisyXTrBatch.shape)

#     W1,b1 = parameters["W1"],parameters["b1"]
#     W2,b2 = parameters["W2"],parameters["b2"]

#     A1,cache1= model.layer_forward(noisyXTsBatch, W1, b1, "relu")
#     A2,cache2 = model.layer_forward(A1, W2, b2, "sigmoid")

#     fig1=plt.figure(figsize=(8, 8))
#     columns = 4
#     rows = 4
#     for i in range(1, columns*rows +1):
#         pixels1 = A2[:,i]
#         img = pixels1.reshape((28, 28))
#         fig1.add_subplot(rows, columns, i)
#         plt.imshow(img)
    
#     fig2=plt.figure(figsize=(8, 8))
#     columns = 4
#     rows = 4
#     for i in range(1, columns*rows +1):
#         pixels2 = XTsBatch[:,i]
#         img = pixels2.reshape((28, 28))
#         fig2.add_subplot(rows, columns, i)
#         plt.imshow(img)

#     fig3=plt.figure(figsize=(8, 8))
#     columns = 4
#     rows = 4
#     for i in range(1, columns*rows +1):
#         pixels3 = noisyXTsBatch[:,i]
#         img = pixels3.reshape((28, 28))
#         fig3.add_subplot(rows, columns, i)
#         plt.imshow(img)
#     plt.show()


def autoencoder(parameters, inputdata):
    model=Model()
    W1,b1 = parameters["W1"],parameters["b1"]
    A1,cache1= model.layer_forward(inputdata.T, W1, b1, "relu")
    return A1.T

def softmaxTr(net_dims, X, Xval, Y, Yval):
    data=myDataset(args)
    learningRate=float(args.learningRate)
    epochs=int(args.epochs)
    decayRate = float(args.decayRate)
    batchSize=int(args.batchSize)
    model=Model()
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
            XTrBatch, YTrBatch= data.getTrMiniBatch(X, Y)
            W1,b1 = parameters["W1"],parameters["b1"]
            AL,cache1=model.layer_forward(XTrBatch, W1, b1, "linear")
            A,cache2,cost = model.softmax_cross_entropy_loss(AL, YTrBatch)
            dZ = model.softmax_cross_entropy_loss_der(YTrBatch, cache2)
            dA_prev, dW1, db1 = model.layer_backward(dZ, cache1, W1, b1, "linear")
            parameters['W1']+= -learningRate*dW1
            parameters['b1']+= -learningRate*db1

            if jj % 50 == 0:
                print("Cost at iteration %i is: %f" %(jj*50, cost))

        costs.append(cost)
        XValBatch, YValBatch = data.getValMiniBatch(Xval, Yval)
        AL_,cache1_=model.layer_forward(XValBatch, W1, b1, "linear")
        A_,cache2_,cost_ = model.softmax_cross_entropy_loss(AL_, YValBatch)
        costs_.append(cost_)

    return costs,costs_, parameters

def classifierTrain(X, Y, X_val, Y_val, net_dims, epochs=100, learningRate=0.1, decayRate=0.5):
    learningRate=float(args.learningRate)
    epochs=int(args.epochs)
    decayRate = float(args.decayRate)
    batchSize=int(args.batchSize)
    data=myDataset(args)
    model=Model()
    parameters = model.initialize_multilayer_weights(net_dims)
    costs = []
    costs_ = []
    for ii in range(epochs):
        noBatches = int(X.shape[0]/batchSize)
        learningRate = learningRate*(1/(1+decayRate*ii))
        for jj in range(noBatches):
            XTrBatch, YTrBatch= data.getTrMiniBatch(X, Y)
            A0 = XTrBatch
            AL,cache1 = model.multi_layer_forward(A0, parameters)
            A,cache2,cost = model.softmax_cross_entropy_loss(AL, YTrBatch)

            # Backward Prop
            dZ = model.softmax_cross_entropy_loss_der(Y, cache2)
            grads = model.multi_layer_backward(dZ, cache1, parameters)
            parameters, alpha = model.update_parameters(parameters, grads, epochs, learningRate, decayRate)
            
            if jj % 50 == 0:
                print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(jj, cost, alpha))

        XValBatch, YValBatch = data.getValMiniBatch(X_val, Y_val)
        costs.append(cost)
        AL_,cache1_ = model.multi_layer_forward(XValBatch, parameters)
        A_,cache2_,cost_ = model.softmax_cross_entropy_loss(AL_, YValBatch)
        costs_.append(cost_)    
    return costs,costs_, parameters

def main(args):
    data = myDataset(args)
    train_data, train_label = data.getTrData() 
    val_data, val_label = data.getValData()
    test_data, test_label = data.getTsData()

    initialM, initial_in = train_data.shape
    m,n_in = train_data.shape
    n_fin = n_in
    n_h1 = int(args.n_h1)
    net_dims = [n_in, n_h1, n_fin]

    learningRate = float(args.learningRate)
    decayRate = float(args.decayRate)
    epochs = int(args.epochs)
    costEstimate = args.costEstimate

    plot_costs=[]
    plot_costs_=[]
    test_acc=[]

    costs,costs_, parameters = train(train_data, train_label, val_data, val_label, net_dims, \
        epochs=epochs, learningRate=learningRate, costEstimate=costEstimate, decayRate = decayRate)
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
    n_h2 = int(args.n_h2)
    net_dims = [n_in, n_h2, n_fin]

    plot_costs=[]
    plot_costs_=[]
    test_acc=[]

    costs,costs_, parameters = train(inputnextTr1, train_label, inputnextVal1, val_label, net_dims, \
        epochs=epochs, learningRate=learningRate, decayRate = decayRate)
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
    n_last = len(args.labelRange)
    net_dims = [n_h2, n_last, n_last]
    costs,costs_, parameters = softmaxTr(net_dims,inputnextTr2,inputnextVal2, train_label, val_label)

    net_dims = [initial_in,n_h1,n_h2,n_last]
    print("Network dimensions are:" + str(net_dims))

    costs, costs_,parameters = classifierTrain(train_data, train_label,val_data, val_label, net_dims, \
            epochs=epochs, learningRate=learningRate,decayRate = decayRate)

    # compute the accuracy for training set and testing set
    train_Pred = model.classify(train_data, parameters)
    test_Pred = model.classify(test_data, parameters)

    trAcc = ((np.sum(train_Pred == train_label)) % train_Pred.shape[0])*100.0
    teAcc = ((np.sum(test_Pred == test_label)) % test_Pred.shape[0])*100.0
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
    parser.add_argument('--n_h1', default='500',
                        help="number of neurons in hidden layer 1")
    parser.add_argument('--n_h2', default='150',
                        help="number of neurons in hidden layer 2")

    args = parser.parse_args()
    
    main(args)