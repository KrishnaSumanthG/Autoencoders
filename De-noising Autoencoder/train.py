'''
This file implements a two layer neural network for a binary classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb

def two_layer_network(X, Y,X_val,Y_val, net_dims, num_iterations=2000, learning_rate=0.1):
    '''
    Creates the 2 layer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''

    ########################
    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    obs = len(Y.T)
    A0 = X
    A0_= X_val
    costs = []
    costs_ = []
    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE

        W1,b1 = parameters["W1"],parameters["b1"]
        W2,b2 = parameters["W2"],parameters["b2"]

        A1,cache1=layer_forward(A0, W1, b1, "tanh")
        A2,cache2 = layer_forward(A1, W2, b2, "sigmoid")

        # cost estimation
        ### CODE HERE

        cost = cost_estimate(A2, Y)
        
        # Backward Propagation
        ### CODE HERE

        dA= ((A2 - Y)/ (A2*(1.0 - A2)))/obs

        dA_prev2, dW2, db2 = layer_backward(dA, cache2, W2, b2, "sigmoid")
        dA_prev1, dW1, db1 = layer_backward(dA_prev2, cache1, W1, b1, "tanh")
        #print(db1.shape)

        #update parameters
        ### CODE HERE
        parameters['W2']+=-learning_rate*dW2
        parameters['b2']+=-learning_rate*db2
        parameters['W1']+=-learning_rate*dW1
        parameters['b1']+=-learning_rate*db1


        if ii % 10 == 0:
            costs.append(cost)
            A1_,cache1_=layer_forward(A0_, W1, b1, "tanh")
            A2_,cache2_ = layer_forward(A1_, W2, b2, "sigmoid")
            cost_ = cost_estimate(A2_, Y_val)
            costs_.append(cost_)

        if ii % 10 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
    
    return costs,costs_, parameters

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return (1.0 - (float(np.count_nonzero(diff)) / len(diff.T)))*100

def main():
    # getting the subset dataset from MNIST
    # binary classification for digits 1 and 7
    digit_range = [1,7]
    train_data, train_label, val_data, val_label, test_data, test_label = \
            mnist(noTrSamples=2400,noValSamples=400,noTsSamples=1000,\
            digit_range=digit_range,\
            noTrPerClass=1200, noValPerClass=200, noTsPerClass=500)
    
    #convert to binary labels
    train_label[train_label==digit_range[0]] = 0
    train_label[train_label==digit_range[1]] = 1
    val_label[val_label==digit_range[0]] = 0
    val_label[val_label==digit_range[1]] = 1
    test_label[test_label==digit_range[0]] = 0
    test_label[test_label==digit_range[1]] = 1

    n_in, m = train_data.shape
    n_fin = 1
    n_h = 100
    net_dims = [n_in, n_h, n_fin]
    # initialize learning rate and num_iterations
    learning_rate = 1.0
    num_iterations = 400
    plot_costs=[]
    plot_costs_=[]
    test_acc=[]
    net_dims = [n_in, n_h, n_fin]

    costs,costs_, parameters = two_layer_network(train_data, train_label, val_data, val_label, net_dims, \
        num_iterations=num_iterations, learning_rate=learning_rate)
    plot_costs.append(costs)
    plot_costs_.append(costs_)
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)

    trAcc = accuracy(train_Pred, train_label)
    teAcc = accuracy(test_Pred, test_label)
    test_acc.append(teAcc)
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Validation cost is{0:0.3f} ".format(costs_[-1]))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    # CODE HERE TO PLOT costs vs iterations
    
    it =  list(range(0,num_iterations,10))
    plt.plot(it, costs, label='train')
    plt.plot(it, costs_, label='val')
    plt.title('Train_cost and Val_cost vs iterations')
    plt.xlabel('iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()




