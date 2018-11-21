import numpy as np
from load_data import myDataset
from model import Model
import matplotlib.pyplot as plt
import pdb

def train(X, Y,X_val,Y_val, net_dims, epochs=2000, learning_rate=0.1):

    n_in, n_h, n_fin = net_dims
    model=Model()
    parameters = model.initialize_2layer_weights(n_in, n_h, n_fin)
    obs = len(Y.T)
    A0 = X
    A0_= X_val
    costs = []
    costs_ = []
    
    for ii in range(epochs):
        noBatches = X,shape[0]/batchSize
        for jj in range(noBatches):

            XTrBatch = myDataset.getTrMiniBatch(batchSize)
            XValBatch = myDataset.getValMiniBatch(batchSize)
            
            W1,b1 = parameters["W1"],parameters["b1"]
            W2,b2 = parameters["W2"],parameters["b2"]

            A1,cache1= model.layer_forward(XTrBatch, W1, b1, "tanh")
            A2,cache2 = model.layer_forward(A1, W2, b2, "sigmoid")

            cost = model.cost_estimate(A2, Y)

            dA= ((A2 - Y)/ (A2*(1.0 - A2)))/obs

            dA_prev2, dW2, db2 = model.layer_backward(dA, cache2, W2, b2, "sigmoid")
            dA_prev1, dW1, db1 = model.layer_backward(dA_prev2, cache1, W1, b1, "tanh")

            parameters['W2']+=-learning_rate*dW2
            parameters['b2']+=-learning_rate*db2
            parameters['W1']+=-learning_rate*dW1
            parameters['b1']+=-learning_rate*db1


            if ii % 10 == 0:
                costs.append(cost)
                A1_,cache1_= model.layer_forward(XValBatch, W1, b1, "tanh")
                A2_,cache2_ = model.layer_forward(A1_, W2, b2, "sigmoid")
                cost_ = model.cost_estimate(A2_, Y_val)
                costs_.append(cost_)

            if ii % 10 == 0:
                print("Cost at iteration %i is: %f" %(ii, cost))
        

    return costs,costs_, parameters

def main(args):
    digit_range = [""]
    train_data, train_label, val_data, val_label, test_data, test_label = \
            myDataset(noTrSamples=2400,noValSamples=400,noTsSamples=1000,\
            digit_range=digit_range,\
            noTrPerClass=1200, noValPerClass=200, noTsPerClass=500)

    m, n_in = train_data.shape
    n_fin = 10
    n_h = 100
    net_dims = [n_in, n_h, n_fin]
    # initialize learning rate and num_iterations
    args.learning_rate = args.learning_rate
    epochs = args.epochs
    plot_costs=[]
    plot_costs_=[]
    test_acc=[]
    net_dims = [n_in, n_h, n_fin]

    costs,costs_, parameters = train(train_data, train_label, val_data, val_label, net_dims, \
        epochs=epochs, learning_rate=learning_rate)
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

    if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='./data',
                        help="data path")
    parser.add_argument('--noTraining', default='10000',
                        help="number of training examples")
    parser.add_argument('--noValidation', default='1000',
                        help="number of validation examples")
    parser.add_argument('--noTesting', default='1000',
                        help="number of testing examples")
    parser.add_argument('--batchSize', default='64',
                        help="batch size")
    parser.add_argument('--learningRate', default='1e-4',
                        help="initial learning rate")

    args = parser.parse_args()

    args.id += '--batchSize' + str(args.batchSize)
    args.id += '--learningRate' + str(args.learningRate)
    
    main(args)




