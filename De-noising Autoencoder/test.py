def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    W1,b1 = parameters["W1"],parameters["b1"]
    W2,b2 = parameters["W2"],parameters["b2"]

    A1,cache1=layer_forward(X, W1, b1, "tanh")
    A2,cache2 = layer_forward(A1, W2, b2, "sigmoid")
    YPred = (A2>=0.5).astype(int)
    return YPred

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
    
    main()