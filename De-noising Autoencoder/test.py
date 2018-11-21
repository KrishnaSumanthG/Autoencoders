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