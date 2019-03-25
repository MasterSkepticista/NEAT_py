import numpy as np



X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0, 1, 1, 0]]
X = np.asarray(X)
X = X.T
Y = np.asarray(Y)

print(X)
print(Y)
#-------------------------------------------------------------------------------
def initialize_params(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    parameters = {  "W1": W1,
                    "b1":   b1,
                    "W2":   W2,
                    "b2":   b2}
    return parameters

#-------------------------------------------------------------------------------

def forward_prop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {   "Z1":   Z1,
                "A1":   A1,
                "Z2":   Z2,
                "A2":   A2}
    return A2, cache
#-------------------------------------------------------------------------------

def sigmoid(M):
    return  1/(1 + np.exp(-M))
#-------------------------------------------------------------------------------
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    cost = -np.sum(1 / m * (np.multiply(Y, np.log(A2))
                            + np.multiply((1 - Y), np.log(1 - A2))))
    cost = np.squeeze(cost)

    assert(isinstance(cost, float))
    return cost
#-------------------------------------------------------------------------------
def backward_prop(cache, Y, parameters, X):

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, np.transpose(A1))
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.multiply(np.dot(np.transpose(W2), dZ2), (1 - np.power(A1, 2)))
    dW1 = 1 / m * np.dot(dZ1, np.transpose(X))
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {   "dW1":  dW1,
                "db1":  db1,
                "dW2":  dW2,
                "db2":  db2}
    return grads
#-------------------------------------------------------------------------------
def update_weights(parameters, grads, lr = 0.1):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - dW1 * lr
    b1 = b1 - db1 * lr
    W2 = W2 - dW2 * lr
    b2 = b2 - db2 * lr

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
#-------------------------------------------------------------------------------
def network_train(X, Y, n_h, num_iterations = 100, print_cost = False):
    np.random.seed(3)
    n_x = 2

    n_y = 1

    parameters = initialize_params(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(num_iterations):

        #Do forward pass
        A2, cache = forward_prop(X, parameters)
        #Compute cost
        cost = compute_cost(A2, Y, parameters)
        #Do backward pass
        grads = backward_prop(cache, Y, parameters, X)
        #Update Weights
        parameters = update_weights(parameters, grads, lr = 1.5)
        #Optionally print cost
        if(i % 10 == 0 and print_cost == True):
            print("Cost @ iter %i : %f" %(i,cost))

    return parameters
#----------------------------Main code here-------------------------------------

parameters = network_train(X, Y, 4, 100, True)
predictions, cache = forward_prop(X, parameters)
print(predictions)
