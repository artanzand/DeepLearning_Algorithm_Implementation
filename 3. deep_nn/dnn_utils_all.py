import numpy as np
import matplotlib.pyplot as plt



def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array
        A scalar or numpy array of any size.

    Returns
    -------
    s : array
        sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    
    return s



def relu(Z):
    """
    Compute ReLU

    Parameters
    ----------
    Z : array
        Output of the linear layer

    Returns
    -------
    A : array
        Post-activation parameter, of the same shape as Z
    cache : dictionary 
        containing Z
    """

    A = np.maximum(0, Z)

    assert A.shape == Z.shape

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Parameters
    ----------
    dA : array
        post-activation gradient
    cache : array
        Z for computing backward propagation

    Returns
    -------
    dZ : array
        Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # copying dA

    # When z <= 0, set dz to 0
    dZ[Z <= 0] = 0

    assert dZ.shape == Z.shape

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Parameters
    ----------
    dA : array
        post-activation gradient
    cache : array
        Z for computing backward propagation

    Returns
    -------
    dZ : array
        Gradient of the cost with respect to Z
    """

    Z = cache

    s = sigmoid(Z)
    dZ = dA * s * (1 - s)

    assert dZ.shape == Z.shape

    return dZ
  


def initialize_parameters_deep(layer_dims):
    """
    Parameters
    ----------
    layer_dims : list
        containing the dimensions of each layer

    Returns
    -------
    parameters : dictionary
        containing weight matrix W, bias vector b for each layer
    """

    np.random.seed(2021)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = (
            np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        )
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_fwd(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Parameters
    ----------
    A_prev : array
        activations from previous layer
    W : array
        weights matrix
    b : array
        bias vector

    Returns:
    Z : array
        pre-activation parameter
    cache : tuple
        containing "A", "W" and "b"
    """

    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)

    return Z, linear_cache


def linear_activation_fwd(A_prev, W, b, activation):
    """
    Compute forward propagation for the LINEAR->ACTIVATION layer

    Parameters
    ----------
    A_prev : array
        activations from previous layer
    W : array
        weights matrix
    b : array
        bias vector
    activation : str
        the activation to be used in this layer("sigmoid" or "relu")

    Returns
    -------
    A : array
        the post-activation value
    cache : tuple
        containing "linear_cache" and "activation_cache"
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_fwd(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_fwd(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation computation

    Parameters
    ----------
    X : array
        input data of shape (input size, number of examples)
    parameters : dict
        output of initialize_parameters_deep()

    Returns
    -------
    AL : array
        activation value from the output (last) layer
    caches : list
        containing every cache of linear_activation_fwd()
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers, 2 parameters per layer

    # Implement [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_fwd(
            A_prev,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            activation="relu",
        )
        caches.append(cache)

    # Implement LINEAR -> SIGMOID
    AL, cache = linear_activation_fwd(
        A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
    )
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function.

    Parameters
    ----------
    AL : array
        probability vector corresponding to label predictions
    Y : array
        true label vector with same shape as AL

    Returns
    -------
    cost : float
        cross-entropy cost
    """
    m = Y.shape[1]

    cost = -1 / m * (np.dot(np.log(AL), Y.T) + np.dot((1 - Y), np.log(1 - AL).T))
    cost = np.squeeze(cost)  # To turn array to float

    return cost

def linear_backward(dZ, linear_cache):
    """
    linear portion of backward propagation for a single layer

    Parameters
    ----------
    dZ : array
        Gradient of the cost with respect to the linear output (of current layer l)
    cache : tuple
        values (A_prev, W, b) coming from the forward propagation

    Returns
    -------
    dA_prev : array
        Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW : array
        Gradient of the cost with respect to W (current layer l), same shape as W
    db : array
        Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Parameters
    ----------
    dA : array
        post-activation gradient for current layer l
    cache : tuple
        of values (linear_cache, activation_cache) we stored during forward propagation
    activation : str
        the activation to be used in this layer("sigmoid" or "relu")

    Returns
    -------
    dA_prev : array
        Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW : array
        Gradient of the cost with respect to W (current layer l), same shape as W
    db : array
        Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    backward propagation for [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID

    Parameters
    ----------
    AL : array
        probability vector from each forward propagation
    Y : array
        true label vector
    caches : list
        of caches containing every cache of linear_activation_fwd() with "relu"
        and the cache of linear_activation_fwd() with "sigmoid"

    Returns
    -------
    grads : dictionary
        of gradients dA, db and dW for each layer
    """
    grads = {}
    L = len(caches)  # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # To make Y the same shape as AL

    # Initializing the backpropagation
    dAL = -(
        np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)
    )  # derivative of cost with respect to AL

    # Last layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
        dA=dAL, cache=current_cache, activation="sigmoid"
    )
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            dA=grads["dA" + str(l + 1)], cache=current_cache, activation="relu"
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(params, grads, alpha):
    """
    Update parameters using gradient descent
    
    Parameters
    ----------
    params : dictionary 
        containing parameters W and b
    grads : dictionary 
        containing gradients
    alpha : float
        learning rate
    
    Returns
    -------
    parameters : dictionary 
        containing your updated parameters W and b
    """
    parameters = params.copy()
    L = len(parameters) // 2  # 2 parameters per layer

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - alpha * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - alpha * grads["db" + str(l+1)]

    return parameters


def predict(X, y, parameters):
    """
    predict the results of a  L-layer neural network.
    
    Parameters
    ----------
    X : array
        data set of examples
    parameters : dict
        W and b parameters of the trained model
    
    Returns
    -------
    p : array
        predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p
