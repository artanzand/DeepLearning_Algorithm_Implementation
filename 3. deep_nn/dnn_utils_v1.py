import numpy as np


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
