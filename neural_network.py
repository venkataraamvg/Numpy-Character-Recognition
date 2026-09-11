import numpy as np

def init_params(input_size=784, hidden_size=128, output_size=35):
    """
    Initializes the weights and biases for a 2-layer neural network.
    
    Objective:
    To establish the initial trainable parameters (W1, b1, W2, b2) required for 
    forward and backward propagation. Weights are initialized randomly to break 
    symmetry. Biases start at zero.
    """
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
    b2 = np.zeros((1, output_size))
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def relu(Z):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.
    
    Objective:
    Introduces non-linearity into the network, allowing it to learn complex 
    patterns. Outputs the input directly if positive, otherwise outputs zero.
    """
    return np.maximum(0, Z)

def softmax(Z):
    """
    Applies the Softmax activation function for multi-class classification.
    
    Objective:
    Converts raw output scores into normalized probability distributions across 
    all 35 classes. Subtracting the max Z value ensures numerical stability.
    """
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward_propagation(X, params):
    """
    Executes the forward pass of the neural network.
    
    Objective:
    Passes the input data through the hidden and output layers to generate 
    predictions. Stores intermediate values in a cache required for backpropagation.
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_loss(Y, A2):
    """
    Calculates the categorical cross-entropy loss.
    
    Objective:
    Quantifies the difference between the true one-hot encoded labels and the 
    predicted probabilities.
    """
    m = Y.shape[0]
    epsilon = 1e-15 # Prevents log(0) errors
    loss = - (1 / m) * np.sum(Y * np.log(A2 + epsilon))
    return loss

def relu_derivative(Z):
    """
    Calculates the derivative of the ReLU function.
    
    Objective:
    Required for the chain rule during backpropagation. The derivative is 1 
    if Z > 0, and 0 if Z <= 0.
    """
    return (Z > 0).astype(float)

def back_propagation(X, Y, params, cache):
    """
    Executes the backward pass using calculus to find gradients.
    
    Objective:
    Computes the partial derivatives of the loss function with respect to every 
    weight and bias to determine how they should be adjusted.
    """
    m = X.shape[0]
    
    W2 = params["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    
    # Output Layer Gradients
    dZ2 = A2 - Y 
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden Layer Gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(params, grads, learning_rate):
    """
    Updates the network's weights and biases.
    
    Objective:
    Applies gradient descent to adjust the parameters in the direction that 
    minimizes the cross-entropy loss.
    """
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    
    return params