import cupy as np
import copy

def sigmoid(Z):
    """
    A function to implement the sigmoid activation on a numpy ndarray Z

    Arguments:
    Z -- a numpy array of any shape

    Returns:
    A -- output of the sigmoid activation on Z -> it has the same shape as Z
    cache -- returns Z in the form of cache for use during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """
    A function to implement the ReLU activation on a numpy ndarray Z

    Arguments:
    Z -- output of the linear layer of a neuron

    Returns:
    A -- output of the ReLU activation on Z -> it has the same shape as Z
    cache -- returns Z in the form of cache for use during backpropagation
    """
    
    A = np.maximum(0, Z)
    cache = Z

    return A, cache

def relu_backward(dA, cache):
    """
    A function to implement the backpropagation for a relu neuron.

    Arguments:
    dA -- post-activation gradient
    cache -- 'Z' computed before, for implementation backpropagation

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    # initializing dZ to a numpy array with values of dA as the ReLU function is linear
    dZ = np.array(dA, copy=True)

    # Setting the parts of dZ as 0 where Z is also 0
    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, cache):
    """
    A function to implement the backpropagation for a sigmoid neuron.

    Arguments:
    dA -- post-activation gradient
    cache -- 'Z' computed before, for implementation backpropagation

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ

def initialize_parameters(layer_dims):
    """
    A function to initialize all the parameters (w and b numpy arrays) before training.

    Arguments:
    layer_dims -- python list containing the number of neurons of each layer in the neural network

    Returns:
    parameters -- python dictionary coontaining parameters "W1", "b1", "W2", "b2", ... "WL", "bL":
                    Wl -- matrix containing weights
                    bl -- bias vector

    """
    
    parameters = {}
    # extracting number of layers in the neural network
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

def linear_forward(A, W, b):
    """
    A function to apply the linear function to a layer during forward propagation.

    Arguments:
    A -- activation matrix from previous layer
    W -- weight matrix of the current layer
    b -- bias vector of the current layer

    Returns:
    Z -- the input that goes into the activation function
    cache -- tuple containing 'A', 'W', 'b', stored as cache for backpropagation computation

    """

    Z = np.dot(W, A) + b

    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    A function to implement the forward propagation for a layer

    Arguments:
    A_prev -- activation matrix from previous layer
    W -- weight matrix of current layer
    b -- bias vector of current layer

    Returns:
    A -- the output of the activation function
    cache -- tuple containing ((Z, ),( A, W, b)), stored as cache for backpropagation computation
    """

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache

def model_forward(X, parameters):
    """
    A function to implement the forward propagation for all layers of a neural network

    Arguments:
    A_prev -- activation matrix from previous layer
    W -- weight matrix of current layer
    b -- bias vector of current layer

    Returns:
    A -- the output of the activation function
    cache -- tuple containing (( A, W, b), (Z, )), stored as cache for backpropagation computation
    """

    caches = []
    A = X
    # extracting the number of layers in the neural network
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

def has_converged(costs, window=5, threshold=0.001):
    """
    A function to check whether the cost computation has converged.

    Arguments:
    costs -- log of costs after each iteration
    window -- the size of the window of iterations for which convergence needs to be checked
    threshold -- the percentage change in the cost below which convergence can be declared

    Returns:
    True/False -- depending on whether the function has converged
    """

    if len(costs) < window + 1:
        return False
    
    recent = costs[-window:]
    relative_changes = [abs(recent[i] - recent[i-1])/abs(recent[i-1]) for i in range(1, len(recent))]

    return max(relative_changes) < threshold

def compute_cost(AL, Y):
    """
    A function to compute the cost of prediction while training.

    Arguments:
    AL -- probability vector corresponding to the predictions made
    Y -- true label vector for the data

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    epsilon = 1e-8
    AL = np.clip(AL, epsilon, 1 - epsilon)
    cost = -1/m * (np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    """
    A function to implement the linear part of backward propagation for a layer.

    Arguments:
    dZ -- gradient of the cost with respect to Z (output of the linear part of forward propagation)
    cache -- tuple containing (A_prev, W, b) coming from the forward propagation of the current layer

    Returns:
    dA_prev -- gradient of the cost with respect to activation (A) of the previous layer
    dW -- gradient of the cost with respect to W of the current layer
    db -- gradient of the cost with respect to b of the current layer
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)

    db = 1/m * np.sum(dZ, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    A function to implement the backward propagation for a single layer.

    Arguments:
    dA - post-activation gradient for current layer
    cache -- a tuple of values ((A, W, b), (Z, ))
    activation -- the type of activation to be used for the current layer

    Returns:
    dA_prev - gradient of the cost with respect to the activation of the previous layer
    dW -- gradient of the cost with respect to W of the current layer
    db -- gradient of the cost with respect to b of the current layer
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def model_backward(AL, Y, caches):
    """
    A function to implement the backward propagation for all the layers of the neural network.

    Arguments:
    AL -- the probabilty vector of the predictions the network has made
    Y -- true label vector of the data
    caches -- python list containing ((A, W, b), (Z, )) of every layer of the neural network

    Returns:
    grads -- A dictionary with the gradients dA, dW, db for all layers of the neural network
    """

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    
    return grads

def update_parameters(params, grads, learning_rate):
    """
    A function to update parameters using gradient descent

    Arguments:
    params -- dictionary containing the parameters W,b for all layers of the neural network
    grads -- A dictionary with the gradients dA, dW, db for all layers of the neural network

    Returns:
    parameters -- dictionary containing updated parameters
    """

    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]

    return parameters

def get_mini_batches(X, Y, batch_size=256, seed=0):
    """
    A function to separate the data into mini batches depending on the batch size.

    Arguments:
    X -- input features of 'm' training examples
    Y -- labels of 'm' training examples
    batch_size -- number of batches that the data is to be divided into
    seed -- used as input to generate shuffled batches with each iteration

    Returns:
    mini_batches - python list of tuples containing each X and Y batch -> (X_batch, Y_batch)
    """
    
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    num_complete_batches = m // batch_size

    for i in range(num_complete_batches):
        X_batch = X_shuffled[:, (i) * batch_size: (i+1) * batch_size]
        Y_batch = Y_shuffled[:, (i) * batch_size: (i+1) * batch_size]
        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        X_batch = X_shuffled[:, num_complete_batches * batch_size:]
        Y_batch = Y_shuffled[:, num_complete_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches

def run_model(X, Y, layers_dims, learning_rate=0.075, num_iterations = 3000, print_cost=False,
              check_convergence=False, convergence_window=5, convergence_threshold=0.001):
    
    """
    A function to run the neural network model.

    Arguments:
    X -- input features of 'm' training examples
    Y -- labels of 'm' training examples
    layers_dims -- list of the number of neurons in each layer.
    learning_rate -- rate at which gradient descent is executed
    num_iterations -- number of iterations that the model has to run for
    print_cost -- boolean variable to toggle whether to print the cost after each 100th iteration
    check-convergence -- boolean variable to toggle whether to check for convergence while running
    convergence_window -- the size of the window of iterations for which convergence needs to be checked
    convergence_threshold -- the percentage change in the cost below which convergence can be declared

    Returns:
    parameters -- dictionary of final weight and bias matrices for each layer of the neural network after training finishes
    costs -- list of cost values logged after each 100th iteration
    """

    costs = []

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
        
        if check_convergence and i % 100 == 0:
            if has_converged(costs, window=convergence_window, threshold=convergence_threshold):
                print(f"Converged at iteration {i} (relative change < {convergence_threshold})")
                break
    
    return parameters, costs

def run_model_epochs(X, Y, layers_dims, learning_rate=0.075, num_epochs = 100, batch_size=256, print_cost=False,
              check_convergence=False, convergence_window=5, convergence_threshold=0.001):
    
    """
    A function to run the neural network model with the mini-batching.

    Arguments:
    X -- input features of 'm' training examples
    Y -- labels of 'm' training examples
    layers_dims -- list of the number of neurons in each layer.
    learning_rate -- rate at which gradient descent is executed
    num_epochs -- number of iterations for which the model has to run
    batch_size -- number of batches that the data is to be divided into
    print_cost -- boolean variable to toggle whether to print the cost after each 100th iteration
    check-convergence -- boolean variable to toggle whether to check for convergence while running
    convergence_window -- the size of the window of iterations for which convergence needs to be checked
    convergence_threshold -- the percentage change in the cost below which convergence can be declared

    Returns:
    parameters -- dictionary of final weight and bias matrices for each layer of the neural network after training finishes
    costs -- list of cost values logged after each 100th iteration
    """
    
    costs = []

    parameters = initialize_parameters(layers_dims)
    
    for i in range(num_epochs):

        mini_batches = get_mini_batches(X, Y, batch_size, seed=i)
        epoch_cost = 0

        for X_batch, Y_batch in mini_batches:
            AL, caches = model_forward(X_batch, parameters)

            batch_cost = compute_cost(AL, Y_batch)

            grads = model_backward(AL, Y_batch, caches)
            parameters = update_parameters(parameters, grads, learning_rate)

            epoch_cost += batch_cost * X_batch.shape[1] / X.shape[1]
        
        if print_cost and (i % 10 == 0 or i == num_epochs - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(epoch_cost)))
        if i % 10 == 0:
            costs.append(epoch_cost)
        
        if check_convergence and i % 10 == 0:
            if has_converged(costs, window=convergence_window, threshold=convergence_threshold):
                print(f"Converged at iteration {i} (relative change < {convergence_threshold})")
                break
    
    return parameters, costs

def predict(X, y, parameters):
    """
    A function to run the neural network model.

    Arguments:
    X -- input features of 'm' training examples
    Y -- labels of 'm' training examples
    parameters -- dictionary of final weight and bias matrices for each layer of the neural network after training finishes

    Returns:
    p -- the probability matrix of predictions made using the model
    np.sum((p==y)/m) -- the accuracy of the predictions of the model
    """

    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    
    print("Accuracy: " + str(np.sum((p==y)/m)))
    return p, np.sum((p==y)/m)
