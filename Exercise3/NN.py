import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from numpy.matlib import repmat as y

## GAMZE SEN 200201032 MACHINE LEARNING HW3
## Hocam bu lisans hayatimin son odevi. 2 gun ugrastim ve dogru olduguna inaniyorum. Umarim 100 alirim. Hersey icin tesekkur ederim simdiden :)

# initialize the weights between units with the standard normal distribution
# (mean:0 variance:1) and the bias weights with 0 for given network architecture
def initialize(input_size, hidden_size, output_size):
    ## YOUR CODE HERE ##
    W1 = 0
    b1 = 0
    W2 = 0
    b2 = 0

    # W1 is the weights between input layer and hidden layer. Input layer is size of 400, hidden layer is size of 30
    W1 = np.random.randn(input_size, hidden_size)
    # W2 is the weights between hidden layer and output layer. Hidden layer is size of 30, output layer is size of 5
    W2 = np.random.randn(hidden_size, output_size)
    # b1 is weights between bias in first layer and hidden layer. There must be 30 weights for this.
    b1 = np.zeros(hidden_size)
    # b2 is weights between bias in hidden layer and output layer. There must be 5 weights for this.
    b2 = np.zeros(output_size)

    ####################
    return W1, b1, W2, b2

# calculate the output of the activation function for the accummulated signal x
def calcActivation(x):
    ## YOUR CODE HERE ##
    z = 0
    ## We should put the result ,that we found for every node, into activation function
    z = 1.0 / (1 + (np.exp((-1) * x)))
    ####################

    return z

# propagate the input and calculate the output at each layer
def forwardPropagate(X, W1, b1, W2, b2):
    ## YOUR CODE HERE ##
    Z1 = 0
    Z2 = 0

    S1 = 0
    S2 = 0
    ## Here we should find 2 result: first for input and hidden layer then for hidden and output layer.
    ## First I found the result for every node in hidden layer then put them into activation function.
    S1 = np.dot(X, W1)+ b1
    Z1 = calcActivation(S1)

    ## Second I found the result for every node in output layer then put them into activation function.
    S2 = np.dot(Z1, W2) + b2
    Z2 = calcActivation(S2)

    ####################
    return Z1, Z2

# calculate the cost
def calcCost(Z2,y):
    ## YOUR CODE HERE ##
    cost = 0

    ## Here we should calcuate cost. If output was 0 or 1 we should use different equations as in below
    for i in range(len(y)):
        if y[i] == 0:
            cost += (-1) * (np.log(1 - Z2[0, i]))
        elif y[i] == 1:
           cost += (-1) *(np.log(Z2[0, i]))
    ####################
    return cost

# propagate the error and calculate the errors at the output and the hidden layer
def backPropagate(Z1, Z2, y, W2, b2):
    ## YOUR CODE HERE ##
    E2 = 0
    E1 = 0
    Eb1 = 0

    # E2 is the error in output layer. To find it we should exract estimated value from actual output.
    # We should find 5 error because there are 5 node in output layer.
    E2 = Z2 - y

    ## E1 is the error in the hidden layer. To find it we should use the error that we found in output layer and the weights between
    ## output and hidden layer
    ## We should find 30 error because there are 30 node in hidden layer.
    E1 = np.dot(W2, np.transpose(E2))

    ## Eb1 is the error bias for hidden layer. To find it we should use the error that we found in output layer and the weights between
    ## output and bias layer
    ## We should find 1 error because there are 1 bias node in hidden layer.
    Eb1 = np.dot(b2, np.transpose(E2))
    ####################
    return E2, E1, Eb1

# calculate the gradients for weights between units and the bias weights
def calcGrads(X, Z1, Z2, E1, E2, Eb1):
    ## YOUR CODE HERE ##
    d_W1 = 0
    d_b1 = 0
    d_W2 = 0
    d_b2 = 0


    ## In here we should the derivatives for gradients. To find derivative, we should multiply.

    # d_w2 is the derivative for weights between hidden layer and the output layer.
    d_W2 = np.dot(np.transpose(E2), Z1)
    # d_w1 is the derivative for weights between hidden layer and the input layer.
    d_W1 = np.dot(E1, X)
    # d_b2 is the derivative for weights between hidden layer bias and the output layer.
    d_b2 = np.dot(np.transpose(E2), Eb1)
    # d_b1 is the derivative for weights between hidden layer bias and the input layer.
    d_b1 = np.dot(np.transpose(E1), 1)


    ####################
    return d_W1, d_W2, d_b1, d_b2

# update the weights between units and the bias weights using a learning rate of alpha
def updateWeights(W1, b1, W2, b2, alpha, d_W1, d_W2, d_b1, d_b2):
    ## YOUR CODE HERE ##
    # W1 = 0
    # b1 = 0
    # W2 = 0
    # b2 = 0

    ## Here we should update weights with usin the result that we found in calcGrads function

    ## W1 is weights between input and the hidden layer
    W1 = W1 - alpha * (np.transpose(d_W1)) # 400*30
    ## W2 is weights between output and the hidden layer
    W2 = W2 - alpha * (np.transpose(d_W2)) # 30*5
    ## b1 is weights between input bias and the hidden layer
    b1 = b1 - alpha * d_b1
    ## b2 is weights between hidden layer bias and the output layer
    b2 = b2 - alpha * (np.transpose(d_b2))
    ####################
    return W1, b1, W2, b2

np.random.seed(62)
X = np.random.randn(400,1).T
y = [0, 0, 1, 0, 0]
input_size = 400
hidden_size = 30
output_size = 5
alpha = 0.001
num_iter = 100
OUT = []
COST = []

# implement the iterations for neural network training, append the output and
# the cost at each iteration to their corresponding lists
W1, b1, W2, b2 = initialize(input_size, hidden_size, output_size)
for i in range(num_iter):
    ## YOUR CODE HERE ##
    ## Now time to use defined functions above.
    ## First forward propagate and find the values in nodes
    Z1, Z2 = forwardPropagate(X, W1, b1, W2, b2)
    ## Then find the cost
    cost = calcCost(Z2,y)
    ## Append cost to the list to plot the graph
    COST.append(cost)
    ## Append output Z2 value to the list to plot the graph
    OUT.append(Z2)
    ## Then backpropaget the error
    E2, E1, Eb1 = backPropagate(Z1, Z2, y, W2, b2)
    ## Find the gradients
    d_W1, d_W2, d_b1, d_b2 = calcGrads(X, Z1, Z2, E1, E2, Eb1)
    ## And update the weights
    W1, b1, W2, b2 = updateWeights(W1, b1, W2, b2, alpha, d_W1, d_W2, d_b1, d_b2)
    ## Do this for 100 times to see the change
    ####################

# plotting part is already implemented for you
cm_subsection = np.linspace(0, 1, num_iter)
colors = [ cm.cool(x) for x in cm_subsection ]
plt.figure()
plt.hold(True)
for i in range(num_iter): plt.plot(OUT[i][0], color=colors[i])
plt.figure()
plt.hold(True)
plt.plot(COST)
plt.show()
