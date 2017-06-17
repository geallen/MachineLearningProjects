import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math as mat

# Part 1: Load the data from file
'''YOUR CODE HERE'''
data = np.loadtxt('LR_population_vs_profit.data')
# Part 2: Plot the data in 2D as population (X) vs. profit (y)
plt.figure(1)
'''YOUR CODE HERE'''
plt.xlabel('City Population in 10.000s')
plt.ylabel('Profit in $10.000s')
# If you like, you can see this plot by calling plt.show() at this line. However, in order to be able to plot the estimated lines on top of the same figure in Parts 7 and 9 (that's why an ID (1) is given), it is not be plotted here.)
data_population = data[0:97, 0]
data_profit = data[0:97, 1]

plt.plot(data_population, data_profit, '.r')



# Part 3: Implement the gradient step calculation for theta

def computeStep(X, y, theta):
    '''YOUR CODE HERE'''
    function_result = np.array([0,0], dtype= np.float)
    m = float(len(X))

    d1 = 0.0
    d2 = 0.0
    for i in range(len(X)):
        h1 = np.dot(theta.transpose(), X[i])
        c1 = h1 - y[i]
        d1 = d1 + c1
    j1 = d1/m
    for u in range(len(X)):
        h2 = np.dot(theta.transpose(), X[u])
        c2 = (h2 - y[u]) * X[u][1]
        d2 = d2 + c2
    j2 = d2/m

    function_result[0] = j1
    function_result[1] = j2
    return function_result



# Part 4: Implement the cost function calculation
def computeCost(X, y, theta):
    '''YOUR CODE HERE'''
    m = float(len(X))

    d = 0
    for i in range(len(X)):
        h = np.dot(theta.transpose(), X[i])
        c = (h - y[i])

        c = (c **2)
        d = (d + c)
    j = (1.0 / (2 * m)) * d
    return j


# Part 5: Prepare the data so that the input X has two columns: first a column of ones to accomodate theta0 and then a column of city population data
'''YOUR CODE HERE'''

a = np.ones((97, 1))
b = data_population
b = np.reshape(b, (97, 1))
X = np.concatenate((a, b), axis=1)
ty = np.array([np.ones(len(data[:, 1])), data[:, 1]])

# Part 6: Apply linear regression with gradient descent
num_iter = 1500
alpha_line = [[0.1, '-b'], [0.03, '-r'], [0.01, '-g'], [0.003, ':b'], [0.001, ':r'], [0.0003, ':g']]

theta = np.array([0, 0])
'''YOUR CODE HERE'''
y = data_profit
init_cost = computeCost(X, y, theta)
print 'The initial cost is %f.' % init_cost

plt.figure()
plt.ylim(0, 100)
plt.xlim(0, 10)
final_theta = []
for alpha, line in alpha_line:
    J_history = []
    theta = np.array([0, 0], dtype=float)
    for i in range(num_iter):
        '''YOUR CODE HERE'''
        result = computeStep(X, y, theta)
        theta[0] = theta[0] - (alpha) * (result[0])
        theta[1] = theta[1] - (alpha) * (result[1])

        J_history.append(computeCost(X, y, theta))

    plt.plot(J_history, line, linewidth=3, label='alpha:%5.4f' % alpha)
    final_theta.append(theta)

    print 'Final cost after %d iterations is %f.' % (num_iter, J_history[-1])
plt.legend(fontsize=12)

# Part 7: Plot the resulting line and make predictions with the best performing theta
plt.figure(1)
'''YOUR CODE HERE'''
best_theta = final_theta[2]
plt.plot(X[:, 1], np.dot(X, best_theta), '-', label='Linear regression with gradient descent')


'''YOUR CODE HERE'''
# h = teta0 + teta1 * x1  If we give population to x1, this equation will give us the profit
y1 = best_theta[0] + best_theta[1]*35

y2 = best_theta[0] + best_theta[1]*70

print 'Estimated profit for a city of population 35000 is % 7.2f thousand US dollars' % y1
print 'Estimated profit for a city of population 70000 is % 7.2f thousand US dollars' % y2

# Part 8: Plot cost function as a 2D surface over theta0 and theta1 axes
grid_size = 200
theta0_vals = np.linspace(-10, 10, grid_size)
theta1_vals = np.linspace(-1, 4, grid_size)
theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)
cost_2d = np.zeros((grid_size, grid_size))
for t0 in range(grid_size):
    for t1 in range(grid_size):
        theta = np.array([theta0[t0, t1], theta1[t0, t1]])
        cost_2d[t0, t1] = computeCost(X, y, theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0, theta1, cost_2d, cmap=cm.jet, linewidth=0, antialiased=False, alpha=0.5)
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost')
plt.figure()
plt.contour(theta0, theta1, cost_2d, 100)
plt.plot(best_theta[0], best_theta[1], 'xr')
plt.xlabel('Theta 0')
plt.ylabel('Theta 1')

# Part 9: Calculate optimal theta values using normal equation and then compute the corresponding cost value
'''YOUR CODE HERE'''

theta_normal = np.dot(np.linalg.pinv(X), y)
cost_normal = computeCost(X, y, theta_normal)

print 'Theta parameters obtained by solving the normal equation are  %f and %f.' % (theta_normal[0], theta_normal[1])
print 'Final cost after solving the normal equation is  %f.' % cost_normal
plt.figure(1)
plt.plot(X[:, 1], np.dot(X, theta_normal), '-r', label='Linear regression with normal equation')
plt.legend(fontsize=12)
plt.show()
