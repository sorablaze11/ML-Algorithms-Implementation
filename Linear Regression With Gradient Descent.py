import numpy as np
import matplotlib.pyplot as plt

# Defining Cost Function
def costFunction(x, y, parameters):
    return np.dot((np.dot(X,parameters) - Y).T, (np.dot(X,parameters) - Y))/(2*len(Y))

# Generating Random Data
x = np.random.uniform(-4, 4, 450)
X = np.c_[np.ones(450), x]

#Creating another numpy array for Gradient Calculation
X_grad=np.c_[x].T
Y = np.c_[x + np.random.standard_normal(450)+2.5]

# Plotting the initial dataset graph
plt.plot(x, Y, 'o')
plt.show()

#Initializing Hyper-parameters
numOfIterations = 40
alpha = 0.1
costLimit = 2
cost_history = []
parameters = np.array([[0, 0]]).T

#Running Gradient Descent
for i in range(numOfIterations):
    temp0 = np.sum(parameters[0] - alpha * (1 / len(Y)) * np.sum((np.dot(X, parameters) - Y)))
    temp1 = np.sum(parameters[1] - alpha * (1 / len(Y)) * np.sum(np.dot(X_grad, (np.dot(X, parameters) - Y))))
    parameters = np.array([[temp0], [temp1]])
    cost_history.append([costFunction(X, Y, parameters), i + 1])
#plt.plot(cost_history[:][1], cost_history[:][1]);
#plt.show()
print("Final Parameters : {}, {}".format(parameters[0], parameters[1]))
J = costFunction(X, Y, parameters)
print("Final Cost : {}".format(J))
