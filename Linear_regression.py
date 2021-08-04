import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('Dataset.txt', delimiter = "\t")
X = data.iloc[:, 0]# 1st value assigned to x
Y = data.iloc[:, 1]# 2nd value assigned to y
plt.scatter(X, Y)
plt.xlabel("Independent Variable(X)")
plt.ylabel("Dependent Variable(Y)")
plt.show()

# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate = how much value changes in each step
epochs = 1000  # The number of iterations to perform gradient descent
n = float(len(X)) # Number of elements in X


Dm = []# initialize list
Dc = []# initialize list


# Performing Gradient Descent and to find m and c for the best fitting line
for i in range(epochs):
    Y_pred = m*X + c  # The current predicted value of Y

    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Dₘ is the value of the partial derivative with respect to m
    D_c = (-2/n) * sum(Y - Y_pred)  # Dₘ is the value of the partial derivative with respect to c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c

    Dm.append(D_m)
    Dc.append(D_c)


print('1st five Derivative with respect to m',Dm[:5])
print('1st five Derivative with respect to m',Dc[:5])


def local_min(a): #find the local minimas
    return [y for i, y in enumerate(a)
            if ((i == 0) or (a[i - 1] >= y))
            and ((i == len(a) - 1) or (y < a[i+1]))]

print('Local minimas in Dm are: ',local_min(Dm))
print('Local minimas in Dc are: ',local_min(Dc))

print('optimum values of m',m)
print('Optimum values of c',c)
# Making predictions
Y_pred = m*X + c
plt.xlabel("Independent Variable(X)")
plt.ylabel("Y_Prediction")
plt.scatter(X, Y_pred)
plt.show()
plt.title("Linear Regression Line for X and Y")
plt.xlabel("Independent Variable(X)")
plt.ylabel("Dependent Variable(Y)")
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='green')  # regression line by the maximum of Y_pred and minimum of Y_pred
plt.show()