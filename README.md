# ANI
Deep Learning coursera.org https://www.coursera.org/specializations/deep-learning

### Week 1

#### What is a Neural Network?

Given structured data set, a sequence of calculations produces a final output.

The # of training sets you give a neural network is like how many problems you give a kid preparing for a math test.  
The # of training sets is denoted: m  
m training examples: {(x1, y1), (x2, y2)... (xm, ym)} where x is a single instance of a problem (like one math problem) and y is the answer.

X = [x1 x2 xm] where X is an m by nx matrix
(the x in nx refers to the descriptions in the problem Ex. math problem says: 5+5, nx would be 3 because 5, +, 5 are used.)

### Week 2

#### Logistic Regression (LR)

Given x (where x can be something like a picture), calculate yhat.  
Yhat is the probability that something is true given x (like if it is true that there is a cat in a picture)  
yhat = P(y = 1|x)  
Parameters: w (an (nx, 1) dimension vector like x) and b (a real number)  
Outputs yhat = sigmoid(w_transpose * x + b) where z is w_transpose * x + b
sigmoid(z) = 1 / (1 + e^(-z))

![alt text](https://user-images.githubusercontent.com/24757872/34464928-f8baadac-ee5b-11e7-89e0-03e79250097b.png)

#### LR Cost Function

Given m training samples, want yhat(i) to be equivalent to y(i) where yhat(i) is specific to that single training sample  
Ex. Given 5 math problems, yhat(3) would be the probability that the answer to question 3 is right  
Loss (error) function: L(yhat, y) = 1/2(yhat - y)^2  
Given yhat and y, you can find the amount of error you have  
   Knowing error tells you how much you can trust your answer

##### Taken from Week 2 Optional Video but relavent to LR Cost function

If y = 1: p(y|x) = yhat  
If y = 0: p(y|x) = 1-yhat  
Generalizing these two equations: p(y|x) = yhat^y(1-yhat)^(1-y)  
Intuitive reasoning behind equation: plug in (when y = 1, p(y|x) = yhat & when y = 0, p(y|x) = 1-yhat)

![alt text](https://user-images.githubusercontent.com/24757872/34469782-47abeebe-eeeb-11e7-9b44-2f6bcc9ee6a3.png)

Taking the log of both sides:

log(p(y|x)) = log(yhat^y(1-yhat)^(1-y))  
= ylog(yhat) + (1-y)log(1-yhat)  
= L(yhat, y)  
log(p(y|x)) = -L(yhat, y)

Note: log with base e, really just natural log  
Negative because you want to minimize loss function

Total probability of the all the predictions made on a training set:

![alt text](https://user-images.githubusercontent.com/24757872/34469792-7cef3112-eeeb-11e7-9341-0364cb1ea349.png)

Note: log of the produts = sum of logs (Ex. log(5x10) = log(5) + log(10))    
yhat is in the range (0,1) because it tells probability

The Cost function is the average of all the calculated Loss values

#### Gradient Descent

Want to find w & b that minimize Cost function

![alt text](https://user-images.githubusercontent.com/24757872/34472032-540cd380-ef1d-11e7-8e80-35d48266133b.png)

![alt text](https://user-images.githubusercontent.com/24757872/34472048-b24ae022-ef1d-11e7-9c1b-43c9eecd7d0c.png)

Partial differentiation is used when differentiating more than one variable

#### Computational Graph

J(a, b, c) = 3(a + bc) where J is a function with three parameters aka variables and 3(a + bc) is an example function  
Substituting u = bc, v = a + u, and J = 3v  

![alt text](https://user-images.githubusercontent.com/24757872/34472070-739fbf9a-ef1e-11e7-9078-477dc88ea5db.png)

Where the derivative of the function J with respect to v (denoted as dJ/dv) = 3  
Other derivatives include:  

dJ/da = (dJ/dv)(dv/da) = 3  
dv/da = 1  
dJ/du = (dJ/dv)(dv/du) = 3  
dJ/db = (dJ/du)(du/db) = 3 * 2 = 6  
dJ/dc = (dJ/du)(du/dc) = 3 * 3 = 9  

#### Forward and Backward Propogation

z = w.T * x + b where .T refers to the transpose of a matrix  
yhat = a = sig(z)  
L(a, y) = -(ylog(yhat) + (1 - y)log(1 - yhat)

![alt text](https://user-images.githubusercontent.com/24757872/34474480-e29107bc-ef44-11e7-9f8d-fa642372cf01.png)

da/dz is the derivative of the sigmoid function with respect to z  
dz/dw is the corresponding x(i) while dz/db is always one (therefore dL/dz = dL/dz)

#### LR on m

```python
import numpy as np

J =0
dL/dw1 = 0
dL/dw2 = 0
dL/db = 0

for i = 1 in m:
   z[i] = w.T * x[i] + b
   a[i] = 1/(1-np.exp(z[i))
   J += -(y[i] * np.log(a[i]) + (1-y[i]) * np.log(1 - a[i]))
   dL/dz[i] = a[i] - y[i]
   dL/dw1 += x1[i] * dL/dz[i]
   dL/dw2 += x2[i] * dL/dz[i]
   dL/db += dL/dz[i]
   
J = J/m
```

##### Side Note: Softmax function normalizes a matrix 

#### Vectorization

Vectorization is the omission of loops to increase efficiency of algorithm

##### Vectorization to find z
z = w.T * x + b

Non-vectorized python code:
```python

z = 0
for i in range (n_x):
   z += w[i] * x[i]

z += b
```

Vectorized python code:
```python

z = np.dot(w.T, X) + b
```

##### Vectorization LR Gradient Regression

Note: a derivative with respect to a variable may commonly be seen as the derivative of the variable  
Ex: dL/dz is commonly written as dz  
Due to a lack of clarity, this convention will not be followed  
dL/dz[1] = a[1] - y[1]  
dL/dZ = [dL/dz[1] dL/dz[2] ... dL/dz[m]]  
A = [a[1] ... a[m]]   Y = [y[1] ... y[m]]  
dL/dZ = A - Y = [a[1] - y[1] ... a[m] - y[m]]  

Non-vectorized dL/db

```python

db = 0  

for i in m:
   db += dz[1] ...  
   db += dz[m]

db /= m
```

Completely Vectorized

```python

A = 1/(1-np.exp(Z))
dZ = A - Y
dw = 1/m * np.dot(X, dZ.T)
db = 1/m * np.sum(dZ)

```

#### Broadcasting
