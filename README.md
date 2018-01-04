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

When using axis = 0, taking vertical columns of a matrix  
When using axis = 1, taking horizontal rows of a matrix

```python
np.dot() #matrix multiplication where you take row * column
matrix * matrix #element-wise multiplication
```
Broadcasting works on element-wise multiplication
Ex.

1. [1 2 3 4] + 100
   In the computer, it becomes:
   [1 2 3 4] + [100 100 100 100]
2. [[1 2 3] [4 5 6]] + [100 200 300]
   Becomes
   [[1 2 3] [4 5 6]] + [[100 200 300] [100 200 300]]

##### General Principle

```python
m x n matrix +, -, *, or / by a:
   1 x n 
   m x 1
#becomes copied in a way so that the 1 x n & m x 1 matrix become m x n matrices
```

NB Do not use a rank 1 array

```python
assert(matrix_name.shape == (m, n)
#produces an errror if the matrix is not m x n
```

#### Outline of Steps 

1. Find the number of training samples, test samples, and how many parameters there are
Ex. for picture recognition, there is usually about 64x64x3 pixels which means that there are 64x64x3 pixels
2. Combine all of the pixels to one column vector
3. Take the transpose of the combined the pixel column vector with the negative of the amount of training samples  
   Ex. pic_flatten = train_set_x_flatten '
   = train_set_x_orig.reshape(train_set_x_orig.shape[0], - train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*3).T
   test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*3).T  
   This produces a single matrix with all the training samples and another one for all the test samples
4. Normalize all of the pixels by dividing by 255 (255 different options in a single pixel)

![alt text](https://user-images.githubusercontent.com/24757872/34501605-3f24451c-efd5-11e7-84d3-a5571690ea0a.png)

Where W is a column vector (with dimensions n by 1 where n is the number of pixels) of numbers that minimizes the cost function value  
& b is a scalar that also minimizes the cost function value by giving neural network an extra degree of freedom

5. Build a function to take the sigmoid of z
6. Find Yhat aka A by taking the sigmoid(dot product between w.T and X + b)
7. Calculate the loss: ylog(a) + (1-y)log(1-a)
8. Calculate the cost: 1/m * sum(loss)
9. Find dJ/dw: 1/m * dot product between X and (A - Y).T
10. Find dJ/db: 1/m * sum(A - Y)
11. Keep on iterating until you find the value of w & b that produce the smallest cost function value  
   Use w = w - np.dot(learning_rate, dJ/dw)  
   &   b = b - np.dot(learning_rate, dJ/db)
12. Once you find the optimal w & b value, calculate Yhat aka A again using the optimal w & b
13. If the value of A > .5, then there is a cat  
    Else, there is not a cat
    
### Week 3

![alt text](https://user-images.githubusercontent.com/24757872/34536645-6aa1dc84-f08b-11e7-865c-5bb119b33d53.png)

![alt text](https://user-images.githubusercontent.com/24757872/34536708-a4ec6f12-f08b-11e7-9439-c7b8cca2c493.png)

![alt text](https://user-images.githubusercontent.com/24757872/34536740-c499f42e-f08b-11e7-8de7-d4ff7cd059aa.png)

![alt text](https://user-images.githubusercontent.com/24757872/34537285-99a5996a-f08d-11e7-806e-3c14c6c55fce.png)

![alt text](https://user-images.githubusercontent.com/24757872/34586468-b3d9f600-f169-11e7-95e4-979029170a34.png)

Where ReLU(A) = max(0, Z) and leakyReLU(A) = max(Z * .01, Z)

Activiation functions are used because a linear function would make all of the hidden layers useless  
ReLU function is more useful in comparison to the sigmoid function because its rate of change when z is positive
is greater (especially on the extremes of the sigmoid function)  
Sigmoid function is still useful in the calculations for the final A because the probability likely results in a number
between one and zero

![alt text](https://user-images.githubusercontent.com/24757872/34540840-a79de104-f09b-11e7-9971-848fb3e0dec1.png)

Note: derivative is written with respect to a variable rather than the normal notation  
Proof for (dL/dZ)^[1]:  

dL[1]/dZ[1] = (dL/dA[1])(dA[1]/dZ[1])  
= (dL/dZ[2])(dZ[2]/dA[1])(dA[1]/dZ[1])  
= (dL/dZ[2])W[2](dA[1]/dZ[1])  
= W[2]"dZ[2]" dot product g[1]'(Z[1])  

There isn't a 1/m for the dL/dZ terms because 1/m is the averaging constant  
1/m could be included in the dL/dZ equation, but it would add a step for the computer to calculate  
Instead, it is just added to dW and dB

![alt text](https://user-images.githubusercontent.com/24757872/34548471-e91aa668-f0c7-11e7-960e-30a9ad2f5bbe.png)

![alt text](https://user-images.githubusercontent.com/24757872/34549770-dcbe546a-f0d0-11e7-8999-940b6ad9a363.png)

![alt text](https://user-images.githubusercontent.com/24757872/34584177-5aa31516-f160-11e7-8d11-055af7885c56.png)

![alt text](https://user-images.githubusercontent.com/24757872/34584200-78c7d31a-f160-11e7-9c9e-ada7db022ca4.png)

![alt text](https://user-images.githubusercontent.com/24757872/34584241-924b4ede-f160-11e7-820e-8e5a8f59ef25.png)
