# ANI
Deep Learning coursera.org https://www.coursera.org/specializations/deep-learning

### Week 1

#### What is a Neural Network?

Given some input, some calculations occur & a final output comes out.

The # of training sets you give a neural network is like how many problems you give a kid preparing for a math test.  
The # of training sets is denoted: m  
m training examples: {(x1, y1), (x2, y2)... (xm, ym)} where x is a single instance of a problem (like one math problem) and y is the answer.

X = [x1 x2 xm] where X is an m x nx matrix
(the x in nx refers to the descriptions in the problem Ex. math problem says: 5+5, nx would be 3 because 5, +, 5 are used.)

### Week 2

#### Logistic Regression (LR)

Given x (where x can be something like a picture), calculate y-hat.  
Y-hat is the probability that something is true given x (like if it is true that there is a cat in a picture)  
y-hat = P(y = 1|x)  
Parameters: w (an (nx, 1) dimension vector like x) and b (a real number)  
Outputs y-hat = sigmoid(w_transpose * x + b) where z is w_transpose * x + b
sigmoid(z) = 1 / (1 + e^(-z))

![alt text](https://user-images.githubusercontent.com/24757872/34464928-f8baadac-ee5b-11e7-89e0-03e79250097b.png)
