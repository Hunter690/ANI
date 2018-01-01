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

#### LR Cost Function

Given m training samples, want y-hat(i) to be equivalent to y(i) where y-hat(i) is specific to that single training sample  
Ex. Given 5 math problems, y-hat(3) would be the probability that the answer to question 3 is right  
Loss (error) function: L(y-hat, y) = 1/2(y-hat - y)^2  
Given y-hat and y, you can find the amount of error you have  
   Knowing error lets tells you how much you can trust your answe

##### Taken from Week 2 Optional Video but relavent to LR Cost function

If y = 1: p(y|x) = y-hat  
If y = 0: p(y|x) = 1-y-hat  
Generalizing these two equations: p(y|x) = y-hat^y(1-y-hat)^(1-y)
Intuitive reasoning behind equation: plug in (when y = 1, p(y|x) = y-hat & when y = 0, p(y|x) = 1-y-hat)

![alt text](

Taking the log of both sides:
log(p(y|x)) = log(y-hat^y(1-y-hat)^(1-y))  
            = ylog(y-hat) + (1-y)log(1-y-hat)
            = L(y-hat, y)
log(p(y|x)) = -L(y-hat, y)  
Negative because you want to minimize loss function

Total probability of the all the predictions made on a training set:


