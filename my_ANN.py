import numpy as np

X = np.array(([1, 2], [3 , 4], [5, 6]), dtype = float)
Y = np.array(([70], [80], [90]), dtype = float)

X = X/np.amax(X, axis = 0)
Y = Y/100

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

epochs = 5000
learning_rate = 0.5
 
hidden_weights = np.random.uniform(size = (2,3))
hidden_bias = np.random.uniform(size = (1,3)) 

output_weights = np.random.uniform(size = (3,1))
output_bias = np.random.uniform(size = (1,1))

for i in range(epochs):
    hidden_input = np.dot(X, hidden_weights) + hidden_bias
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, output_weights) + output_bias
    output = sigmoid(output_input)

    output_gradient = sigmoid_derivative(output)
    hidden_gradient = sigmoid_derivative(hidden_output)
    
    output_error = Y - output
    output_weight_delta = output_error * output_gradient

    hidden_error = output_weight_delta.dot(output_weights.T)
    hidden_weight_delta = hidden_error * hidden_gradient

    output_weights += hidden_output.T.dot(output_weight_delta) * learning_rate
    hidden_weights += X.T.dot(hidden_weight_delta) * learning_rate

print("Input: \n", X)
print("Actual Output: \n", Y)
print("Predicted Output: \n", output)