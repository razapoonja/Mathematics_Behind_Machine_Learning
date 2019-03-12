import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

def predict(i1, i2, w1, w2, b):
	return sigmoid(i1 * w1 + i2 * w2 + b)

def train():
    #random init of weights
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()
    
    iterations = 10000
    learning_rate = 0.1
    
    for i in range(iterations):
        # get a random point
        ri = np.random.randint(len(data))
        point = data[ri]
        
        # networks prediction
        pred = predict(point[0], point[1], w1, w2, b)
        
        target = point[2]
        
        # cost for current random point
        cost = np.square(pred - target)
        
        # derivative of cost with respect to weights and bias
        dcost_dw1 = 2 * (pred - target) * sigmoid_p(pred) * point[0]
        dcost_dw2 = 2 * (pred - target) * sigmoid_p(pred) * point[1]
        dcost_db = 2 * (pred - target) * sigmoid_p(pred) * 1
        
        # updating weights and bias
        w1 -= learning_rate * dcost_dw1
        w2 -= learning_rate * dcost_dw2
        b -= learning_rate * dcost_db
        
    return w1, w2, b


data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]]

test_point = [4.5, 1]

w1, w2, b = train()

pred = predict(test_point[0], test_point[1], w1, w2, b)

print(pred)
