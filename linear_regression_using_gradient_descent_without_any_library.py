def compute_error(b, m, points):
    total_error = 0
    for i in points:
        x = i[0]
        y = i[1]
        total_error += (y - (m * x + b)) ** 2

    return total_error / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in points:
        x = i[0]
        y = i[1]
        b_gradient += (2/N) * (y - ((m_current * x) + b_current))
        m_gradient += (2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current + (learningRate * b_gradient)
    new_m = m_current + (learningRate * m_gradient)

    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, iterations):
    b = starting_b
    m = starting_m
    for i in range(iterations):
        b, m = step_gradient(b, m, points, learning_rate)

    return [b, m]


def predict(x, m, b):
    return (m * x) + b

points = []

fle = open('lin_reg.csv', 'r')
fl = fle.readlines()
for f in fl:
    temp_arr = []
    for val in f.split(','):
        temp_int = float(val)
        temp_arr.append(temp_int)
    points.append(temp_arr)
    temp_arr = []
fle.close()

learning_rate = 0.0001
initial_b = 0
initial_m = 0
iterations = 5000

print('Initial_b: ' + str(initial_b))
print('Initial_m: ' + str(initial_m))
print('Error: ' + str(compute_error(initial_b, initial_m, points)))
print('Prediction: ' + str(predict(32.16847071685779, initial_m, initial_b)))

print("\nAfter Training\n")
[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, iterations)

print('Updated_b: ' + str(b))
print('Updated_m: ' + str(m))
print('Error: ' + str(compute_error(b, m, points)))
print('Prediction: ' + str(predict(32.16847071685779, m, b)))
