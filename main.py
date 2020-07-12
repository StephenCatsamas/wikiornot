from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.datasets import make_blobs
   
class Perceptron:
    
    def __init__(self, 
                 weights,
                 bias=1,
                 learning_rate=0.3):
        """
        'weights' can be a numpy array, list or a tuple with the
        actual values of the weights. The number of input values
        is indirectly defined by the length of 'weights'
        """
        self.weights = np.array(weights)
        self.bias = bias
        self.learning_rate = learning_rate
        
    @staticmethod
    def unit_step_function(x):
        if  x <= 0:
            return 0
        else:
            return 1
        
    def __call__(self, in_data):
        in_data = np.concatenate( (in_data, [self.bias]) )
        result = self.weights @ in_data
        return Perceptron.unit_step_function(result)
    
    def adjust(self, 
               target_result, 
               in_data):
        if type(in_data) != np.ndarray:
            in_data = np.array(in_data)  # 
        calculated_result = self(in_data)
        error = target_result - calculated_result
        if error != 0:
            in_data = np.concatenate( (in_data, [self.bias]) )
            correction = error * in_data * self.learning_rate
            self.weights += correction
            
    def evaluate(self, data, labels):
        evaluation = Counter()
        for sample, label in zip(data, labels):
            result = self(sample) # predict
            if result == label:
                evaluation["correct"] += 1
            else:
                evaluation["wrong"] += 1
        return evaluation

def labelled_samples(n):
    for _ in range(n):
        s = np.random.randint(0, 2, (2,))
        yield (s, 1) if s[0] != s[1] else (s,0)

p1 = Perceptron(weights=[0.3, 0.3, 0.3],
               learning_rate=0.005)
p2 = Perceptron(weights=[0.3, 0.3, 0.3],
               learning_rate=0.005)
p3 = Perceptron(weights=[0.3, 0.3, 0.3],
               learning_rate=0.2)

def evaluate_network(data, labels):
        evaluation = Counter()
        for sample, label in zip(data, labels):
            output = np.array([p1(sample), p2(sample)])
            result = p3(output) # predict
            if result == label:
                evaluation["correct"] += 1
            else:
                evaluation["wrong"] += 1
        return evaluation


for in_data, label in labelled_samples(100):
    p1.adjust(label, 
             in_data)
    p2.adjust(label, 
             in_data)
    output = np.array([p1(in_data), p2(in_data)])
    p3.adjust(label,  
             output)
    

test_data, test_labels = list(zip(*labelled_samples(1)))
print(test_data, test_labels)

evaluation = evaluate_network(test_data, test_labels)

print(evaluation)



fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
X = np.arange(xmin, xmax, 0.1)
'''ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="r")
ax.scatter(1, 1, color="g")'''
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
m = -p.weights[0] / p.weights[1]
c = -p.weights[2] / p.weights[1]
print(m, c)
ax.plot(X, m * X + c )
plt.plot()







