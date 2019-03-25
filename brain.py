import numpy as np

class brain:

    def __init__(self, n_x, n_h, n_y):
        if(isinstance(n_x, int)):

            self.input_nodes = n_x
            self.hidden_nodes = n_h
            self.output_nodes = n_y
            self.W1 = np.random.randn(n_h, n_x)
            self.b1 = np.random.randn(n_h, 1)
            self.W2 = np.random.randn(n_y, n_h)
            self.b2 = np.random.randn(n_y, 1)
            self.score = 0
            self.fitness = 0
            self.lr = 0.5
        else:
            print("Failure. Expected integer")


    def show_weights(self):
        print("W1:", self.W1)
        print("W2:", self.W2)
        print("b1:", self.b1)
        print("b2:", self.b2)

    def copy(self):
        if(isinstance(self, brain)):
            return self
        else:
            print("Failure. Expected object")

    def assign_score(self, value):
        self.score = value
    def assign_fitness(self, value):
        self.fitness = value

    def predict(self, inputs):
        Z1 = np.dot(self.W1, inputs) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)


        return A1, A2

    def sigmoid(self, M):
        return  1/(1 + np.exp(-M))

    def train(self, X, Y):
        m = Y.shape[1]
        A1, A2 = self.predict(X)
        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, np.transpose(A1))
        db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
        dZ1 = np.multiply(np.dot(np.transpose(self.W2), dZ2), (1 - np.power(A1, 2)))
        dW1 = 1 / m * np.dot(dZ1, np.transpose(X))
        db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
        self.W1 = self.W1 - dW1 * self.lr
        self.b1 = self.b1 - db1 * self.lr
        self.W2 = self.W2 - dW2 * self.lr
        self.b2 = self.b2 - db2 * self.lr


    def mutate(self, rate):


        self.W1 = self.mapp(self.W1, rate)
        self.W2 = self.mapp(self.W2, rate)
        self.b1 = self.mapp(self.b1, rate)
        self.b2 = self.mapp(self.b2, rate)


    def mutation(self, value, rate):
        if(np.random.rand() < rate):
            return (value + np.random.uniform(0, 0.1))
        else:
            return value

    def mapp(self, array, rate):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i][j] = self.mutation(array[i][j], rate)
        return array
