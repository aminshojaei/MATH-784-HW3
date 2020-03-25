
import numpy as np
import matplotlib.pylab as plt
#################################################################### Functions


class ANN:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.once = True
 

 
    def activation_function(self,type, x , derivation):
        if type == "Sigmoid" :
            if derivation == False:
                return 1 / (1 + np.exp(-x))
            else:
                s = 1 / (1 + np.exp(-x))
                return s * (1 - s)

        if type == "ReLU":
            if derivation == False:
                return x * (x > 0)
            else:
                return 1. * (x > 0)

        if type == "Softmax":
            expZ = np.exp(x - np.max(x))
            return expZ / expZ.sum(axis=0, keepdims=True)

    



    
 
    def initialize_parameters(self):
        np.random.seed(1)
 
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
 
    def forward(self, X):
        store = {}
 
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.activation_function("ReLU" ,Z, derivation=False)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z
 
        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.activation_function( "Softmax",Z , derivation= False)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z
 
        return A, store
 
   
 
    def backward(self, X, Y, store):
 
        derivatives = {}
 
        store["A0"] = X.T
 
        A = store["A" + str(self.L)]
        dZ = A - Y.T
 
        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)
 
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
 
        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.activation_function( "ReLU", store["Z" + str(l)] , derivation = True)
            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)
 
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
 
        return derivatives
 
    def train(self, X, Y, learning_rate=0.01, n_iterations=50):
        np.random.seed(1)
 
        self.n = X.shape[0]
        
        if self.once == True :
            self.layers_size.insert(0, X.shape[1])
            self.once = False
 
        self.initialize_parameters()
        for loop in range(n_iterations):
            A, store = self.forward(X)
            cost = -np.mean(Y * np.log(A.T+ 1e-8))
            derivatives = self.backward(X, Y, store)
 
            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]
 
            if loop % 100 == 0:
                print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y))

            if loop % 10 == 0:
                self.costs.append(cost)
 
    def predict(self, X, Y):
        A, cache = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100
 
    
 

################################################################## Load Daraset


dataset= np.load(r'C:\Users\a335s717\Desktop\HW2\mnist148.npz')
new_dataset= dataset.files
X = dataset['arr_0']
Y = dataset['arr_1']
Test = dataset['arr_2']


############################################################# Preparing Dataset
Input = []
Output=[]
count = np.zeros((10))
w = np.random.random((28 * 28, 3))
for x, y in zip(X,Y):
    if y in [1, 4, 8]:
        Input.append(x.reshape((28 * 28)) / 255)
        count[y] += 1
        
        if y == [1]:
            y = [1, 0, 0]
        elif y == [4]:
            y = [0, 1, 0]
        elif y == [8]:
            y = [0, 0, 1]   
        Output.append(y)
x_test=[]
for x in Test : # reshape and normalize data
    x_test.append(x.reshape((28 * 28)) / 255)

samples = np.asarray(Input)
labels = np.asarray(Output)
test = np.asarray(x_test)

X_train = samples
Y_train = labels
X_Test = test

print("train input shape: " + str(X_train.shape))
print("Train Output shape: "+ str(Y_train.shape))
print("Test Input shape: " + str(X_Test.shape))




################################################################### main

layers_nodes = [800 ,800, 3] # how many nodes for hidden layers and output layer is needed
   
ann = ANN(layers_nodes)
for i in range(10):  # number of epoch
    print("epoch: ", i)

    # Shuffle dataset before going to trainig
    s = np.arange(X_train.shape[0]) 
    np.random.shuffle(s)
    X_train= X_train[s]
    Y_train = Y_train[s]
    zzz = X_train.shape[0]

    for i in range(0, 300,30): # select a batch of data 
        XX_train = X_train[i:i+30]
        YY_train = Y_train[i:i+30]
        ann.train(XX_train, YY_train, learning_rate=0.1)

    
########################################################################## Plot

print("Train Accuracy:", ann.predict(X_train, Y_train))
test , storage = ann.forward(X_Test)
print("Predicted output: "+ str(test))

for c in range(3):
    plt.subplot(2,2,c+1)
    plt.imshow(Test[c])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Sample test'+str(c+1))
    
for i in range(3):
    if np.argmax(test[i]) == 0: print("predicted class for sample " + str(i+1) + ' is: ',  1)
    if np.argmax(test[i]) == 1: print('predicted class for sample ' +str(i+1)+ ' is: ',  4)
    if np.argmax(test[i]) == 2: print('predicted class for sample ' +str(i+1)+ ' is: ',  8)







