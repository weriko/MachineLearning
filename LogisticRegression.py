import matplotlib.pyplot as plt
import numpy as np
from numpy import log
import random as rd

class Regressor:
    e = 2.7182818284
    def __init__(self , x,y,lr = 0.03,epsilon =0.0000001 ):
        self.lr = lr
        self.epsilon = epsilon
        if isinstance(x, np.ndarray):
            self.x = x
            self.y = y
        else:
            self.x = np.array(x)
            self.y = np.array(y)
        self.n_examples = self.x.shape[0]
        self.weights = np.random.rand(self.x.shape[1])
        self.b = 0
        self.losses = []
        
            
    def plot(self):
        plt.scatter(self.x,self.y)
        plt.show()
    def sigmoid(self,x):
        return (1/(1+np.exp(-x)+self.epsilon))
    def loss(self,y,ypred):
        
       
        return -sum(y*log(ypred+self.epsilon) + (1-y)*log(1-ypred+self.epsilon) )/self.n_examples
    def forward(self):
      
        return self.sigmoid(np.dot(self.x,self.weights)+self.b)
    def regress(self,epochs = 10):
        
        for i in range(epochs):
            ypred = self.forward()
            self.weights -= self.lr *( np.dot(self.x.T, ypred-self.y)/self.x.shape[0])
            self.b -= np.sum(ypred - y)/self.x.shape[0]*self.lr
            self.losses.append(self.loss(self.y,ypred))
            
    def predict(self,x):
        return self.sigmoid(np.dot(x,self.weights)+self.b)
    def plot_loss(self):
        plt.plot(np.arange(0,len(self.losses)),self.losses)
        plt.show()
            
        
        
        
        
        
        
x = [[rd.randint(0,20),rd.randint(0,20)] for i in range(1000)]
y = [1 if sum(i)>=21 else 0 for i in x]
"""
x = [[10,3],
      [10,10],
    [10,4],
    [5,20],
    [10,2]]
y = [0,1,0,1,0]
"""
            
      
    
x = np.array(x)
y = np.array(y)
print(x.shape[1])
regressor = Regressor(x,y,lr=0.03)

regressor.regress(epochs = 2000)
print(regressor.predict(np.array([[12,3],
                                  [12,4],
                                  [12,6],
                                  [35,42],
                                  [4,27],
                                  [11,5],
                                  [2,3],
                                  [5,3],
                                  [10,8],
                                  [20,0],
                                  [10,10],
                                  [10,11]
                                 ])))

#print(regressor.losses)
regressor.plot_loss()
print(regressor.weights)
print(regressor.b)
