import numpy as np
class perception():
    def __init__(self,dimensions):
        self.lr = 1e-2
        self.w = np.random.normal(loc=0,scale=1e-3,size=dimensions)
        self.bias = [0]
    def train(self,x,y,epochs):
        # x: (N×D) y: (N,1)
        # SGD
        N = len(x)
        for i in range(epochs):
            for j in x:
                self.w += x[j]*y[j]*self.lr
                self.bias += y[j]*self.lr
            res = x.dot(self.w.T)+self.bias
            res = np.where(res>0,1,0)
            res = np.where(abs(res-y)==0,1,0)
            print("accuracy%f:",res.sum()/N)
        return self.w,self.bias
    def infer(self,x):
        result = x.dot(self.w.T)+self.bias
        result = np.where(result>0,1,0)
        return result




