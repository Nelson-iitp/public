
import math
import numpy as np

SIN_xlow, SIN_xhigh =  0, 2*math.pi 
# Define a task 
# ... is a set of known constants

# 1. consider sin tasks
# y = a*sin(x) + b where known constants are KC = (a,b)


# Define points (datapoints) of a task
# ... are the samples from a function that need to be estimated and depends on known constants (task)

# so now define a known function to taks samples from and estimate it
class SINTASK3:
    def __init__(self, task, seed=None):
        self.task = task
        self.rngs = np.random.default_rng(seed)
        self.xlow, self.xhigh =  SIN_xlow, SIN_xhigh
        
    def forward(self, x):
        return  self.task[0] * np.sin(x + self.task[1]) + \
                self.task[2] * np.cos(x + self.task[1]) 

    def sample(self, size):
        x = self.rngs.uniform(self.xlow, self.xhigh, size=(size,1) )
        y = self.forward(x)
        return np.hstack((x,y))

    def space(self, size):
        x = np.expand_dims(np.linspace(self.xlow, self.xhigh, num=size), axis=-1)
        y = self.forward(x)
        return np.hstack((x,y))

    def __str__(self) -> str:
        return str(self.task)
    def __repr__(self) -> str:
        return self.__str__()


class SINTASK2:
    def __init__(self, task, seed=None):
        self.task = task
        self.rngs = np.random.default_rng(seed)
        self.xlow, self.xhigh = SIN_xlow, SIN_xhigh
        
    def forward(self, x):
        return  self.task[0] * np.sin(x) + self.task[1] 

    def sample(self, size):
        x = self.rngs.uniform(self.xlow, self.xhigh, size=(size,1) )
        y = self.forward(x)
        return np.hstack((x,y))

    def space(self, size):
        x = np.expand_dims(np.linspace(self.xlow, self.xhigh, num=size), axis=-1)
        y = self.forward(x)
        return np.hstack((x,y))

    def __str__(self) -> str:
        return str(self.task)
    def __repr__(self) -> str:
        return self.__str__()


class SINTASK:
    def __init__(self, task, seed=None):
        self.task = task
        self.rngs = np.random.default_rng(seed)
        self.xlow, self.xhigh = SIN_xlow, SIN_xhigh
        
    def forward(self, x):
        return  self.task[0] * np.sin(x) 

    def sample(self, size):
        x = self.rngs.uniform(self.xlow, self.xhigh, size=(size,1) )
        y = self.forward(x)
        return np.hstack((x,y))

    def space(self, size):
        x = np.expand_dims(np.linspace(self.xlow, self.xhigh, num=size), axis=-1)
        y = self.forward(x)
        return np.hstack((x,y))

    def __str__(self) -> str:
        return str(self.task)
    def __repr__(self) -> str:
        return self.__str__()


class LINTASK:
    def __init__(self, task, seed=None):
        self.task = task
        self.rngs = np.random.default_rng(seed)
        self.xlow, self.xhigh =  SIN_xlow, SIN_xhigh
        
    def forward(self, x):
        return  self.task[0] * x + self.task[1]

    def sample(self, size):
        x = self.rngs.uniform(self.xlow, self.xhigh, size=(size,1) )
        y = self.forward(x)
        return np.hstack((x,y))

    def space(self, size):
        x = np.expand_dims(np.linspace(self.xlow, self.xhigh, num=size), axis=-1)
        y = self.forward(x)
        return np.hstack((x,y))

    def __str__(self) -> str:
        return str(self.task)

    def __repr__(self) -> str:
        return self.__str__()