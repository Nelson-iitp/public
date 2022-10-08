import torch, math, os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
            
            
def shape2size(shape):
    res = 1
    for d in shape:
        res*=d
    return res  


  
class MLP_NN:
    def tensor(self, data, rgrad=True):
        return torch.tensor(data, device=self.device, dtype=self.dtype, requires_grad=rgrad)

    def __init__(self, layer_dims, device, dtype, actF, init_range=(-0.1, 0.1), seed=None, from_param=False):
        self.device, self.dtype = device, dtype
        self.rng = np.random.default_rng(seed)
        self.init_low, self.init_high = init_range
        
        if from_param:
            self.parameters = layer_dims
            self.acts = []
            self.n_layers = 0

            for i in range(1, int(len(self.parameters)/2)):
                self.acts.append(actF())
                self.acts.append(None)
                self.n_layers +=1
        else:
            self.parameters = [] 
            self.acts = []
            self.n_layers = 0

            for i in range(1, len(layer_dims)):
                self.parameters.append( self.tensor(self.rng.uniform(self.init_low, self.init_high, size=(layer_dims[i], layer_dims[i-1]))) )
                self.parameters.append( self.tensor(self.rng.uniform(self.init_low, self.init_high, size=(layer_dims[i]))) )
                self.acts.append(actF())
                self.acts.append(None)
                self.n_layers +=1
        
    def __call__(self, x):
        z = x
        for l in range(0, len(self.parameters)-2, 2):
            z = self.acts[l]( self.parameters[l] @ z + self.parameters[l+1] ) 
        logits = self.parameters[-2] @ z + self.parameters[-1] 
        return logits

    def forward(self, X):
        Y= []
        for x in X:
            Y.append(self.__call__(x))
        return torch.stack(Y)

    def step_grad(self, lr):
        with torch.no_grad():
            for p in self.parameters:
                p -= lr*p.grad

    def zero_grad(self):
        with torch.no_grad():
            for p in self.parameters:
                p.grad = None
            
    def info(self, show_vals=False, P=print):
        P('--------------------------')
        P('~ N_LAYERS:[{}]\n~ D_TYPE:[{}]\n~ DEV:[{}]'.format(self.n_layers, self.dtype, self.device))
        P('--------------------------')
        total_params = 0
        for px,param in enumerate(self.parameters):
            nos_params = shape2size(param.shape)
            if px%2 == 0:
                P('--> Weights[{}]:: Params[{}] of Shape[{}]'.format(int(px/2), nos_params,  param.shape))
            else:
                P('--> Bias[{}]:: Params[{}] of Shape[{}]'.format(int(px/2), nos_params,  param.shape))
            if show_vals:
                P(' ~--> [PARAMETER TENSOR]:', param)
            #P(px, '\t', nos_params, '\t', param.shape )
            total_params+=nos_params
        P('--------------------------')
        P('PARAMS:\t', f'{total_params:,}') # 
        P('--------------------------')
        return total_params

    def copy_weights(self, M):
        with torch.no_grad():
            for to_layer, from_layer in zip(self.parameters, M.parameters):
                to_layer.data.copy_(from_layer.data)
        return

    def save_external(self, path):
        os.makedirs(path, exist_ok=True)
        with torch.no_grad():
            for l,layer in enumerate(self.parameters):
                np.save(os.path.join(path, str(l)+'.npy'), layer.data.numpy() )
        return
        
    def load_external(self, path):
        with torch.no_grad():
            for l,layer in enumerate(self.parameters):
                layer.data.copy_(self.tensor( np.load(os.path.join(path, str(l)+'.npy')),rgrad=False))
                layer.grad = None  
        #self.zero_grad()
        return

class Qnetn(nn.Module):
    def __init__(self, layer_dims, actF):
        super(Qnetn, self).__init__()
        state_dim, LL, action_dim = layer_dims[0], layer_dims[1:-1], layer_dims[-1]
        self.n_layers = len(LL)
        if self.n_layers<1:
            raise ValueError('need at least 1 layers')
        layers = [nn.Linear(state_dim, LL[0]), actF()]
        for i in range(self.n_layers-1):
            layers.append(nn.Linear(LL[i], LL[i+1]))
            layers.append(actF())
        layers.append(nn.Linear(LL[-1], action_dim))
        self.SEQL = nn.Sequential( *layers )
        self.n_layers+=1

    def forward(self, x):
        logits = self.SEQL(x)
        return logits

    def info(self, cap="", P=print):
        P('--------------------------')
        P(cap, 'No. Layers:\t', self.n_layers)
        P('--------------------------')
        # print(2*(pie.Q.n_layers+1)) <-- this many params
        std = self.state_dict()
        total_params = 0
        for param in std:
            nos_params = shape2size(std[param].shape)
            P(param, '\t', nos_params, '\t', std[param].shape )
            total_params+=nos_params
        P('--------------------------')
        P('PARAMS:\t', f'{total_params:,}') # 
        P('--------------------------')
        return total_params

