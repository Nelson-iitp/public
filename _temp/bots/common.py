import os
from io import BytesIO
import numpy as np
import torch as tt
import torch.nn as nn
import torch.distributions as td
import gym.spaces

observation_key,    action_key,     reward_key,     done_key,    step_key    = \
'state',            'action',       'reward',       'done',       'step'

def default_spaces(observation_space, action_space):    
    # set default spaces - this is directly used by other components like explorer, memory and nets
    return {
        observation_key:    observation_space,
        action_key:         action_space,
        reward_key:         gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
        done_key:           gym.spaces.Box(low=0, high=1, shape=(), dtype=np.int8),
        step_key:           gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),
    }

class MEM:
    """ [MEM] - A key based static replay memory """

    def __init__(self, observation_space, action_space, capacity, seed) -> None:
        """ named_spaces is a dict like string_name vs gym.space """
        assert(capacity>0)
        self.capacity = capacity+2
        self.spaces = default_spaces(observation_space, action_space)
        # why add2 to capacity -> one transition requires 2 slots and we need to put an extra slot for current pointer
        self.rng = np.random.default_rng(seed)
        self.build_memory()
        
    def build_memory(self):
        self.data={}
        for key, space in self.spaces.items():    
            if key!='':
                self.data[key] = np.zeros((self.capacity,) + space.shape, space.dtype)
        self.ranger = np.arange(0, self.capacity, 1)
        self.mask = np.zeros(self.capacity, dtype=np.bool8) #<--- should not be changed yet
        self.keys = self.data.keys()
        self.clear()

    def clear(self):
        self.at_max, self.ptr = False, 0
        self.mask*=False

    def length(self): # Excludes the initial observation (true mask only)
        return len(self.ranger[self.mask])

    def count(self): # Includes the initial observation (any mask)
        return self.capacity if self.at_max else self.ptr
    
    def snap_info(self, mask, **info):
        """ snaps all keys in self.keys - assuming info has the keys already """
        for k in self.keys:
            self.data[k][self.ptr] = info[k]
        self.mask[self.ptr] = mask
        self.ptr+=1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0
        self.mask[self.ptr] = False
        return

    def snap(self, mask, observation, action, reward, done, step):
        self.data[observation_key][self.ptr] = observation
        self.data[action_key][self.ptr] = action
        self.data[reward_key][self.ptr] = reward
        self.data[done_key][self.ptr] = done
        self.data[step_key][self.ptr] = step
        
        self.mask[self.ptr] = mask
        self.ptr+=1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0
        self.mask[self.ptr] = False
        return

    """ NOTE: Sampling

        > sample_methods will only return indices, use self.read(i) to read actual tensors
        > Valid Indices - indices which can be choosen from, indicates which transitions should be considered for sampling
            valid_indices = lambda : self.ranger[self.mask]
    """    

    def sample_recent(self, size):
        self.mask[self.ptr] = True
        valid_indices = self.ranger[self.mask]
        self.mask[self.ptr] = False
        iptr = np.where(valid_indices==self.ptr)[0] # find index of self.ptr in si
        pick = min ( len(valid_indices)-1, size )
        return valid_indices[ np.arange(iptr-pick, iptr, 1) ]

    def sample_recent_(self, size):
        self.mask[self.ptr] = True
        valid_indices = self.ranger[self.mask]
        self.mask[self.ptr] = False
        iptr = np.where(valid_indices==self.ptr)[0] # find index of self.ptr in si
        pick = min ( len(valid_indices)-1, size )
        return pick, valid_indices[ np.arange(iptr-pick, iptr, 1) ]

    def sample_all_(self):
        return self.sample_recent_(self.length())

    def sample_all(self):
        return self.sample_recent(self.length())

    def sample_random(self, size, replace=False):
        valid_indices = self.ranger[self.mask]
        pick = min ( len(valid_indices), size )
        return self.rng.choice(valid_indices, pick, replace=replace)

    def sample_random_(self, size, replace=False):
        valid_indices = self.ranger[self.mask]
        pick = min ( len(valid_indices), size )
        return pick, self.rng.choice(valid_indices, pick, replace=replace)

    def read(self, i): # reads [all keys] at [given sample] indices
        return { key:self.data[key][i] for key in self.keys }

    def readkeys(self, i, keys): # reads [given keys] at [given sample] indices
        return { key:self.data[key][i] for key in keys }

    def readkeis(self, ii, keys, teys): # reads [given keys] at [given sample] indices and rename as [given teys]

        return { t:self.data[k][i%self.capacity] for i,k,t in zip(ii,keys,teys) }

    def readkeist(self, *args): # same as 'readkeis' but the args are tuples like: (index, key, tey)
        return { t:self.data[k][i] for i,k,t in args }
                        
        
    """ NOTE: Rendering """
    def render(self, low, high, step=1, p=print):
        p('=-=-=-=-==-=-=-=-=@[MEMORY]=-=-=-=-==-=-=-=-=')
        p("Length:[{}]\tCount:[{}]\nCapacity:[{}]\tPointer:[{}]".format(self.length(), self.count(), self.capacity, self.ptr))
        for i in range (low, high, step):
            p('____________________________________')  #p_arrow=('<--------[PTR]' if i==self.ptr else "")
            if self.mask[i]:
                p('SLOT: [{}]+'.format(i))
            else:
                p('SLOT: [{}]-'.format(i))
            for key in self.data:
                p('\t{}: {}'.format(key, self.data[key][i]))
        p('=-=-=-=-==-=-=-=-=![MEMORY]=-=-=-=-==-=-=-=-=')

    def render_all(self, p=print):
        self.render(0, self.count(), p=p)

    def render_last(self, nos, p=print):
        self.render(-1, -nos-1, step=-1,  p=p)

class REMAP:
    def __init__(self, Input_Range, Mapped_Range) -> None:
        self.input_range(Input_Range)
        self.mapped_range(Mapped_Range)

    def input_range(self, Input_Range):
        self.Li, self.Hi = Input_Range
        self.Di = self.Hi - self.Li
    def mapped_range(self, Mapped_Range):
        self.Lm, self.Hm = Mapped_Range
        self.Dm = self.Hm - self.Lm
    def map2in(self, m):
        return ((m-self.Lm)*self.Di/self.Dm) + self.Li
    def in2map(self, i):
        return ((i-self.Li)*self.Dm/self.Di) + self.Lm

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [D] [torch.nn] 
    Some basic Neural Net models and helpers functions using torch.nn """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_models(dir_name, file_names, models):
    os.makedirs(dir_name, exist_ok=True)
    for θ, f in zip(models, file_names):
        tt.save(θ, os.path.join(dir_name, f))

def load_models(dir_name, file_names):
    return tuple( [tt.load(os.path.join(dir_name, f)) for f in file_names ])

def save_model(path, model):
    tt.save(model, path)

def load_model(path):
    return tt.load(path)

def clone_model(model, detach=False):
    """ use detach=True to sets the 'requires_grad' to 'False' on all of the parameters of the cloned model. """
    buffer = BytesIO()
    tt.save(model, buffer)
    buffer.seek(0)
    model_copy = tt.load(buffer)
    if detach:
        for p in model_copy.parameters():
            p.requires_grad=False
    model_copy.eval()
    del buffer
    return model_copy

def build_sequential(in_dim, layer_dims, out_dim, actF, actL ):
    layers = [nn.Linear(in_dim, layer_dims[0]), actF()]
    for i in range(len(layer_dims)-1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        layers.append(actF())
    layers.append(nn.Linear(layer_dims[-1], out_dim))
    _ = None if actL is None else layers.append(actL())
    return nn.Sequential( *layers )

class MLP(nn.Module):
    """ Multi layer Perceptron based parameterized networks for policy and value networks """
    def __init__(self, in_dim, layer_dims, out_dim, actF, actL):
        super(MLP, self).__init__()
        self.net = build_sequential(in_dim, layer_dims, out_dim, actF, actL )
    def forward(self, x):
        return self.net(x)

class DLP(nn.Module):
    """ Decoupled Multi layer Perceptron for dueling-DQN architecture """
    def __init__(self, 
    in_dim, 
    layer_dims_net, 
    out_dim_net,
    actF_net, 
    actL_net,
    layer_dims_vnet, 
    actF_vnet, actL_vnet, 
    layer_dims_anet, 
    actF_anet, 
    actL_anet, 
    out_dim):
        super(DLP, self).__init__()
        self.net = build_sequential(in_dim, layer_dims_net, out_dim_net, actF_net, actL_net )
        self.vnet = build_sequential(out_dim_net, layer_dims_vnet, 1, actF_vnet, actL_vnet)
        self.anet = build_sequential(out_dim_net, layer_dims_anet, out_dim, actF_anet, actL_anet)
    def forward(self, x):
        net_out = self.net(x)
        v_out = self.vnet(net_out)
        a_out = self.anet(net_out)
        return v_out + (a_out -  tt.mean(a_out, dim = -1, keepdim=True))

class MLP2(nn.Module):
    """ Multi layer Perceptron based parameterized networks for policy and value networks """
    def __init__(self, in_dim_s, in_dim_a, layer_dims, out_dim, actF, actL):
        super(MLP2, self).__init__()
        self.net = build_sequential(in_dim_s+in_dim_a, layer_dims, out_dim, actF, actL )
    def forward(self, xs, xa):
        return self.net(tt.concat((xs,xa), dim=-1))

