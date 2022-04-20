import torch
import torch.nn as nn
from numpy.random import default_rng

#-----------------------------------------------------------------------------------------------------
class SPACE:
    """ 
    [SPACE] - represents a vector space 

    * SPACE object has following attributes:
        [.] shape:      dimension (tuple)
        [.] dtype:      data-type (torch.dtype)
        [.] low:        lower bound - can be a scalar or an array (broadcast rules apply)
        [.] high:       upper bound - can be a scalar or an array (broadcast rules apply)
        [.] zero:       a default zero value - can be a scalar or an array (broadcast rules apply)
        [.] discrete    if True, indicates that all dimensions take discrete integer values
        [.] ndim        dimensions
        [.] scalar      True if 0-dimensions
        [.] nflat       total number of elements(scalar) in a flattened version - a product of all dimensions
    
    * SPACE object has following functions:

        (.) zeros(device):-> Tensor             : returns a zero tensor from this space
        (.) zeron(n, device):-> Tensor          : returns a stack of 'n' zero tensor from this space
        (.) enum(flat, device):-> int, Tensor   : enumerates space and returns 2-tuple (count, elements)
        (.) sample(rng, device):-> Tensor       : sample uniformly from given space using numpy-like rng 

    """
    def __init__(self, shape, dtype, low=0, high=0, zero=0, discrete=False):
        self.shape , self.dtype, self.low, self.high, self.discrete, self.zero = \
             shape,       dtype,      low,      high,      discrete,      zero
        self.ndim = len(shape) 
        self.scalar = (self.ndim == 0)
        self.nflat = torch.prod(torch.tensor(shape),0).item()
    def zeros(self, device='cpu'):
        return torch.zeros(size=self.shape, dtype=self.dtype, device=device) + self.zero
    def zeron(self, n, device='cpu'):
        return torch.zeros(size=(n,)+self.shape, dtype=self.dtype, device=device)+ self.zero
    def __str__(self):
        return '[SPACE] : shape:[{}], dtype:[{}], discrete[{}], ndim:[{}], nflat:[{}]'.format(
                self.shape, self.dtype, self.discrete, self.ndim, self.nflat)
    def __repr__(self):
        return self.__str__()
    def enum(self, flat=False, device='cpu'):
        """ enumerates the space and returns all possible vectors 

            returns 2-tuple
                count:      number of elements
                elements:   a tensor containing all elements
        """
        assert(self.discrete) #<-- this should be true
        low, high = self.zeros(device) , self.zeros(device) 
        low += torch.tensor(self.low, dtype=self.dtype, device=device)
        high += torch.tensor(self.high, dtype=self.dtype, device=device)
        #list(itertools.product(a, b))
        flbase, fhbase = low.flatten(), high.flatten()
        fbase = fhbase - flbase
        #assert(len(fbase)==self.nflat)
        count = torch.prod(fbase,0).item()
        ranges = [ torch.arange(flbase[i], fhbase[i], 1, device=device) for i in range(self.nflat) ]
        elements = torch.cartesian_prod(*ranges)
        #assert (count == len(elements))
        return  count, elements if flat else elements.reshape((count,) + self.shape)
    def sample(self, rng, device='cpu'):
        """ sample uniformly from given space using numpy-like rng """
        return  torch.tensor(rng.integers(self.low, self.high, size=self.shape), dtype=self.dtype, device=device) \
                if self.discrete else \
                torch.tensor(rng.uniform(self.low, self.high, size=self.shape), dtype=self.dtype, device=device)

class ENV:
    """
    [ENV] - represents an environment simulator with buffers

    * ENV object has following attributes:
        [.] device:     torch device for buffer tensors
        [.] spaces:     a dictionary of named spaces  (name<str> v/s space<SPACE>) - each space has its own buffer
        [.] buffers:    a dictionary of named buffers (name<str> v/s buffer<Tensor>) - each space has its own buffer
        [.] done:       boolean flag indicating end of episode
    
    * ENV object has following built-in functions that must not be defined on inheritance:
        (.) build_buffer_no_attr()
        (.) build_buffer_with_attr()
        (.) start():-> bool
        (.) next():-> bool

    * ENV object has following functions that must be defined on inheritance:
        (.) init():-> None         : called once at ENV.__init__()
        (.) reset():-> bool         : resets the buffers, returns done flag
        (.) state():-> Any          : returns a agent state
        (.) act(action):-> None     : copy the 'action' provided by an agent to appropiate buffer
        (.) step(): -> bool         : 'state', 'action' are available in buffers, step to next state and return done flag
    """
    def __init__(self, device, spaces, buffer_as_attr=False) -> None:
        """
        Args:
            device          : torch device for buffer
            spaces          : a dictionary of spaces (name<str> v/s space<SPACE>)
            buffer_as_attr  : bool, if True, adds the buffer keys as attributes on self using setattr
        """
        self.device = device
        self.spaces = spaces
        self.buffers={}
        self.disable_snap(True)
        self.build_buffer_with_attr() if buffer_as_attr else self.build_buffer_no_attr()
        self.init()
    def build_buffer_no_attr(self):
        for key, val in self.spaces.items():    
            if key!='':
                self.buffers[key] = val.zeros(self.device)
            else:
                print('Warning: cannot create buffer with name [{}]! Use another name.'.format(key))
    def build_buffer_with_attr(self):
        for key, val in self.spaces.items():    
            if not (hasattr(self, key) or key==''):
                setattr(self, key, val.zeros(self.device))
                self.buffers[key] = getattr(self, key)
            else:
                print('Warning: cannot create buffer with name [{}]! Use another name.'.format(key))

    def enable_snap(self, memory):
        self.memory = memory
        self.snap = lambda mask: self.memory.snap(mask, self.buffers)
    def disable_snap(self, reset_memory=False):
        self.snap = lambda mask: None
        if reset_memory:
            self.memory=None


    # for SELF-SIMULATED Environment -------------------------------
    def start(self):
        """ start a new episode or trajectory """
        done = self.reset()
        self.snap(False)
        return done
    def next(self):
        done = self.step()
        self.snap(True)
        return done

    def init(self) -> None:
        print ('init ~ not implemented ~')
    def reset(self) -> bool:
        print ('reset ~ not implemented ~')
        return False
    def state(self):
        print ('state ~ not implemented ~')
        return None
    def act(self, action) -> None:
        print ('act ~ not implemented ~')
    def step(self) -> bool:
        print ('step ~ not implemented ~')
        return True

class MEMORY:
    """
        [MEMORY] - A key based replay memory
    """
    def __init__(self, device, capacity, spaces, memory_as_attr=False) -> None:
        self.device, self.capacity, self.spaces = device, capacity, spaces
        self.data={}
        self.build_memory_with_attr() if memory_as_attr else self.build_memory_no_attr()
        self.ranger = torch.arange(0, self.capacity, 1, dtype=torch.long)
        self.mask = torch.zeros(self.capacity, dtype=torch.bool) #<--- should not be changed yet
        self.clear()

    def build_memory_no_attr(self):
        for key, val in self.spaces.items():    
            if key!='':
                self.data[key] = val.zeron(self.capacity, self.device)
    def build_memory_with_attr(self):
        for key, val in self.spaces.items():    
            if not (hasattr(self, key) or key==''):
                setattr(self, key, val.zeron(self.capacity, self.device))
                self.data[key] = getattr(self, key)
            else:
                print('Warning: cannot create buffer with name [{}]! Use another name.'.format(key))

    def __call__(self, key):
        return self.data[key]
    def clear(self):
        self.at_max, self.ptr = False, 0
        self.mask.fill_(False)
    def count(self):
        return self.capacity if self.at_max else self.ptr
    
    def snap(self, mask, data):
        """ snaps all the keys that are self.data, assumes that buffer already has those keys """
        for key in self.data:
            self.data[key][self.ptr].copy_(data[key])
        self.mask[self.ptr] = mask
        self.ptr+=1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0
        self.mask[self.ptr] = False
        return None
    def snapkeys(self, mask, data, keys):
        """ snaps a subset of keys in self.data, assumes that buffer already has those keys """
        for key in keys:
            self.data[key][self.ptr].copy_(data[key])
        self.mask[self.ptr] = mask
        self.ptr+=1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0
        self.mask[self.ptr] = False
        return None
    def snapdata(self, mask, data, keys):
        """ snaps a subset of keys in self.data, assumes that buffer already has those keys """
        for i,key in enumerate(keys):
            self.data[key][self.ptr].copy_(data[i])
        self.mask[self.ptr] = mask
        self.ptr+=1
        if self.ptr == self.capacity:
            self.at_max, self.ptr = True, 0
        self.mask[self.ptr] = False
        return None

    def read(self, i): 
        return { key:self.data[key][i] for key in self.data }
    def readkeys(self, i, keys): 
        return { key:self.data[key][i] for key in keys }
    def readkeis(self, ii, keys, teys): 
        return { t:self.data[key][i] for i,t,key in zip(ii,teys,keys) }
    """ sampling

        > sample_methods will only return indices, use self.read(i) to read actual tensors
        > Valid Indices - indices which can be choosen from, indicates which transitions should be considered for sampling
            valid_indices = lambda : self.ranger[self.mask]
    """    
    def seed(self, x=None):
        self.rng = default_rng(x) # a default rng for reproducability, required by self.sample_random()
    def sample_last(self):
        iptr = self.ptr
        count = self.count()
        if count<2:
            raise Exception('not enough transitions')

        max_reverse = self.ptr - count
        while(not self.mask[iptr] and iptr>max_reverse):
            iptr -= 1
        
        if not self.mask[iptr]:
            raise Exception('no transitions found')
        samples = torch.tensor(iptr, dtype=torch.long)
        #samples = torch.arange(self.ptr - min( self.count(), size ), self.ptr, 1, device=self.memory_device)
        return samples
    def sample_recent(self, size):
        self.mask[self.ptr] = True
        valid_indices = self.ranger[self.mask]
        self.mask[self.ptr] = False
        iptr = torch.where(valid_indices==self.ptr)[0].item() # find index of self.ptr in si
        pick = min ( len(valid_indices)-1, size )
        return valid_indices[ torch.arange(iptr-pick, iptr, 1, dtype=torch.long) ]
    def sample_random(self, size, replace=False):
        valid_indices = self.ranger[self.mask]
        pick = min ( len(valid_indices), size )
        return torch.tensor(self.rng.choice(valid_indices, pick, replace=replace), dtype=torch.long)

    def render(self, low, high, step=1, p=print):
        p('=-=-=-=-==-=-=-=-=@[MEMORY]=-=-=-=-==-=-=-=-=')
        p("Device [{}]\tCount [{}]\nCapacity[{}]\tPointer [{}]".format(self.device, self.count(), self.capacity, self.ptr))
        for i in range (low, high, step):
            p('____________________________________')
            print('SLOT: [{}]\t.{}.'.format(i, self.mask[i].item()))
            for key in self.data:
                p('\t{}: [{}]'.format(key, self.data[key][i]))
        p('=-=-=-=-==-=-=-=-=![MEMORY]=-=-=-=-==-=-=-=-=')
    def render_all(self, p=print):
        self.render(0, self.count(), p=p)
    def render_last(self, nos, p=print):
        self.render(-1, -nos-1, step=-1,  p=p)


#-----------------------------------------------------------------------------------------------------
class QnetnRelu(nn.Module):
    def __init__(self, state_dim, LL, action_dim):
        super(QnetnRelu, self).__init__()
        self.n_layers = len(LL)
        if self.n_layers<1:
            raise ValueError('need at least 1 layers')
        layers = [nn.Linear(state_dim, LL[0]), nn.ReLU()]
        for i in range(self.n_layers-1):
            layers.append(nn.Linear(LL[i], LL[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(LL[-1], action_dim))
        self.SEQL = nn.Sequential( *layers )

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
            nos_params = self.shape2size(std[param].shape)
            P(param, '\t', nos_params, '\t', std[param].shape )
            total_params+=nos_params
        P('--------------------------')
        P('PARAMS:\t', f'{total_params:,}') # 
        P('--------------------------')
        return total_params
    def shape2size(self, shape):
        res = 1
        for d in shape:
            res*=d
        return res  

class QnetnTanh(nn.Module):
    def __init__(self, state_dim, LL, action_dim):
        super(QnetnTanh, self).__init__()
        self.n_layers = len(LL)
        if self.n_layers<1:
            raise ValueError('need at least 1 layers')
        layers = [nn.Linear(state_dim, LL[0]), nn.Tanh()]
        for i in range(self.n_layers-1):
            layers.append(nn.Linear(LL[i], LL[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(LL[-1], action_dim))
        self.SEQL = nn.Sequential( *layers )

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
            nos_params = self.shape2size(std[param].shape)
            P(param, '\t', nos_params, '\t', std[param].shape )
            total_params+=nos_params
        P('--------------------------')
        P('PARAMS:\t', f'{total_params:,}') # 
        P('--------------------------')
        return total_params
    def shape2size(self, shape):
        res = 1
        for d in shape:
            res*=d
        return res  

#-----------------------------------------------------------------------------------------------------
class DQN:
    def __init__(self, state_dim, LL, action_dim, opt, cost, lr, double=False, tuf=0,  device='cpu', dtype=torch.float32, relu_act=True): 
        """ 
        Implements DQN based Policy - Requires Spaces on ENV : S, A, R, D
        
        state_dim       Observation Space Shape
        LL              List of layer sizes for eg. LL=[32,16,8]
        action_dim      Action Space (should be discrete)
        opt             torch.optim     (eg - torch.optim.Adam)
        cost            torch.nn.<loss> (eg - torch.nn.MSELoss)
        lr              Learning Rate for DQN Optimizer ()
        dis             discount factor
        tuf             target update frequency (if tuf==0 then doesnt use target network)
        double          uses double DQN algorithm (with target network)
        device          can be 'cuda' or 'cpu'  # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Note:
            # single DQN can either be trained with or without target
            # if self.tuf > 0 then it means target T exists and need to updated, otherwise T is same as Q
            # Note that self.T = self.Q if self.tuf<=0
        """
        
        if double and tuf<=0:
            raise ValueError("double DQN requires a target network, set self.tuf>0")
        self.lr = lr
        self.state_dim=state_dim
        self.LL = LL
        self.action_dim=action_dim
        #self.rand = np.random.default_rng(seed)
        self.tuf = tuf
        self.double=double

        self.device = device
        self.dtype = dtype
        self.opt=opt
        self.cost=cost
        
        if relu_act:
            Qnetn = QnetnRelu
        else:
            Qnetn = QnetnTanh
        self.base_model = Qnetn(state_dim, LL, action_dim).to(dtype=self.dtype, device = self.device)
        self.Q = Qnetn(state_dim, LL, action_dim).to(dtype=self.dtype, device = self.device)
        self.T = Qnetn(state_dim, LL, action_dim).to(dtype=self.dtype, device = self.device) if (self.tuf>0) else self.Q
        self.clear()

    def clear(self):
        self._clearQ()
        self.optimizer = self.opt(self.Q.parameters(), lr=self.lr) # opt = optim.Adam
        self.loss_fn = self.cost()  # cost=nn.MSELoss()
        self.train_count=0
        self.update_count=0
    def _clearQ(self):
        with torch.no_grad():
            self.Q.load_state_dict(self.base_model.state_dict())
            self.Q.eval()
            if (self.tuf>0):
                self.T.load_state_dict(self.base_model.state_dict())
                self.T.eval()
    def _loadQ(self, from_dqn):
        with torch.no_grad():
            self.Q.load_state_dict(from_dqn.Q.state_dict())
            self.Q.eval()

    def predict(self, state):
        qvals = self.Q(state)
        m,i =  torch.max(  qvals , dim=0  )
        return i.item()

    def seed_eps(self, seed, eps):
        self.eps_rng = default_rng(seed)

    def predict_eps(self, state, eps):
        if self.eps_rng.random()<eps:
            return self.eps_rng.integers(0, self.action_dim)
        else:
            qvals = self.Q(state)
            m,i =  torch.max(  qvals , dim=0  )
            return i.item()

    def _prepare_batch(self, memory, batch_size):
        #batch = memory.sample(size)
        indices = memory.sample_random(size=batch_size, replace=False)
        batch = memory.readkeis((indices-1, indices, indices, indices, indices), ('S', 'S', 'A', 'R', 'D'), ('cS', 'nS', 'A', 'R', 'D'))
        return  batch_size, \
                batch['cS'].to(dtype=self.dtype, device = self.device), \
                batch['nS'].to(dtype=self.dtype, device = self.device), \
                batch['A'], \
                batch['R'].to(dtype=self.dtype, device = self.device), \
                batch['D'].to(dtype=self.dtype, device = self.device)
        
    def learn(self, memory, batch, lr=0.0, dis=1.0):
        """ 
        Args:
            memory      ENV.memory - (memory should be seeded first)
            batch       int - batch size
            lr          float - learning rate (if lr=0.0, then does not change lr) 
            dis         float - discount factor
            """
        if lr:
            self.optimizer.param_groups[0]['lr']=lr
        steps, cS, nS, act, reward, done  = self._prepare_batch(memory, batch)
        indices = torch.arange(0, steps, 1, dtype=torch.long)
        #print('______', indices.dtype, indices )
        target_val = self.T(nS) #if type(target_pie)==type(None) else torch.tensor(target_pie.QVT(nSnp), dtype=torch.float32)
        #print('target_val', target_val.shape, target_val.dtype)
        if not self.double:
            updater, _ = torch.max(target_val, dim=1)
        else:            
            _, target_next = torch.max(self.Q(nS), dim=1) # tensor.max returns indices as well
            updater=torch.zeros(steps,dtype=torch.float32)
            updater[indices] = target_val[indices, (target_next[indices]).to(dtype=torch.long)]
        updated_q_values = reward + dis * updater * (1 - done)
        
        # Compute prediction and loss
        pred = self.Q(cS)
        target = pred.detach().clone()
        target[indices, (act[indices]).to(dtype=torch.long)] = updated_q_values[indices]
        loss =  self.loss_fn(pred, target)  #torch.tensor()
        self.optimizer.zero_grad()

        #if reg_l2_lambda>0: # adding L2 regularization
        #    l2_lambda = reg_l2_lambda
        #    l2_norm = sum(p.pow(2.0).sum() for p in self.Q.parameters())
        #    loss = loss + l2_lambda * l2_norm
                        
        # this does not happen
        #target[indices, act[indices]] = updated_q_values[indices]*(self.theta) + target[indices, act[indices]]*(1-self.theta)
        #if lr>0.0:
        #    with torch.no_grad():
        #        for g in self.optimizer.param_groups:
         #           g['lr'] = lr               
        # Backpropagation

        #for param in self.Q.parameters():
        #    param.grad.data.clamp_(-1, 1)  # clip norm <-- dont do it
        
        #if do_step:
        #grads=None
        loss.backward()
        self.optimizer.step()
        self.train_count+=1
        #else:
           # grads = torch.autograd.grad(loss, self.Q.parameters(), create_graph=True)
            #grads=[ param.grad.data.detach().clone() for param in self.Q.parameters() ]
            #self.optimizer.zero_grad()

        if (self.tuf>0):
            if self.train_count % self.tuf == 0:
                self.update_target()
                self.update_count+=1
        return loss.item() #grads

    def update_target(self):
        with torch.no_grad():
            self.T.load_state_dict(self.Q.state_dict())
            self.T.eval()

    def render(self, mode=0, P=print):
        P('=-=-=-=-==-=-=-=-=\nQ-NET\n=-=-=-=-==-=-=-=-=')
        P(self.Q)
        if mode>0:
            self.Q.info(cap='[LAYER INFO]',P=P)
        #P('\nPARAMETERS\n')
        #Q = self.Q.parameters()
        #for i,q in enumerate(Q):
        #    print(i, q.shape)
        P('Train Count:', self.train_count)
        if (self.tuf>0):
            P('Update Count:', self.update_count)
        P('=-=-=-=-==-=-=-=-=!Q-net=-=-=-=-==-=-=-=-=')
        return
        
    def save_external(self, filename):
        torch.save(self.Q, filename)

    def load_external(self, filename):
        self.base_model = torch.load(filename)
        self._clearQ()
        return
#-----------------------------------------------------------------------------------------------------
# Foot-Note:
"""
NOTE:
    * Author:           Nelson.S
"""