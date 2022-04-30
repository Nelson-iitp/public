#-----------------------------------------------------------------------------------------------------
# relearn/core.py
#-----------------------------------------------------------------------------------------------------
import gym
import gym.spaces

#from numpy.random import default_rng
from UEMEC.rl import ENV, SPACE, MEMORY

import torch as T
class cartpole(ENV):
    def __init__(self, device, cap) -> None:
        spaces = {
            'S' : SPACE( (4,), dtype=T.float32),
            'A' : SPACE( (), dtype=T.int32, low=0, high=2, discrete=True),
            'R':  SPACE( (), dtype=T.float32),
            'D':  SPACE( (), dtype=T.int8),

        }
        super().__init__(device, spaces, buffer_as_attr=True)
        self.enable_snap( MEMORY('cpu', cap, spaces))
        self.memory.seed(None)

    def init(self) -> None:
        self.gid = 'CartPole-v0'
        self.genv = gym.make(self.gid)

    def state(self):
        return self.S.flatten()

    def reset(self) -> bool:
        cs = self.genv.reset()
        self.S.copy_(T.tensor(cs))
        self.A.fill_(-1)
        self.R.fill_(0)
        self.D.fill_(0)
        return False

    def act(self, action) -> None:
        self.A.fill_(action)

    def step(self) -> bool:
        ns, r, d, _ = self.genv.step(self.A.item())
        self.S.copy_(T.tensor(ns))
        self.R.fill_(r)
        self.D.fill_(int(d))
        return d
#-----------------------------------------------------------------------------------------------------

import numpy as np
from UEMEC.basic import RCONV
class contienv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.gx, self.gy = 200, 100
        self.rfx = RCONV((-1,1), (0, self.gx))
        self.rfy = RCONV((-1,1), (0, self.gy))
        self.action_space = gym.spaces.Box(low=np.zeros(2)-1,high=np.zeros(2)+1)
        self.observation_space = gym.spaces.Box(low=np.zeros(4), high=np.array([self.gx, self.gy, self.gx, self.gy]))
        self.max_episode_steps = 1000
        self.S = np.zeros(2+2, dtype= np.float32) #<--- position of target, self position
        self.rng = np.random.default_rng(None)
        self.idis = 0

    def get_dis(self):
        return np.sum( (self.S[0:2] - self.S[2:4])**2 )**0.5

    def state(self):
        return self.S.flatten()

    def reset(self):
        self.S[0] = self.rng.integers(0, self.gx)
        self.S[1] = self.rng.integers(0, self.gy)
        self.S[2] = self.rng.integers(0, self.gx)
        self.S[3] = self.rng.integers(0, self.gy)
        self.idis = self.get_dis()
        self.steps = 0
        return self.state()

    def render(self):
        print('TARGE:[{}], SELF:[{}]'.format(self.S[0:2], self.S[2:4]))
    def step(self, action):
        
        self.S[0], self.S[1] = self.rfx.in2map(action[0]), self.rfy.in2map(action[1])

        idis = self.get_dis()
        reward = self.idis - idis
        self.idis = idis

        # move target randomly
        self.S[0:2] += 10*(self.rng.random(size=(2,)) - 0.5)

        self.steps+=1
        done = (self.steps>=self.max_episode_steps)
        return self.state(), reward, done, {}


    def pause(self):
        print('Done Training!')



"""
ENV class simulates an environment 

    self.device         : the torch device for buffers
    self.spaces         : a dict of <str> : <SPACE>
    self.buffers        : a dict of <str> : <tensor>   <--- a named buffer is a tensor from its corresponding space
    self.memory         : a MEMORY object that stores copies of buffers at each timestep
    

    > use self.enable_snap(memory) to enable auto-snapping to memory at each timestep

    > to simulate enironment use the self.start() and self.next() functions

    done = env.start()
    while not done:
        state = env.state()
        action = agent.predict(state)
        env.act(action)
        done = env.next() #<----
        reward = env.buffers['R'] #<--- which ever buffer is for the reward, 
        # can ready other buffer likewise for eg- timestep, cost, terminal signal, custom data etc

"""

""" ARCHIVE

            
            'POS_BSV': SPACE( (self.params.n_BSV, 3), dtype=tt.float32 ),
            'POS_UAV': SPACE( (self.params.n_UAV, 3), dtype=tt.float32 ),
            'BW_UAV_IOT': SPACE( (self.params.n_UAV, self.params.n_IOT), dtype=tt.float32 ),
            'BW_BSV_UAV': SPACE( (self.params.n_BSV, self.params.n_UAV), dtype=tt.float32 ),
            'OFF_LOC':  SPACE ( (self.params.n_IOT,) , dtype=tt.int8, low=0, high=2, discrete=True ),
            'OFF_CC':   SPACE ( (self.params.n_IOT,) , dtype=tt.float32)

            # check if b/w assigned is within range of transmission of UAV
            # BW_UAV = for each uav - which bsv its connected and how much b/w assigned
            bsv_bw_util = tt.zeros(self.params.n_BSV, dtype=tt.float32)
            for u in range(self.params.n_UAV):
                b, bw = int(self.BW_UAV[u,0].item()), self.BW_UAV[u,1].item()
                #print(' b, bw', b, bw)
                if b<0 or b>=self.params.n_BSV or bw<=0:
                    #raise StopIteration('invalid BSV [{}]'.format(b))
                    b=-1
                    bw=0
                    #raise StopIteration('In valid bandwidth assigned [{}]'.format(bw))

                if self.DUB[u, b] > self.uav[u].dtx: #<--- is within transmission distance of uav?
                    raise StopIteration( 'BSV [{}]m not in transmission range [{}]m of UAV - {},{}\n'.format(self.DUB[u, b], self.uav[u].dtx, b, u ))
                
                bsv_bw_util[b] += bw
                self.uav[u].parent[0] = b
                self.uav[u].parent_bw[0] = bw
            
            for b in range(self.params.n_BSV):
                if bsv_bw_util[b] > self.bsv[b].avail_bw[0]:
                    raise StopIteration('Bandwith Utiization Overflow on BSV[{}]'.format(b))

            # check if b/w assigned is within range of transmission IOT
            # BW_IOT = for each iot - which uav its connected and how much b/w assigned
            uav_bw_util = tt.zeros(self.params.n_UAV, dtype=tt.float32)
            for i in range(self.params.n_IOT):
                u, bw = int(self.BW_IOT[i,0].item()), self.BW_IOT[i,1].item()

                if u<0 or u>=self.params.n_UAV:
                    raise StopIteration('invalid UAV [{}]'.format(u))

                if bw<0: # if some bandwidth is assigned - check if within range
                    raise StopIteration('In valid bandwidth assigned [{}]'.format(bw))

                if self.DIU[i, u] > self.iot[i].dtx: #<--- is within transmission distance of uav?
                    raise StopIteration( 'UAV [{}]m not in transmission range [{}]m of IOT \n'.format(self.DIU[i, u], self.iot[i].dtx) )

                uav_bw_util[u] += bw
                self.iot[i].parent[0] = u
                self.iot[i].parent_bw[0] = bw

            for u in range(self.params.n_UAV):
                if uav_bw_util[u] > self.uav[u].avail_bw[0]:
                    raise StopIteration('Bandwith Utiization Overflow on UAV[{}]'.format(u))



            # check off ocations - OFF_LOC, OFF_CC
            bsv_cc_util = tt.zeros(self.params.n_BSV, dtype=tt.float32)
            uav_cc_util = tt.zeros(self.params.n_UAV, dtype=tt.float32)
            for i in range(self.params.n_IOT):
                offloc = self.OFF_LOC[i].item()
                offcc = self.OFF_CC[i].item()
                self.iot[i].off[0] = offloc
                self.iot[i].cc[0] = offcc
                uav = int(self.iot[i].parent[0])
                bsv = int(self.uav[uav].parent[0])
                if offloc==0: # 0 = offloading to uav
                    uav_cc_util[uav]+=offcc
                elif offloc==1 : # offloading to BSV
                    bsv_cc_util[bsv]+=offcc
                else:
                    raise StopIteration('Invalid offloading location [{}]'.format(offloc))

            for u in range(self.params.n_UAV):
                if uav_cc_util[u] > self.uav[u].avail_cc[0]:
                    raise StopIteration('Compute Utiization Overflow on UAV[{}]'.format(u))

            for b in range(self.params.n_BSV):
                if bsv_cc_util[b] > self.bsv[b].avail_cc[0]:
                    raise StopIteration('Compute Utiization Overflow on BSV[{}]'.format(b))
        except StopIteration as ex:
            print('Caught StopIteration', ex)
            done=True
        finally:
            reward = (0.0 if done else 1.0)
            self.REW.fill_(reward)








class DEVICE:
    def __init__(self, 
            locx = 0,
            locy = 0,
            locz = 0,
            parent = -1,
            parent_bw = 0,
            ptx = -1,
            dtx = 0,
            avail_cc=0,
            avail_bw=0,
            prx=-1,
            task_l=0,
            task_c=0,
            task_a=0,
            task_o=0,
            tout=-1,
            off=-1,
            dtype=tt.float32,
            device='cpu') -> None:
        self.tensor = tt.tensor( 
            [locx, locy, locz, 
            parent, parent_bw, ptx, dtx, 
            avail_cc, avail_bw, prx, 
            task_l, task_c, task_a, task_o, tout, off], dtype=dtype, device=device) # 16 size

        self.loc = self.tensor[0:3] # x,y,z +3
        self.parent = self.tensor[3:4]      # not connected - uplink parent +1
        self.parent_bw = self.tensor[4:5]      # not connected - bw assigned uplink network +1
        self.ptx = self.tensor[5:6]             # transmit power uplink network +1
        self.dtx = self.tensor[6:7]           # transmit distance (max) # reciver has to be in this range +1
        self.avail_cc = self.tensor[7:8]        # Hz available on this device +1
        self.avail_bw = self.tensor[8:9]        # total BW (Hz) available on downlink network +1
        self.prx = self.tensor[9:10]            # recieve power downlink network +1

        self.task = self.tensor[10:14]  # li ci Ai, oi,   - only for IOT device + 4
        self.off = self.tensor[14:15]  # offloading location which can be UAV, BSV or SAT + 1
        self.tout = self.tensor[15:16]  # location where output data is required - can be None, BSV or SAT + 1

        self.shape = self.tensor.shape
"""

""" ARCHIVE:

    def act_discrete(self, action):
        self.print('--> ENV::[act-discrete] @ [{}]'.format(action))
        assert(action>=0 and action<self.nA)
        self.A.fill_(int(action))

        # convert action which is discrete ---> uaction which is (4,)
        aF, a1, a2, a3 = self.AD['noop'], 0, 0, 0
        _debugs = ""
        pnIOT, pnUAV, pnBSV = \
            1 + 2*self.params.n_IOT, \
            1 + 2*self.params.n_IOT + 4*self.params.n_UAV, \
            1 + 2*self.params.n_IOT + 4*self.params.n_UAV + 4*self.params.n_BSV

        # if we start at S and have n-items, End is at E = S + (n-1) <<-- inclusive, excluse --> E = S + n
        if action: # action>0, otherwise noop


            if action < pnIOT: # is iot offloading
                _debugs+='auto_offload_IOT'
                aF = self.AD['auto_offload_IOT']

                pnlow = 1                
                a2 = int(float(action - pnlow) / self.params.n_IOT ) # 0 or 1
                assert(a2==0 or a2==1)
                a1 = (action-pnlow) - a2*self.params.n_IOT
                a3 = self.iot[int(a1)].cc_req()
                #baseA = 0
                #if action < 1 + 1*self.params.n_IOT:
                #    xa2 = 0
                #    baseA = 1 + 0*self.params.n_IOT
                #elif action < 1 + 2*self.params.n_IOT :
                #    xa2 = 1
                #    baseA = 1 + 1*self.params.n_IOT
                #else:
                #    raise StopIteration('Not possible')
                #xa1 = action - baseA  # which iot

                


            elif action < pnUAV: # is uav move action
                pnlow = pnIOT

                _debugs+='moveby_UAV'
                aF = self.AD['moveby_UAV']
                
                baseA = 0
                n = self.params.n_UAV
                if action < pnlow + 1*n:
                    a2,a3 = 0,self.fixed_move #pass # is north
                    baseA = pnlow + 0*n
                elif action < pnlow + 2*n:
                    a2,a3 = self.fixed_move,0 #pass # is East
                    baseA = pnlow + 1*n
                elif action < pnlow + 3*n:
                    a2,a3 = 0,-self.fixed_move #pass # is South
                    baseA = pnlow + 2*n
                elif action < pnlow + 4*n:
                    a2,a3 = -self.fixed_move,0 #pass # is West
                    baseA = pnlow + 3*n
                else:
                    raise StopIteration('Not possible')
                a1 = action - baseA  # which uav

            elif action < pnBSV: # is bsv move action
                pnlow = pnUAV
                aF = self.AD['moveby_BSV']
                _debugs+='moveby_BSV'
                baseA = 0
                n = self.params.n_BSV
                if action < pnlow + 1*n:
                    a2,a3 = 0, self.fixed_move #pass # is north
                    baseA = pnlow + 0*n
                elif action < pnlow + 2*n:
                    a2,a3 = self.fixed_move,0 #pass # is East
                    baseA = pnlow + 1*n
                elif action < pnlow + 3*n:
                    a2,a3 = 0,-self.fixed_move #pass # is South
                    baseA = pnlow + 2*n
                elif action < pnlow + 4*n:
                    a2,a3 = -self.fixed_move,0 #pass # is West
                    baseA = pnlow + 3*n
                else:
                    raise StopIteration('Not possible')
                a1 = action - baseA  # which bsv
            else: # is invalid action
                raise StopIteration('Invalid action: [{}]'.format(action))
        # copy to action vector
        _debugs+=(' --> [{}], [{}], [{}], [{}]'.format(aF, a1, a2, a3))
        #print(_debugs)
        
        for i,v in enumerate([aF, a1, a2, a3]):
            self.ACT[i].fill_(v)
        #self.ACT.copy_(tt.tensor([aF, a1, a2, a3], dtype=tt.float32))
        return _debugs

#    def _act(self, action):
#        self.print('--> ENV::[act] @ [{}]'.format(action))
#        assert(action[0]>=0 and action[0]<len(self.ACTIONS))
#        self.ACT.copy_(action)

    def step_discrete(self):
        self.print('--> ENV::[step_begin]')
        # execute action
        status = self.ACTIONS[int(self.ACT[0])]( self.ACT[1].item(), self.ACT[2].item(), self.ACT[3].item()) 
        self.D.fill_((0 if status else 1))
        
        # calculate reward        
        #current_cost = self.get_cost()
        #reward =   current_cost - self.COST.item()
        #self.R.fill_(reward)
        reward = self.R.item()
        self.COST += self.R

        self.STEP+=1 # increment step
        done = (self.D.item()>0)
        self.print('<--- ENV::[step_end], REW:[{}], DONE:[{}]: TotalREW:[{}]'.format(reward, done, self.COST.item()))
        return done


"""