#-----------------------------------------------------------------------------------------------------
# relearn/core.py
#-----------------------------------------------------------------------------------------------------
import gym

#from numpy.random import default_rng
from .rl import ENV, SPACE, MEMORY

import torch as T
class cartpole(ENV):
    def __init__(self, device) -> None:
        spaces = {
            'S' : SPACE( (4,), dtype=T.float32),
            'A' : SPACE( (), dtype=T.int32),
            'R':  SPACE( (), dtype=T.float32),
            'D':  SPACE( (), dtype=T.int8),

        }
        super().__init__(device, spaces, buffer_as_attr=True)
        self.enable_snap( MEMORY('cpu', 10_000, spaces))
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