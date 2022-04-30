# On Demand UEC (UAV-enabled Edge Computing) 

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

from numpy.random import default_rng
import numpy as np
from .known.basic import RCONV

import gym
import gym.spaces

class BSV: 
    """ Base Station Vehicle
    
        > BSV has highest computation resources
        > BSV can connect to multiple UAVs - in such case, UAV can Tx and BSV can Rx 
        > BSV does not transmit anything
        > BSV to UAV links have more bandwidth that UAV to IOT links
    """
    dims = 6
    def __init__(self, tensor, avail_cc, avail_bw, prx) -> None:
        self.tensor = tensor 

        # view
        self.loc = self.tensor[0:3]             # x,y,z 
        self.avail_cc = self.tensor[3:4]        # Hz available compute on this device 
        self.avail_bw = self.tensor[4:5]        # total BW (Hz) available on downlink network to UAVs
        self.prx = self.tensor[5:6]             # Rx Power

        # init
        self.reset_arg = (avail_cc, avail_bw, prx)
        self.reset()

    def reset(self):
        self.avail_cc[0], self.avail_bw[0], self.prx[0] =  self.reset_arg


    def set_location(self, x, y, z):
        self.loc[0], self.loc[1], self.loc[2] = x, y, z
    def set_location2(self, x, y):
        self.loc[0], self.loc[1] = x, y

    def move_location(self, x, y, z):
        self.loc[0]+=x
        self.loc[1]+=y 
        self.loc[2]+=z
    def move_location2(self, x, y):
        self.loc[0]+=x
        self.loc[1]+=y

    def __str__(self) -> str:
        return '[BSV]:: LOC:[{}], ACC:[{}], ABW:[{}], PRX:[{}]'.format(self.loc, self.avail_cc, self.avail_bw, self.prx)        
    def __repr__(self) -> str:
        return self.__str__()

class UAV:
    """ Unmanned Ariel Vehicle
    
        > UAV has moderate computation resources
        > UAV can connect to multiple IOTs - in such case, IOT can Tx and UAV can Rx 
        > UAV can connect to a limited number of IOTs
        > UAV can connect to a single BSV
    """
    dims = 10
    def __init__(self, tensor, ptx, dtx, avail_cc, avail_bw, prx) -> None:
        self.tensor = tensor

        # view
        self.loc = self.tensor[0:3]             # x,y,z 
        self.avail_cc = self.tensor[3:4]        # Hz available compute on this device 
        self.avail_bw = self.tensor[4:5]        # total BW (Hz) available on downlink network to UAVs
        self.prx = self.tensor[5:6]             # Rx Power

        self.parent = self.tensor[6:7]          # not connected - uplink parent +1
        self.parent_bw = self.tensor[7:8]       # not connected - bw assigned uplink network +1
        self.ptx = self.tensor[8:9]             # transmit power uplink network +1
        self.dtx = self.tensor[9:10]             # transmit distance (max) # reciver has to be in this range +1

        # init
        self.reset_arg = (avail_cc, avail_bw, prx, ptx, dtx, -1, 0)
        self.reset()

    def reset(self):        
        self.avail_cc[0], self.avail_bw[0], self.prx[0], self.ptx[0], self.dtx[0], self.parent[0], self.parent_bw[0] = self.reset_arg

    def set_location(self, x, y, z):
        self.loc[0], self.loc[1], self.loc[2] = x, y, z
    def set_location2(self, x, y):
        self.loc[0], self.loc[1] = x, y

    def move_location(self, x, y, z):
        self.loc[0]+=x
        self.loc[1]+=y 
        self.loc[2]+=z
    def move_location2(self, x, y):
        self.loc[0]+=x
        self.loc[1]+=y

    def __str__(self) -> str:
        return '[UAV]:: LOC:[{}], ACC:[{}], ABW:[{}], RX:[{}], TX:[{}/{}], PARENT:[{}/{}]'.format(self.loc, self.avail_cc, self.avail_bw, self.prx, self.ptx, self.dtx, self.parent, self.parent_bw)        
    def __repr__(self) -> str:
        return self.__str__()

class IOT:
    dims = 14
    def __init__(self, tensor, ptx, dtx) -> None:
        self.tensor = tensor 

        # view
        self.loc = self.tensor[0:3]             # x,y,z +3
        self.parent = self.tensor[3:4]          # not connected - uplink parent +1
        self.parent_bw = self.tensor[4:5]       # not connected - bw assigned uplink network +1
        self.ptx = self.tensor[5:6]             # transmit power uplink network +1
        self.dtx = self.tensor[6:7]             # transmit distance (max) # reciver has to be in this range +1
        self.task = self.tensor[7:12]           # LCAOR,   - only for IOT device + 5

        self.off = self.tensor[12:13]           #<-- binary - 0 for UAV offload, 1 for BSV offload
        self.cc = self.tensor[13:14]

        self.reset_arg = (ptx, dtx, -1, 0, -1, 0)
        self.reset()

    def reset(self):        
        self.ptx[0], self.dtx[0], self.off[0], self.cc[0],self.parent[0], self.parent_bw[0] = self.reset_arg

    def set_location(self, x, y, z):
        self.loc[0], self.loc[1], self.loc[2] = x, y, z
    def set_location2(self, x, y):
        self.loc[0], self.loc[1] = x, y
    def set_task(self, 
            task_l=0,   # task size in bits
            task_c=0,   # cc per bit
            task_a=0,   # task arrival rate
            task_o=0,   # output data size
            task_r=0,   # if true then requires data output produced by this task at BSV
            ):
        self.task[0], self.task[1], self.task[2], self.task[3], self.task[4] = \
            task_l, task_c, task_a, task_o, task_r

    def move_location(self, x, y, z):
        self.loc[0]+=x
        self.loc[1]+=y 
        self.loc[2]+=z
    def move_location2(self, x, y):
        self.loc[0]+=x
        self.loc[1]+=y
        
    def cc_req(self):
        return self.task[0]*self.task[1]*self.task[2]
    def bw_req(self):
        return self.task[0]*self.task[2]
    def __str__(self) -> str:
        return '[IOT]:: LOC:[{}], TX:[{}/{}], PARENT:[{}/{}], TASK:[{}], OFF[{}/{}]'.format(self.loc, self.ptx, self.dtx, self.parent, self.parent_bw, self.task, self.off, self.cc)        
    def __repr__(self) -> str:
        return self.__str__()

class PARAMS:
    def __init__(self, n_BSV, n_UAV, n_IOT) -> None:
        self.n_BSV, self.n_UAV, self.n_IOT = n_BSV, n_UAV, n_IOT
        self.n_OFF = self.n_UAV + self.n_BSV

        self.XR, self.YR, self.ZR =             (-1000, 1000),    (-1000, 1000),  (0, 120)
        self.XR_IOT, self.YR_IOT, self.ZR_IOT = (-1000, 1000),    (-1000, 1000),    (1, 5)
        self.XR_UAV, self.YR_UAV, self.ZR_UAV = (-1000, 1000),    (-1000, 1000),    (10, 100)
        self.XR_BSV, self.YR_BSV, self.ZR_BSV = (-1000, 1000),    (-1000, 1000),    (1, 2)

        self.LR = (1000, 5000) # bits
        self.CR = (10, 20) # cc/bit
        self.AR = (1, 5) # task per sec
        self.OR = (1000, 5000) # bits out put
        self.RR = (0, 2) # this can be 0=data-not required@bsv or 1=data required at bsv
        #LCAOR

        self.AVAIL_CC_BSV = 1000_0000
        self.AVAIL_BW_BSV = 5000_000
        self.PRX_BSV = 3 #watts

        self.PTX_IOT = 1 # Watts
        self.DTX_IOT = 250 # meters

        #ptx, dtx, avail_cc, avail_bw, prx
        self.PTX_UAV = 2 # Watts
        self.DTX_UAV = 400 # meters
        self.AVAIL_CC_UAV = 1000_000 #Hz
        self.AVAIL_BW_UAV = 1000_000 # total bw
        self.PRX_UAV = 2 #watts

    def local_dis(self, x1, x2):
        return (np.sum( ( x1 - x2 ) **2 )) **0.5
        
class UeMEC(gym.Env):
    
    def __init__(self, params, seed=None, logging="", frozen=False, max_episode_steps=5) -> None:
        super().__init__()
        self.rng = default_rng(seed)
        self.params = params

        # state dimension
        self.nS = int(self.params.n_BSV*BSV.dims + self.params.n_UAV*UAV.dims + self.params.n_IOT* IOT.dims)

        # continuous action - # x,y for each bsv and uav = 2b + 2u and auto offload for each iot = +i
        self.nA = int( 2*self.params.n_BSV +  2*self.params.n_UAV + self.params.n_IOT ) 
        self.rfx = RCONV((-1,1), (self.params.XR[0], self.params.XR[1]))
        self.rfy = RCONV((-1,1), (self.params.YR[0], self.params.YR[1]))

        self._max_episode_steps = max_episode_steps
        
        basic_spaces={
            'S' :   ( (self.nS,), np.float32 ),
            'A' :   ( (self.nA,), np.float32), #<--- discrete / continuous
        }
        extended_spaces={
            'DIU': ( (self.params.n_IOT, self.params.n_UAV), np.float32),
            'DUB': ( (self.params.n_UAV, self.params.n_BSV), np.float32),
            'ACT': ( (self.nA,), np.float32), 
            'STEP': ( (1,), np.int32),
            'COST': ( (1,), np.float32),
            'ROT': ( (1,), np.float32),
        }
        spaces = {}
        spaces.update(basic_spaces)
        spaces.update(extended_spaces)
        for k,(s,d) in spaces.items():
            setattr(self, k, np.zeros(s, d))
        
        state_vector = np.zeros_like(self.S)
        self.observation_space = gym.spaces.Box(low=state_vector-np.inf, high=state_vector+np.inf, shape=state_vector.shape, dtype=np.float32)

        action_vector = np.zeros_like(self.A)
        self.action_space = gym.spaces.Box(low=action_vector-np.inf, high=action_vector+np.inf, shape=action_vector.shape, dtype=np.float32)

        self.open_log(logging) if logging else self.close_log()
        self.frozen=frozen
        self.init()

    """ ===============================================================================================================  """
    """ Logging """
    """ ===============================================================================================================  """

    def log_msg(self, *msg):
        for m in msg:
            self.logfile.write(str(m))
            self.logfile.write(' ')
        self.logfile.write('\n')
        
    def open_log(self, file, mode='w'):
        if not hasattr(self, 'logfile'):
            print('[> open logging on UeMEC :[file={}, mode={}]'.format(file, mode))
            self.logfile = open(file, mode=mode, encoding='utf-8')
            self.print = self.log_msg

    def close_log(self):
        if hasattr(self, 'logfile'):
            print('[> close logging on UeMEC')
            self.logfile.close()
            del self.logfile
        self.print = lambda *msg: None
        

    """ ===============================================================================================================  """
    """ Rendering """
    """ ===============================================================================================================  """

    def render(self, caption="UEMEC", ant=False, cover=False, offloading=False):

        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(self.params.XR[0], self.params.XR[1] )
        ax.set_ylim(self.params.YR[0], self.params.YR[1])
        ax.set_zlim(self.params.ZR[0], self.params.ZR[1])

        # draw points
        ax.scatter(self.IOT[:,0],self.IOT[:,1], self.IOT[:,2],color='tab:green', marker='d', label='IoT')
        ax.scatter(self.UAV[:,0],self.UAV[:,1], self.UAV[:,2], color='tab:blue', marker='o', label='UAV')
        ax.scatter(self.BSV[:,0],self.BSV[:,1], self.BSV[:,2], color='tab:red', marker='s', label='BSV')
        

        for i, iot in enumerate(self.iot): #<--- for each IOT
            ii = iot.loc

            # prints its index
            ax.text(ii[0], ii[1], 0, '['+str(i)+']', color='tab:green') if ant else None

            if iot.parent[0]<0:
                # drop a perpendicular
                ax.plot( (ii[0], ii[0]), (ii[1], ii[1]), (ii[2], 0), color='green', linewidth=1, linestyle='dashed' )
            else:
                ax.plot( (ii[0], ii[0]), (ii[1], ii[1]), (ii[2], 0), color='green', linewidth=0.5, linestyle='solid' )

            if cover:
                # draw its IOT cover
                cover = Circle((ii[0], ii[1]), iot.dtx[0], color='tab:green', linewidth=0.5, fill=False)
                ax.add_patch(cover)
                art3d.pathpatch_2d_to_3d(cover, z=0, zdir="z")
            
            if offloading:
                off, cc = iot.off, iot.cc
                if not (off<0):
                    cover = Circle((ii[0], ii[1]), 10, color=('tab:blue' if off==0 else 'tab:red'), linewidth=1, fill=True)
                    ax.text(ii[0], ii[1], -5, str(cc)+'cc', color='tab:green')
                else:
                    cover = Circle((ii[0], ii[1]), 10, color=('tab:green'), linewidth=1, fill=True)
                
                ax.add_patch(cover)
                art3d.pathpatch_2d_to_3d(cover, z=0, zdir="z")



        for i, uav in enumerate(self.uav):#<--- for each UAV

            ui = uav.loc
            # prints its index
            ax.text(ui[0], ui[1], 0, '['+str(i)+']', color='tab:blue') if ant else None

            # drop a perpendicular
            if uav.parent[0]<0:
                ax.plot( (ui[0], ui[0]), (ui[1], ui[1]), (ui[2], 0), color='blue', linewidth=1, linestyle='dashed' )
            else:
                ax.plot( (ui[0], ui[0]), (ui[1], ui[1]), (ui[2], 0), color='blue', linewidth=0.5, linestyle='solid' )
            
            if cover:
                # draw its UAV cover
            
                cover = Circle((ui[0], ui[1]), uav.dtx[0], color='tab:blue', fill=False, linewidth=0.5)
                ax.add_patch(cover)
                art3d.pathpatch_2d_to_3d(cover, z=0, zdir="z")
        

        for i, bsv in enumerate(self.bsv):#<--- for each BSV

            bi = bsv.loc
            # prints its index
            ax.text(bi[0], bi[1], 0, '['+str(i)+']', color='tab:red') if ant else None

            # drop a perpendicular
            ax.plot( (bi[0], bi[0]), (bi[1], bi[1]), (bi[2], 0), color='red', linewidth=0.5, linestyle='solid' )
            

        for i, iot in enumerate(self.iot):#<--- for each iot, connect to its parent
            uav, bw = int(iot.parent[0]), iot.parent_bw[0]
            if uav>=0:
                ui = self.uav[uav].loc
                ii = iot.loc
                ax.plot( (ui[0], ii[0]), (ui[1], ii[1]), (ui[2], ii[2]), color='tab:green', linewidth=1, linestyle='solid' ) # 0.1*bw

        for i, uav in enumerate(self.uav):#<--- for each iot, connect to its parent
            bsv, bw = int(uav.parent[0]), uav.parent_bw[0]
            if bsv>=0:
                ui = self.bsv[bsv].loc
                ii = uav.loc
                ax.plot( (ui[0], ii[0]), (ui[1], ii[1]), (ui[2], ii[2]), color='tab:blue', linewidth=1, linestyle='solid' )

        
        #self.bw_UAV_IOT = np.zeros((self.n_UAV, self.n_IOT), dtype=np.float32)
        #self.bw_UAV_UAV = np.zeros((self.n_UAV, self.n_UAV), dtype=np.float32)
        #self.bw_UAV_SAT = 0.0 + self.TOTAL_BW_UAV_SAT
        
        plt.legend()
        plt.title(caption)
        #plt.tight_layout()
        #plt.show()
        return fig


    """ ===============================================================================================================  """
    """ helper methods [initialization, validation] """
    """ ===============================================================================================================  """

    def _random_IOT(self):
        zz = self.rng.integers(self.params.ZR_IOT[0], self.params.ZR_IOT[1])
        for iot in self.iot:
            iot.reset()
            iot.set_location(
                x=self.rng.integers(self.params.XR_IOT[0], self.params.XR_IOT[1]), 
                y=self.rng.integers(self.params.YR_IOT[0], self.params.YR_IOT[1]),
                z=zz
                ) # randomly set location
            iot.set_task(
                task_l=self.rng.integers(self.params.LR[0], self.params.LR[1]), 
                task_c=self.rng.integers(self.params.CR[0], self.params.CR[1]), 
                task_a=self.rng.integers(self.params.AR[0], self.params.AR[1]), 
                task_o=self.rng.integers(self.params.OR[0], self.params.OR[1]), 
                task_r=self.rng.integers(self.params.RR[0], self.params.RR[1])
            ) # randomly set task parameters
        self.last_iot_state = np.zeros_like(self.IOT) + self.IOT
    def _random_UAV(self):
        zz = self.rng.integers(self.params.ZR_UAV[0], self.params.ZR_UAV[1])
        for uav in self.uav:
            uav.reset()
            uav.set_location(
                x=self.rng.integers(self.params.XR_UAV[0], self.params.XR_UAV[1]), 
                y=self.rng.integers(self.params.YR_UAV[0], self.params.YR_UAV[1]),
                z=zz
                ) # randomly set location
        self.last_uav_state = np.zeros_like(self.UAV) + self.UAV
    def _random_BSV(self):
        zz=self.rng.integers(self.params.ZR_BSV[0], self.params.ZR_BSV[1])
        for bsv in self.bsv:
            bsv.reset()
            bsv.set_location(
                x=self.rng.integers(self.params.XR_BSV[0], self.params.XR_BSV[1]), 
                y=self.rng.integers(self.params.YR_BSV[0], self.params.YR_BSV[1]),
                z=zz
                ) # randomly set location
        self.last_bsv_state = np.zeros_like(self.BSV) + self.BSV
    def random_INIT(self):
        self._random_BSV()
        self._random_UAV()
        self._random_IOT()

    def is_vaild_iot(self, iot):
        return not(iot<0 or iot>=self.params.n_IOT)
    def is_vaild_uav(self, uav):
        return not(uav<0 or uav>=self.params.n_UAV)
    def is_vaild_bsv(self, bsv):
        return not(bsv<0 or bsv>=self.params.n_BSV)

    """ ===============================================================================================================  """
    """ helper methods [distance matrix] """
    """ ===============================================================================================================  """

    def _update_distances_iot_uav(self):
        for i in range(self.params.n_IOT):
            for u in range(self.params.n_UAV):
                self.DIU[i,u] = self.params.local_dis( self.iot[i].loc , self.uav[u].loc )
    def _update_distances_bsv_uav(self):
        for b in range(self.params.n_BSV):
            for u in range(self.params.n_UAV):
                self.DUB[u, b] = self.params.local_dis( self.bsv[b].loc , self.uav[u].loc )
    def update_distances(self):
        self._update_distances_iot_uav()
        self._update_distances_bsv_uav()
    

    """ ===============================================================================================================  """
    """ Inherited """
    """ ===============================================================================================================  """

    def state(self):
        return self.S.flatten() # self.DIU.flatten(), self.DUB.flatten()

    def init(self) -> None:
        #self.print('--> ENV::[init]')

        si = 0
        ei = self.params.n_BSV*BSV.dims
        self.BSV = self.S[si: ei].reshape((self.params.n_BSV,BSV.dims))

        si = ei
        ei += self.params.n_UAV*UAV.dims
        self.UAV = self.S[si:ei].reshape((self.params.n_UAV, UAV.dims))

        si = ei
        ei += self.params.n_IOT* IOT.dims
        self.IOT = self.S[si:ei].reshape((self.params.n_IOT, IOT.dims))

        si = 0
        ei = int( 2*self.params.n_BSV)
        self.ACT_BSV=self.ACT[si:ei]

        si = ei
        ei += int( 2*self.params.n_UAV )
        self.ACT_UAV=self.ACT[si:ei]

        si = ei
        ei += int( self.params.n_IOT )
        self.ACT_IOT=self.ACT[si:ei]

        self.iot = [ IOT(self.IOT[n], self.params.PTX_IOT, self.params.DTX_IOT) for n in range(self.params.n_IOT) ] # iot devices
        self.uav = [ UAV(self.UAV[n], self.params.PTX_UAV, self.params.DTX_UAV,self.params.AVAIL_CC_UAV,self.params.AVAIL_BW_UAV, self.params.PRX_UAV) for n in range(self.params.n_UAV) ] # uav devices 
        self.bsv = [ BSV(self.BSV[n], self.params.AVAIL_CC_BSV, self.params.AVAIL_BW_BSV, self.params.PRX_BSV) for n in range(self.params.n_BSV) ] # bsv devices

        self.random_INIT()

    def reset(self) -> bool:
        return (self._reset() if self.frozen else self._restart())

    def _restart(self):
        self.print('--> ENV::[restart]')
        self.random_INIT()
        return self._reset()

    def cost(self): #<--- minimizing cost
       # print('[COST')
        cost = 0
        for iot in self.iot:
            cost -= (iot.off[0]+1)
        return cost

    def reset(self):
        self.print('--> ENV::[reset]')
        self.IOT[:] = self.last_iot_state
        self.UAV[:] = self.last_uav_state
        self.BSV[:] = self.last_bsv_state
        self.update_distances()

        self.ACT_BSV*=0
        self.ACT_UAV*=0
        self.ACT_IOT*=0
        self.A[:] = (0)
        self.ROT[:] = (0)
        self.STEP[:] = (0)
        self.COST[:] = (self.cost())

        return self.state()

    def act(self, action):
        self.print('--> ENV::[act] @ [{}]'.format(action))
        #assert(action.shape[0] == self.nA)
        self.A[:] = action
        #for b in range(self.params.n_BSV):
            #self.ACT_BSV[b] =                   self.rfx.in2map( self.A[b] )
            #self.ACT_BSV[b+self.params.n_BSV] =   self.rfy.in2map( self.A[b+self.params.n_BSV] )
        self.ACT_BSV[0:self.params.n_BSV] =                   self.rfx.in2map( self.A[0:self.params.n_BSV] )
        self.ACT_BSV[self.params.n_BSV:self.params.n_BSV*2] =   self.rfy.in2map( self.A[self.params.n_BSV:self.params.n_BSV*2] )
        #self.ACT_BSV[:] =  self.rfx.in2map( self.A[low:low+self.params.n_BSV] )

        
        low = self.params.n_BSV*2
        high = low+self.params.n_UAV
        #for u in range(self.params.n_UAV):
        #    self.ACT_UAV[u] =         self.rfx.in2map( self.A[u+low] )
        #    self.ACT_UAV[u+self.params.n_UAV] =   self.rfy.in2map( self.A[u+low+self.params.n_UAV] )
        self.ACT_UAV[0:high] =         self.rfx.in2map( self.A[0+low:high+low] )
        self.ACT_UAV[high+low:high+self.params.n_UAV+low] =   self.rfy.in2map( self.A[high+low:high+self.params.n_UAV+low] )

        low = (self.params.n_BSV+self.params.n_UAV)*2
        for i in range(self.params.n_IOT):
            self.ACT_IOT[i] = 0 if self.A[i+low] < 0 else 1

        return #self.ACT.flatten()

    def step(self, action):
        self.act(action)
        self.print('--> ENV::[step_begin]')
        # execute action
        status=True
        if status:
            for b in range(self.params.n_BSV):
                truth = self.move_BSV(b, self.ACT_BSV[b], self.ACT_BSV[b+self.params.n_BSV])
                if not truth:
                    status=False
                    break
            if status:
                for u in range(self.params.n_UAV):
                    truth = self.move_UAV(u, self.ACT_UAV[u], self.ACT_UAV[u+self.params.n_UAV])
                    if not truth:
                        status=False
                        break
                if status:
                    for i in range(self.params.n_IOT):
                        truth = self.auto_offload_IOT(i, self.ACT_IOT[i], self.iot[i].cc_req() )
                        if not truth:
                            status=False
                            break            

        #status = not((False in move_bsv_truth) or (False in move_uav_truth) or (False in iot_off_truth))
        
        
        cost = self.cost()
        reward = self.COST - cost
        self.COST[:] = (cost)
        self.ROT+=reward
        done = ((0 if status else 1))
        self.STEP+=1 # increment step
        self.print('<--- ENV::[step_end], REW:[{}], DONE:[{}]: COST:[{}], TotalREW:[{}]'.format(reward, done, self.COST, self.ROT))
        return self.state(), reward, done, {'step':self.STEP[0], 'cost':self.COST[0], 'RoT':self.ROT[0]}


    def close(self):
        self.close_log()

    """ ===============================================================================================================  """

    """ Terminator """

    def term(self, a=None, b=None, c=None):
        self.print('\t[_] term()')
        return False

    """ Movement """

    def move_BSV(self, b, x, y):
        bsv = int(b)
        self.print('\t[_] Moving BSV( bsv={}, x={}, y={} )'.format(bsv, x,y))
        if not (self.is_vaild_bsv(bsv)):
            self.print('\t[!] invalid args supplied to move_BSV()')
            return False
        
        self.print('\t[?] Current BSV location is [{}]'.format(self.bsv[bsv].loc))
        if  (x < self.params.XR_BSV[0]) or (x>self.params.XR_BSV[1]) or (y < self.params.YR_BSV[0]) or (y>self.params.YR_BSV[1]):
            self.print('\t[!] Failed to move BSV, Trying to move out of bounds')
            return False
        
        self.bsv[bsv].set_location2(x,y)
        
        #self._moved_bsv(bsv)# moving bsv might disconnect uavs
        for u in range(self.params.n_UAV):
            self.DUB[u, bsv] = self.params.local_dis( self.bsv[bsv].loc , self.uav[u].loc )
            if self.uav[u].parent[0]==bsv:
                if self.DUB[u, bsv] > self.uav[u].dtx:
                    self.print('\t[x] BSV[{}] Moved out of transmit range of UAV[{}] - will be disconnected'. format(bsv, u))
                    self.disconnect_UAV_BSV(u)

        self.print('\t[+] Success moving BSV, current location at [{}]'.format(self.bsv[bsv].loc))
        return True
        
    def move_UAV(self, u, x, y):
        uav = int(u)
        self.print('\tMoving UAV( uav={}, x={}, y={} )'.format(uav, x,y))
        if not (self.is_vaild_uav(uav)):
            self.print('\t[!] invalid args supplied to move_UAV()')
            return False
        
        self.print('\t[?] Current UAV location is [{}]'.format(self.uav[uav].loc))
        if  (x < self.params.XR_UAV[0]) or (x>self.params.XR_UAV[1]) or (y < self.params.YR_UAV[0]) or (y>self.params.YR_UAV[1]) :
            self.print('\t[!] Failed to move UAV, Trying to move out of bounds')
            return False

        self.uav[uav].set_location2(x,y)
        #self._moved_uav(uav)
        for i in range(self.params.n_IOT):
            self.DIU[i,uav] = self.params.local_dis( self.iot[i].loc , self.uav[uav].loc )
            if self.iot[i].parent[0]==uav:
                if self.DIU[i,uav] > self.iot[i].dtx:
                    self.print('\t[x] UAV[{}] Moved out of transmit range of IOT[{}] - will be disconnected'. format(uav, i))
                    self.disconnect_IOT_UAV(i)
        
        for b in range(self.params.n_BSV):
            self.DUB[uav, b] = self.params.local_dis( self.bsv[b].loc , self.uav[uav].loc )
        myb = int(self.uav[uav].parent[0])
        if myb>=0:
            if self.DUB[uav, myb] > self.uav[uav].dtx:
                self.print('\t[x] BSV[{}] is now out of transmit range of UAV[{}] - will be disconnected'. format(myb, uav))
                self.disconnect_UAV_BSV(uav)

        self.print('\t[+] Success moving UAV, now at [{}]'.format(self.uav[uav].loc))
        return True

    def moveby_BSV(self, b, x, y):
        bsv = int(b)
        if not (self.is_vaild_bsv(bsv)):
            self.print('\t[!] invalid args supplied to moveby_BSV()')
            return False
        b_loc  = self.bsv[bsv].loc[0:2]
        return self.move_BSV(bsv, b_loc[0]+x, b_loc[1]+y)
        
    def moveby_UAV(self, u, x, y):
        uav = int(u)
        if not (self.is_vaild_uav(uav)):
            self.print('\t[!] invalid args supplied to moveby_UAV()')
            return False
        u_loc  = self.uav[uav].loc[0:2]
        return self.move_UAV(uav, u_loc[0]+x, u_loc[1]+y)
        
    
    """ Connection """

    def connect_UAV_BSV(self, u, b, bw):
        uav, bsv = int(u), int(b)
        self.print('\t[_] Connecting UAV[{}] --> BSV[{}/{}]'.format(uav, bsv, bw))
        if not (self.is_vaild_bsv(bsv) and self.is_vaild_uav(uav)):
            self.print('\t[!] invalid args supplied to connect_UAV_BSV()')
            return False
        
        
        ibsv, ibw  = int(self.uav[uav].parent[0]), self.uav[uav].parent_bw[0]  
        if ibsv >= 0: # already connected to some-bsv
            assert(self.uav[uav].parent[0] >= 0) 
            assert(self.uav[uav].parent_bw[0] > 0)
            self.print('\t[!] UAV[{}] already connected to BSV[{}/{}]'.format(uav, ibsv, ibw))
            return False

        if self.DUB[uav, bsv] > self.uav[uav].dtx: #<--- is within transmission distance of uav?
            self.print('\t[!] not in transmission range')
            return False

        if self.bsv[bsv].avail_bw[0] < bw:
            print('\t[!] Connection Failed... Not enough bandwidth[{}] available at parent BSV[{}/{}]'.format(bw, bsv, self.bsv[bsv].avail_bw[0]))
            return False

        self.uav[uav].parent[0] = bsv
        self.uav[uav].parent_bw[0] = bw
        self.bsv[bsv].avail_bw[0] -= bw
        self.print('\t[+] Connection Success')
        return True
            
    def connect_IOT_UAV(self, i, u, bw):
        iot, uav = int(i), int(u)
        self.print('\t[_] Connecting IOT[{}] --> UAV[{}/{}]'.format(iot, uav, bw))
        if not (self.is_vaild_iot(iot) and self.is_vaild_uav(uav)):
            self.print('[!] invalid args supplied to connect_IOT_UAV()')
            return False
        
        iuav, ibw  = int(self.iot[iot].parent[0]), self.iot[iot].parent_bw[0] 
        if iuav >= 0: # not already connected to some-uav
            assert(self.iot[iot].parent[0] >= 0) 
            assert(self.iot[iot].parent_bw[0] > 0) 
            self.print('\t[!] IOT[{}] already connected to UAV[{}/{}]'.format(iot, iuav, ibw))
            return False
        
        if self.DIU[iot, uav] > self.iot[iot].dtx: #<--- is within transmission distance of iot?
            self.print('\t[!] not in transmission range')
            return False

        #pbsv = int(self.uav[uav].parent[0])
        #if pbsv<0:
        #    # print('Not connected to a BSV)
        #    return

        if self.uav[uav].avail_bw[0] < bw:
            self.print('\t[!] Connection Failed... Not enough bandwidth[{}] available at parent UAV[{}/{}]'.format(bw, uav, self.uav[uav].avail_bw[0]))
            return False
        
        self.iot[iot].parent[0] = uav
        self.iot[iot].parent_bw[0] = bw
        self.uav[uav].avail_bw[0] -= bw
        self.print('\t[+] Connection Success')
        return True

    def disconnect_UAV_BSV(self, u):
        
        uav = int(u)
        self.print('\t[_] disconnect_UAV_BSV( uav={} )'.format(uav))
        if not (self.is_vaild_uav(uav)):
            self.print('[!] invalid args supplied to disconnect_UAV_BSV()')
            return False
        ibsv, ibw  = int(self.uav[uav].parent[0]), self.uav[uav].parent_bw[0]  # if already connected, disconnect first
        self.print('\t[_] Dis-Connecting UAV[{}] --> BSV[{}/{}]'.format(uav, ibsv, ibw))

        if ibsv < 0: # already connected to some-bsv?
            assert(self.uav[uav].parent[0] < 0) 
            assert(self.uav[uav].parent_bw[0] == 0) 
            self.print('\t[^] UAV is already disconnected')
            return True

        for iot in range(self.params.n_IOT):
            if self.iot[iot].parent[0]==uav:
                if self.iot[iot].off[0]==1: # to bsv
                    self.onload_IOT(iot)

        self.bsv[ibsv].avail_bw[0]+=ibw # disconnect and reclaim bandwidth
        self.print('\t[+] UAV-Disconnected from BSV[{}] reclaimed bandwidth [{}/{}]'.format(ibsv, ibw, self.bsv[ibsv].avail_bw[0]))

        self.uav[uav].parent[0] = -1 
        self.uav[uav].parent_bw[0] = 0 
        return True

    def disconnect_IOT_UAV(self, i):
        iot = int(i)
        self.print('\t[_] disconnect_IOT_UAV( uav={} )'.format(iot))
        if not (self.is_vaild_iot(iot)):
            self.print('[!] invalid args supplied to disconnect_IOT_UAV()')
            return False
        iuav, ibw  = int(self.iot[iot].parent[0]), self.iot[iot].parent_bw[0]  
        self.print('\t[_] Dis-Connecting IOT[{}] connected to UAV[{}/{}]'.format(iot, iuav, ibw))
        if iuav < 0: # already connected to some-uav?
            assert(self.iot[iot].parent[0] < 0 ) 
            assert(self.iot[iot].parent_bw[0] == 0) 
            self.print('\t[^] IOT is already disconnected')
            return True
        
        self.onload_IOT(iot) if self.iot[iot].off>=0 else None # check if its offloading as well
        self.uav[iuav].avail_bw[0]+=ibw # disconnect and reclaim bandwidth
        self.print('\t[+] IOT-Disconnected from UAV [{}] reclaimed bandwidth [{}/{}]'.format(iuav, ibw, self.uav[iuav].avail_bw[0]))

        self.iot[iot].parent[0] = -1 
        self.iot[iot].parent_bw[0] = 0 
        return True


    """ Offloading """

    def onload_IOT(self, i): 
        
        iot = int(i)
        self.print('\t[_] onload_IOT( iot={} )'.format(iot))
        if not (self.is_vaild_iot(iot)):
            self.print('\t[!] invalid args supplied to onload_IOT()')
            return False
        iiot = self.iot[iot]
        i_off_loc, i_off_cc = int(iiot.off[0]), iiot.cc[0]
        self.print('\t[?] IOT[{}] is currently offloaded at LOC[{}] with [{}]CC'.format(iot, i_off_loc, i_off_cc))
        #self.print('[*] Parent-Chain: IOT[{}] --> UAV[{}] --> BSV[{}]'.format(iot, puav, pbsv))
        
        if i_off_loc == 0: # offloading currently to uav
            puav =  int(iiot.parent[0])
            self.uav[puav].avail_cc[0]+= i_off_cc
            self.print('\t[+] Reclaimed [{}]CC to UAV[{}] now has [{}]CC'.format(i_off_cc, puav, self.uav[puav].avail_cc[0]))
           
        elif i_off_loc == 1: # offloading to bsv
            puav =  int(iiot.parent[0])
            pbsv = ( -1 if puav<0 else int(self.uav[puav].parent[0]))
            if not (pbsv>=0 and pbsv<self.params.n_BSV):
                raise StopIteration('Invalid pbsv:[{}] - {}'.format(pbsv, iiot))
            self.bsv[pbsv].avail_cc[0]+= i_off_cc
            self.print('\t[+] Reclaimed [{}]CC to BSV[{}] now has [{}]CC'.format(i_off_cc, pbsv, self.bsv[pbsv].avail_cc[0]))
            
        else:
            #print('... not offloading anywhere')
            assert(self.iot[iot].off[0] < 0 )
            assert(self.iot[iot].cc[0] == 0)
            pass # not offloading anywhere - check if

        self.iot[iot].off[0] = -1 # assert(off_loc)
        self.iot[iot].cc[0] = 0 # assert(off_cc==0)
        self.print('\t[+] Offloading Disabled')
        
        return True

    def offload_IOT(self, i, off_01, off_cc ): # offloading location of iot - UAV or BSV  # cc assigned by offloading device
        iot = int(i)
        off_loc = int(off_01)
        self.print('\t[_] offload_IOT( iot={}, off_loc={}, off_cc={} )'.format(iot, off_loc, off_cc))
        if not (self.is_vaild_iot(iot)):
            self.print('\t[!] invalid args supplied to offload_IOT()')
            return False
        #assert(off_loc==0 or off_loc==1)
        #print('Trying to offload IOT[{}] at LOC[{}] with [{}]CC'.format(iot, off_loc, off_cc))
        iiot = self.iot[iot]
        i_off_loc, i_off_cc = int(iiot.off[0]), iiot.cc[0]
        self.print('\t[?] IOT[{}] is currently offloaded at LOC[{}] with [{}]CC'.format(iot, i_off_loc, i_off_cc))
        if i_off_loc>=0:
            self.print('\t[!] cannot offload an already offloading iot')
            return False

        puav =  int(iiot.parent[0])
        pbsv = ( -1 if puav<0 else int(self.uav[puav].parent[0]))

        if off_loc==0: # trying to offload to uav
            if puav<0:
                self.print('\t[!] Trying to offload to UAV but not connected to one')
                return False
            if self.uav[puav].avail_cc[0]>=off_cc:
                self.iot[iot].off[0] = off_loc
                self.iot[iot].cc[0] = off_cc
                self.uav[puav].avail_cc[0] -= off_cc
                self.print('\t[+] Offloading to UAV Success')
            else:
                self.print('\t[^] Offloading to UAV Failed ... Not enough CC[{}] available on UAV[{}/{}]CC'.format(off_cc, puav, self.uav[puav].avail_cc[0]))
        elif off_loc==1: # trying to offload to bsv
            if pbsv<0:
                self.print('\t[!] Trying to offload to BSV but not connected to one')
                return False
            if self.bsv[pbsv].avail_cc[0]>=off_cc:
                self.iot[iot].off[0] = off_loc
                self.iot[iot].cc[0] = off_cc
                self.bsv[pbsv].avail_cc[0]-=off_cc
                self.print('\t[+] Offloading to BSV Success')
            else:
                self.print('\t[^] Offloading to BSV Failed ... Not enough CC[{}] available on BSV[{}/{}]CC'.format(off_cc, puav, self.bsv[pbsv].avail_cc[0]))
        else:
            raise StopIteration('Not possible!')
        return True

    def auto_offload_IOT(self, i, off_01, off_cc ): # offloading location of iot - UAV or BSV  # cc assigned by offloading device

        # first check if iot is offloading
        # if yes - then offloading has to be valid - remove offloading

        # if no - then check if uav and basv are connected
        iot = int(i)
        off_loc = int(off_01)
        self.print('\t[_] auto_offload_IOT( iot={}, off_loc={}, off_cc={} )'.format(iot, off_loc, off_cc))
        if not (self.is_vaild_iot(iot)):
            self.print('\t[!] invalid args supplied to auto_offload_IOT()')
            return False
        #assert(off_loc==0 or off_loc==1)
        #print('Trying to offload IOT[{}] at LOC[{}] with [{}]CC'.format(iot, off_loc, off_cc))
        iiot = self.iot[iot]
        puav =  int(iiot.parent[0])
        if puav <0 : # not connected
            
            # connect to nearesst uav
            near_uav = np.argmin(self.DIU[iot, :])
            self.print('\t[?] Parent-UAV not connected!\n\t ... trying to connect to nearest UAV [{}]'.format(near_uav))
            if not self.connect_IOT_UAV(iot,near_uav , self.iot[iot].bw_req()):
                self.print('\t[!] Could not connect to near-UAV[{}]'.format(near_uav))
                return True # could not connect
            self.print('\t[+] Connected to near-UAV[{}]'.format(near_uav))
        puav =  int(iiot.parent[0])
        assert(puav>=0)
        

        i_off_loc, i_off_cc = int(iiot.off[0]), iiot.cc[0]
        self.print('\t[?] Currently Oflloading to [{}/{}]'.format(i_off_loc, i_off_cc))
        if i_off_loc>=0:
            self.onload_IOT(iot)

        if off_loc==0: # trying to offload to uav

            if self.uav[puav].avail_cc[0]>=off_cc:
                self.iot[iot].off[0] = off_loc
                self.iot[iot].cc[0] = off_cc
                self.uav[puav].avail_cc[0] -= off_cc
                self.print('\t[+] Offloading to UAV Success', self.iot[iot].off[0])
                
            else:
                self.print('\t[^] Offloading to UAV Failed ... Not enough CC[{}] available on UAV[{}] now has [{}]CC'.format(off_cc, puav, self.uav[puav].avail_cc[0]))
        elif off_loc==1: # trying to offload to bsv
            pbsv = int(self.uav[puav].parent[0])
            if pbsv <0 : # not connected
                
                # connect to nearesst bsv
                near_bsv=np.argmin(self.DUB[puav, :])
                self.print('\t[?] Parent-BSV not connected! ... trying to connect to nearest BSV [{}]'.format(near_bsv))
                if not self.connect_UAV_BSV(puav, near_bsv, self.params.AVAIL_BW_BSV/self.params.n_UAV):
                    self.print('\t[!] Could not connect to near-BSV[{}]'.format(near_bsv))
                    return True # could not connect
                self.print('\t[+] Connected to near-BSV[{}]'.format(near_bsv))
            pbsv = int(self.uav[puav].parent[0])
            assert(pbsv>=0)

            if self.bsv[pbsv].avail_cc[0]>=off_cc:
                self.iot[iot].off[0] = off_loc
                self.iot[iot].cc[0] = off_cc
                self.bsv[pbsv].avail_cc[0]-=off_cc
                self.print('\t[+] Offloading to BSV Success', self.iot[iot].off[0])
                
            else:
                self.print('\t[^] Offloading to BSV Failed ... Not enough CC[{}] available on BSV[{}] now has [{}]CC'.format(off_cc, puav, self.bsv[pbsv].avail_cc[0]))
        else:
            raise StopIteration('Not possible!')
        return True



