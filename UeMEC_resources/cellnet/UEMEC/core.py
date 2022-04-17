# On Demand UEC (UAV-enabled Edge Computing) 

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from .rl import SPACE, ENV, MEMORY
import gym
import gym.spaces
from numpy.random import default_rng
#from known.dummy import O
import torch as tt
def local_dis(x1, x2):
    return (tt.sum( ( x1 - x2 )**2 ))**0.5

class BSV: # bsv does not transmit anything
    dims = 6
    def __init__(self, tensor, avail_cc, avail_bw, prx) -> None:
        self.tensor = tensor #tt.zeros( size=(self.dims, ), dtype=dtype, device=device)      

        self.loc = self.tensor[0:3]             # x,y,z +3
        self.avail_cc = self.tensor[3:4]        # Hz available on this device +1
        self.avail_bw = self.tensor[4:5]        # total BW (Hz) available on downlink network +1
        self.prx = self.tensor[5:6]    

        # init filling
        self.avail_cc[0], self.avail_bw[0], self.prx[0] = avail_cc, avail_bw, prx
    def set_location(self, x, y, z):
        self.loc[0], self.loc[1], self.loc[2] = x, y, z
    def set_location2(self, x, y):
        self.loc[0], self.loc[1] = x, y
    def move(self, x, y):
        self.loc[0]+=x,
        self.loc[1]-=y
    def __str__(self) -> str:
        return '[BSV]:: LOC:[{}], ACC:[{}], ABW:[{}], PRX:[{}]'.format(self.loc, self.avail_cc, self.avail_bw, self.prx)        
    def __repr__(self) -> str:
        return self.__str__()

class UAV:
    dims = 10
    def __init__(self, tensor, ptx, dtx, avail_cc, avail_bw, prx) -> None:
        self.tensor = tensor

        self.loc = self.tensor[0:3] # x,y,z +3
        self.parent = self.tensor[3:4]      # not connected - uplink parent +1
        self.parent_bw = self.tensor[4:5]      # not connected - bw assigned uplink network +1
        self.ptx = self.tensor[5:6]             # transmit power uplink network +1
        self.dtx = self.tensor[6:7]           # transmit distance (max) # reciver has to be in this range +1
        self.avail_cc = self.tensor[7:8]        # Hz available on this device +1
        self.avail_bw = self.tensor[8:9]        # total BW (Hz) available on downlink network +1
        self.prx = self.tensor[9:10]            # recieve power downlink network +1
        
        # init filling
        self.ptx[0], self.dtx[0], self.prx[0] = ptx, dtx, prx
        self.avail_cc[0], self.avail_bw[0] = avail_cc, avail_bw
        self.parent[0] = -1
        self.parent_bw[0] = 0
    def set_location(self, x, y, z):
        self.loc[0], self.loc[1], self.loc[2] = x, y, z
    def set_location2(self, x, y):
        self.loc[0], self.loc[1] = x, y
    def __str__(self) -> str:
        return '[UAV]:: LOC:[{}], ACC:[{}], ABW:[{}], RX:[{}], TX:[{}/{}], PARENT:[{}/{}]'.format(self.loc, self.avail_cc, self.avail_bw, self.prx, self.ptx, self.dtx, self.parent, self.parent_bw)        
    def __repr__(self) -> str:
        return self.__str__()

class IOT:
    dims = 14
    def __init__(self, tensor, ptx, dtx) -> None:
        self.tensor = tensor #tt.zeros( size=(self.dims, ), dtype=dtype, device=device) 

        self.loc = self.tensor[0:3] # x,y,z +3
        self.parent = self.tensor[3:4]      # not connected - uplink parent +1
        self.parent_bw = self.tensor[4:5]      # not connected - bw assigned uplink network +1
        self.ptx = self.tensor[5:6]             # transmit power uplink network +1
        self.dtx = self.tensor[6:7]           # transmit distance (max) # reciver has to be in this range +1
        self.task = self.tensor[7:12]  # LCAOR,   - only for IOT device + 5

        self.off = self.tensor[12:13] #<-- binary - 0 for UAV offload, 1 for BSV offload
        self.cc = self.tensor[13:14]

        # init filling
        self.ptx[0], self.dtx[0] = ptx, dtx
        self.off[0], self.cc[0] = -1, 0
        self.parent[0] = -1
        self.parent_bw[0] = 0
        
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
    def __str__(self) -> str:
        return '[IOT]:: LOC:[{}], TX:[{}/{}], PARENT:[{}/{}], TASK:[{}], OFF[{}/{}]'.format(self.loc, self.ptx, self.dtx, self.parent, self.parent_bw, self.task, self.off, self.cc)        
    def __repr__(self) -> str:
        return self.__str__()

class PARAMS:
    def __init__(self, n_BSV, n_UAV, n_IOT) -> None:
        self.n_BSV, self.n_UAV, self.n_IOT = n_BSV, n_UAV, n_IOT
        self.n_OFF = self.n_UAV + self.n_BSV

        self.XR, self.YR, self.ZR =             (-1000, 1000),  (-1000, 1000),  (0, 120)
        self.XR_IOT, self.YR_IOT, self.ZR_IOT = (-900, 900),    (-900, 900),    (1, 5)
        self.XR_UAV, self.YR_UAV, self.ZR_UAV = (-900, 900),    (-900, 900),    (10, 100)
        self.XR_BSV, self.YR_BSV, self.ZR_BSV = (-900, 900),    (-900, 900),    (1, 2)

        self.LR = (1000, 5000) # bits
        self.CR = (10, 20) # cc/bit
        self.AR = (1, 5) # task per sec
        self.OR = (1000, 5000) # bits out put
        self.RR = (0, 2) # this can be 0=data-not required@bsv or 1=data required at bsv
        #LCAOR

        self.AVAIL_CC_BSV = 50_000
        self.AVAIL_BW_BSV = 50_000
        self.PRX_BSV = 3 #watts

        self.PTX_IOT = 1 # Watts
        self.DTX_IOT = 250 # meters

        #ptx, dtx, avail_cc, avail_bw, prx
        self.PTX_UAV = 2 # Watts
        self.DTX_UAV = 400 # meters
        self.AVAIL_CC_UAV = 10_000 #Hz
        self.AVAIL_BW_UAV = 10_000 # total bw
        self.PRX_UAV = 2 #watts

        self.max_episode_steps=1000

class UeMEC(ENV):
    
    def __init__(self, device, params,  seed=None) -> None:
        self.rng = default_rng(seed)
        self.max_episode_steps = params.max_episode_steps
        self.params = params
        spaces = {
            'BSV': SPACE( (self.params.n_BSV, BSV.dims), tt.float32 ),
            'UAV': SPACE( (self.params.n_UAV, UAV.dims), tt.float32 ),
            'IOT': SPACE( (self.params.n_IOT, IOT.dims), tt.float32 ),

            'DIU': SPACE( (self.params.n_IOT, self.params.n_UAV), tt.float32),
            'DUB': SPACE( (self.params.n_UAV, self.params.n_BSV), tt.float32),

            #'POS_BSV': SPACE( (self.params.n_BSV, 2), dtype=tt.float32 ), # x, y cordinates of BSVs
            #'POS_UAV': SPACE( (self.params.n_UAV, 2), dtype=tt.float32 ), # x, y cordinates of UAVs

            #'BW_UAV': SPACE( (self.params.n_UAV, 2), dtype=tt.float32 ), # BW_UAV = for each uav - which bsv its connected and how much b/w assigned
            #'BW_IOT': SPACE( (self.params.n_IOT, 2), dtype=tt.float32 ), # BW_IOT = for each iot - which uav its connected and how much b/w assigned

            #'OFF_LOC':  SPACE ( (self.params.n_IOT,) , dtype=tt.int8, low=0, high=2, discrete=True ), # offloading location - 0 for UAV, 1 for BSV
            #'OFF_CC':   SPACE ( (self.params.n_IOT,) , dtype=tt.float32) , # cc assigned from either uav or bsv based on off locations

            'REW':  SPACE( (), dtype=tt.float32),
            'TERM':  SPACE( (), dtype=tt.bool),
            'TS':    SPACE( (), dtype=tt.int32),
        }
        super().__init__(device, spaces, buffer_as_attr=True)
        
    def snapping(self, cap):
        self.enable_snap(MEMORY(self.device, cap, self.spaces))
    
    def state(self):
        return tt.hstack((self.BSV.flatten(), self.UAV.flatten(), self.IOT.flatten(), self.DIU.flatten(), self.DUB.flatten()))

    def init(self) -> None:
        self.iot = [ IOT(self.IOT[n], self.params.PTX_IOT, self.params.DTX_IOT) for n in range(self.params.n_IOT) ] # iot devices
        self.uav = [ UAV(self.UAV[n], self.params.PTX_UAV, self.params.DTX_UAV,self.params.AVAIL_CC_UAV,self.params.AVAIL_BW_UAV, self.params.PRX_UAV) for n in range(self.params.n_UAV) ] # uav devices 
        self.bsv = [ BSV(self.BSV[n], self.params.AVAIL_CC_BSV, self.params.AVAIL_BW_BSV, self.params.PRX_BSV) for n in range(self.params.n_BSV) ] # bsv devices
        self.random_INIT()
        self.ACTIONS = [
            self.noop,                  # 0
            self.move_BSV,              # 1
            self.move_UAV,              # 2
            self.disconnect_IOT_UAV,    # 3
            self.connect_IOT_UAV,       # 4
            self.disconnect_UAV_BSV,    # 5
            self.connect_UAV_BSV,       # 6
            self.onload_IOT,            # 7
            self.offload_IOT,           # 8
        ]
        self.AD={a.__name__:i for i,a in enumerate(self.ACTIONS) }
        

    """ ===============================================================================================================  """
    """ helper methods [1] """
    """ ===============================================================================================================  """

    def _update_distances_iot_uav(self):
        for i in range(self.params.n_IOT):
            for u in range(self.params.n_UAV):
                self.DIU[i,u] = local_dis( self.iot[i].loc , self.uav[u].loc )
    def _update_distances_bsv_uav(self):
        for b in range(self.params.n_BSV):
            for u in range(self.params.n_UAV):
                self.DUB[u, b] = local_dis( self.bsv[b].loc , self.uav[u].loc )
    def update_distances(self):
        self._update_distances_iot_uav()
        self._update_distances_bsv_uav()

    def _random_IOT(self):
        zz = self.rng.integers(self.params.ZR_IOT[0], self.params.ZR_IOT[1])
        for iot in self.iot:
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
        self.last_iot_state = tt.zeros_like(self.IOT) + self.IOT
    def _random_UAV(self):
        zz = self.rng.integers(self.params.ZR_UAV[0], self.params.ZR_UAV[1])
        for uav in self.uav:
            uav.set_location(
                x=self.rng.integers(self.params.XR_UAV[0], self.params.XR_UAV[1]), 
                y=self.rng.integers(self.params.YR_UAV[0], self.params.YR_UAV[1]),
                z=zz
                ) # randomly set location
        self.last_uav_state = tt.zeros_like(self.UAV) + self.UAV
    def _random_BSV(self):
        zz=self.rng.integers(self.params.ZR_BSV[0], self.params.ZR_BSV[1])
        for bsv in self.bsv:
            bsv.set_location(
                x=self.rng.integers(self.params.XR_BSV[0], self.params.XR_BSV[1]), 
                y=self.rng.integers(self.params.YR_BSV[0], self.params.YR_BSV[1]),
                z=zz
                ) # randomly set location
        self.last_bsv_state = tt.zeros_like(self.BSV) + self.BSV
    def random_INIT(self):
        self._random_BSV()
        self._random_UAV()
        self._random_IOT()
    

    """ ===============================================================================================================  """
    """ gym.ENV based methods """
    """ ===============================================================================================================  """

    def reset(self):
        self.IOT.data.copy_(self.last_iot_state)
        self.UAV.data.copy_(self.last_uav_state)
        self.BSV.data.copy_(self.last_bsv_state)
        self.update_distances()
        self.REW.fill_(0.0)
        self.TERM.fill_(False)
        self.TS.fill_(0)
        return self.state()

    def step(self, action):
        # action 
        # - move bsv/uav to a location  move(bsv, x, y) move(uav, x, y)
        # - connect or disconnect uat-iot   connect(iot, uav, bw), dis(iot)
        # - connect or disconnect bsv connect(uav, bsv, bw), dis(uav)
        # - offload iot's task off( loc, cc)
        aF, a1, a2, a3 = int(action[0]), action[1], action[2], action[3]
        assert(aF>=0 and aF<len(self.ACTIONS))
        self.ACTIONS[aF](a1, a2, a3)
        

        self.REW.fill_(0.0)
        
        self.TS+=1
        if self.TS>self.max_episode_steps:
            self.TERM.fill_(True)

        return self.state(), self.REW.item(), self.TERM.item(), self.TS.item()

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
                off, cc = iot.off.item(), iot.cc.item()
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
            uav, bw = int(iot.parent[0].item()), iot.parent_bw[0].item()
            if uav>=0:
                ui = self.uav[uav].loc
                ii = iot.loc
                ax.plot( (ui[0], ii[0]), (ui[1], ii[1]), (ui[2], ii[2]), color='tab:green', linewidth=1, linestyle='solid' ) # 0.1*bw

        for i, uav in enumerate(self.uav):#<--- for each iot, connect to its parent
            bsv, bw = int(uav.parent[0].item()), uav.parent_bw[0].item()
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
        plt.show()
        return fig

    """ ===============================================================================================================  """
    """ helper methods [2] """
    """ ===============================================================================================================  """

    def is_vaild_iot(self, iot):
        return not(iot<0 or iot>=self.params.n_IOT)
    def is_vaild_uav(self, uav):
        return not(uav<0 or uav>=self.params.n_UAV)
    def is_vaild_bsv(self, bsv):
        return not(bsv<0 or bsv>=self.params.n_BSV)
    def _moved_bsv(self, bsv):
        for u in range(self.params.n_UAV):
            self.DUB[u, bsv] = local_dis( self.bsv[bsv].loc , self.uav[u].loc )
            if self.uav[u].parent[0].item()==bsv:
                if self.DUB[u, bsv] > self.uav[u].dtx:
                    print('BSV[{}] Moved out of transmit range of UAV[{}]'. format(bsv, u))
                    self.disconnect_UAV_BSV(u)
    def _moved_uav(self, uav):
        for i in range(self.params.n_IOT):
            self.DIU[i,uav] = local_dis( self.iot[i].loc , self.uav[uav].loc )
            if self.iot[i].parent[0].item()==uav:
                if self.DIU[i,uav] > self.iot[i].dtx:
                    print('UAV[{}] Moved out of transmit range of IOT[{}]'. format(uav, i))
                    self.disconnect_IOT_UAV(i)
        myb = int(self.uav[uav].parent[0].item())
        for b in range(self.params.n_BSV):
            self.DUB[uav, b] =local_dis( self.bsv[b].loc , self.uav[uav].loc )
        if self.DUB[uav, myb] > self.uav[uav].dtx:
            print('BSV[{}] is now out of transmit range of UAV[{}]'. format(myb, uav))
            self.disconnect_UAV_BSV(uav)


    """ ===============================================================================================================  """
    """ Actions Available """
    """ ===============================================================================================================  """

    def noop(self, d1=None, d2=None, d3=None):
        pass

    def move_BSV(self, bsv, x, y):
        if not (self.is_vaild_bsv(bsv)):
            print('! - Invalid args')
            return
        print('Trying to move BSV[{}] from [{}] to [{}]'.format(bsv, self.bsv[bsv].loc, (x,y)))
        if  (x >= self.params.XR_BSV[0]) and (x<self.params.XR_BSV[1]) and \
            (y >= self.params.YR_BSV[0]) and (y<self.params.YR_BSV[1]):

            self.bsv[bsv].set_location2(x,y)
            self._moved_bsv(bsv)# moving bsv might disconnect uavs
            print('... Success moving BSV, now at [{}]'.format(self.bsv[bsv].loc))
        else:
            print('... Failed moving BSV, Trying to move out of bounds')
        return
        
    def move_UAV(self, uav, x, y):
        if not (self.is_vaild_uav(uav)):
            print('! - Invalid args')
            return
        print('Trying to move UAV[{}] from [{}] to [{}]'.format(uav, self.uav[uav].loc, (x,y)))
        if  (x >= self.params.XR_UAV[0]) and (x<self.params.XR_UAV[1]) and \
            (y >= self.params.YR_UAV[0]) and (y<self.params.YR_UAV[1]) :
            self.uav[uav].set_location2(x,y)
            self._moved_uav(uav)
            print('... Success moving UAV, now at [{}]\n'.format(self.uav[uav].loc))
        else:
            print('... Failed moving UAV, Trying to move out of bounds')
        return

    def disconnect_IOT_UAV(self, iot, d1=None, d2=None):
        if not (self.is_vaild_iot(iot)):
            print('! - Invalid args')
            return
        iuav, ibw  = int(self.iot[iot].parent[0]), self.iot[iot].parent_bw[0]  
        print('Dis-Connecting IOT[{}] connected to UAV[{}/{}]\n'.format(iot, iuav, ibw))
        if iuav >= 0: # already connected to some-uav

            self.uav[iuav].avail_bw[0]+=ibw # disconnect and reclaim bandwidth
            print('IOT-Disconnected from UAV [{}] reclaimed bandwidth [{}/{}]\n'.format(iuav, ibw, self.uav[iuav].avail_bw[0]))
            if not(self.iot[iot].off<0):
                self.onload_IOT(iot)
            self.iot[iot].parent[0] = -1 # assert(uav==-1)
            self.iot[iot].parent_bw[0] = 0 # assert(bw==0)
            # check if its offloading as well

        else:
            assert(self.iot[iot].parent[0] < 0 ) # assert(uav==-1)
            assert(self.iot[iot].parent_bw[0] == 0) # assert(bw==0))
            print('IOT is already disconnected')
        return

    def connect_IOT_UAV(self, iot, uav, bw):
        
        if not (self.is_vaild_iot(iot) and self.is_vaild_uav(uav)):
            print('!- Invalid args')
            return
        print('Connecting IOT[{}] --> UAV[{}/{}]\n'.format(iot, uav, bw))
        iuav, ibw  = int(self.iot[iot].parent[0]), self.iot[iot].parent_bw[0]  # if already connected, disconnect first

        if iuav < 0: # not already connected to some-uav
            if self.DIU[iot, uav] > self.iot[iot].dtx: #<--- is within transmission distance of iot?
                print('UAV [{}]m not in transmission range [{}]m of IOT \n'.format(self.DIU[iot, uav], self.iot[iot].dtx))
            else:
                pbsv = int(self.uav[uav].parent[0].item())
                if pbsv<0:
                    print('Cannot connect to AdHoC UAV')
                else:
                    if self.uav[uav].avail_bw[0] >= bw:
                        self.iot[iot].parent[0] = uav
                        self.iot[iot].parent_bw[0] = bw
                        self.uav[uav].avail_bw[0] -= bw
                        print('... Connection Success\n')
                    else:
                        print('... Connect Failed\n... Not enough bandwidth[{}] available at parent UAV[{}] now has [{}]\n'.format(bw, uav, self.uav[uav].avail_bw[0]))
        else: 
            assert(self.iot[iot].parent[0] >= 0) # assert(uav==-1)
            assert(self.iot[iot].parent_bw[0] > 0) # assert(bw==0))
            print('IOT[{}] already connected to UAV[{}/{}]\n'.format(iot, iuav, ibw))
        return

    def connect_UAV_BSV(self, uav, bsv, bw):
        if not (self.is_vaild_bsv(bsv) and self.is_vaild_uav(uav)):
            print('!- Invalid args')
            return
        print('Connecting UAV[{}] --> BSV[{}/{}]\n'.format(uav, bsv, bw))
        ibsv, ibw  = int(self.uav[uav].parent[0]), self.uav[uav].parent_bw[0]  # if already connected, disconnect first

        if ibsv < 0: # not already connected to some-bsv
            if self.DUB[uav, bsv] > self.uav[uav].dtx: #<--- is within transmission distance of uav?
                print('BSV [{}]m not in transmission range [{}]m of UAV \n'.format(self.DUB[uav, bsv], self.uav[uav].dtx))
            else:
                if self.bsv[bsv].avail_bw[0] >= bw:
                    self.uav[uav].parent[0] = bsv
                    self.uav[uav].parent_bw[0] = bw
                    self.bsv[bsv].avail_bw[0] -= bw
                    print('... Connection Success')
                else:
                    print('... Connect Failed\n... Not enough bandwidth[{}] available at parent BSV[{}/{}]\n'.format(bw, bsv, self.bsv[bsv].avail_bw[0]))
        else: # trying to disconnect
            assert(self.uav[uav].parent[0] >= 0) # assert(uav==-1)
            assert(self.uav[uav].parent_bw[0] > 0) # assert(bw==0))
            print('UAV[{}] already connected to BSV[{}/{}]\n'.format(uav, ibsv, ibw))
        return

    def disconnect_UAV_BSV(self, uav, d1=None, d2=None):
        if not (self.is_vaild_uav(uav)):
            print('! - Invalid args')
            return
        ibsv, ibw  = int(self.uav[uav].parent[0]), self.uav[uav].parent_bw[0]  # if already connected, disconnect first
        print('Dis-Connecting UAV[{}] --> BSV[{}/{}]\n'.format(uav, ibsv, ibw))

        if ibsv >= 0: # already connected to some-bsv
            self.bsv[ibsv].avail_bw[0]+=ibw # disconnect and reclaim bandwidth
            print('UAV-Disconnected from BSV[{}] reclaimed bandwidth [{}/{}]\n'.format(ibsv, ibw, self.bsv[ibsv].avail_bw[0]))

            for i in range(self.params.n_IOT):
                if self.iot[i].parent[0].item() == uav:
                    self.disconnect_IOT_UAV(i)

            self.uav[uav].parent[0] = -1 # assert(uav==-1)
            self.uav[uav].parent_bw[0] = 0 # assert(bw==0)

            # when uav ded from bsv, all its iot will dc

        else:
            assert(self.uav[uav].parent[0] < 0) # assert(uav==-1)
            assert(self.uav[uav].parent_bw[0] == 0) # assert(bw==0))
            print('UAV is already disconnected\n')
        return

    def onload_IOT(self, iot, d1=None, d2=None): 
        if not (self.is_vaild_iot(iot)):
            print('! - Invalid args')
            return
        iiot = self.iot[iot]
        i_off_loc, i_off_cc = int(iiot.off[0]), iiot.cc[0]
        print('Trying to onload IOT[{}] currently offloaded at LOC[{}] with [{}]CC\n'.format(iot, i_off_loc, i_off_cc))

        puav =  int(iiot.parent[0].item())
        pbsv = ( -1 if puav<0 else int(self.uav[puav].parent[0]))
        print('Parent-Chain: IOT[{}] --> UAV[{}] --> BSV[{}]'.format(iot, puav, pbsv))

        if i_off_loc == 0: # offloading currently to uav
            assert(puav>=0 and puav<self.params.n_UAV)
            self.uav[puav].avail_cc[0]+= i_off_cc
            print('Reclaimed [{}]CC to UAV[{}] now has [{}]CC\n'.format(i_off_cc, puav, self.uav[puav].avail_cc[0]))
        elif i_off_loc == 1: # offloading to bsv
            assert(pbsv>=0 and pbsv<self.params.n_BSV)
            self.bsv[pbsv].avail_cc[0]+= i_off_cc
            print('Reclaimed [{}]CC to BSV[{}] now has [{}]CC\n'.format(i_off_cc, pbsv, self.bsv[pbsv].avail_cc[0]))
        else:
            print('... not offloading anywhere')
            assert(self.iot[iot].off[0] < 0 )
            assert(self.iot[iot].cc[0] == 0)
            pass # not offloading anywhere - check if

        self.iot[iot].off[0] = -1 # assert(off_loc)
        self.iot[iot].cc[0] = 0 # assert(off_cc==0)
        print('... Oddloading Disabled')
        return

    def offload_IOT(self, iot, off_loc, off_cc ): # offloading location of iot - UAV or BSV  # cc assigned by offloading device
        if not (self.is_vaild_iot(iot)):
            print('! - Invalid args')
            return
        assert(off_loc==0 or off_loc==1)
        print('Trying to offload IOT[{}] at LOC[{}] with [{}]CC\n'.format(iot, off_loc, off_cc))
        iiot = self.iot[iot]
        i_off_loc, i_off_cc = int(iiot.off[0]), iiot.cc[0]
        print('Currently Oflloading to [{}/{}]\n'.format(i_off_loc, i_off_cc))
        if i_off_loc>=0:
            print('already offloading')
            return

        puav =  int(iiot.parent[0].item())
        pbsv = ( -1 if puav<0 else int(self.uav[puav].parent[0]))

        
        if off_loc==0: # trying to offload to uav
            if self.uav[puav].avail_cc[0]>=off_cc:
                self.iot[iot].off[0] = off_loc
                self.iot[iot].cc[0] = off_cc
                self.uav[puav].avail_cc[0] -= off_cc
                print('... Offloading to UAV Success')
            else:
                print('... Offloading to UAV Failed\n... Not enough CC[{}] available on UAV[{}] now has [{}]CC\n'.format(off_cc, puav, self.uav[puav].avail_cc[0]))
        elif off_loc==1: # trying to offload to bsv
            if self.bsv[pbsv].avail_cc[0]>=off_cc:
                self.iot[iot].off[0] = off_loc
                self.iot[iot].cc[0] = off_cc
                self.bsv[pbsv].avail_cc[0]-=off_cc
                print('... Offloading to BSV Success')
            else:
                print('... Offloading to BSV Failed\n... Not enough CC[{}] available on BSV[{}] now has [{}]CC\n'.format(off_cc, puav, self.bsv[pbsv].avail_cc[0]))
        else:
            raise StopIteration('Not possible!')
        return


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