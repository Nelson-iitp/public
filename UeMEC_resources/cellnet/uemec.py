

xx_range = lambda V, R: ( V>R[0]     and     V<R[1]  )
xi_range = lambda V, R: ( V>R[0]     and     V<=R[1] ) 
ix_range = lambda V, R: ( V>=R[0]    and     V<R[1]  )
ii_range = lambda V, R: ( V>=R[0]    and     V<=R[1] )

class KNOW:
    """ Fixed constants that do not change accross or within an experiments """
    KILO = 10**3
    MEGA = 10**6
    GIGA = 10**9


    """ World-related """
    Xr, Yr, Zr = (-25, 25),  (-25, 25),  (0, 31) ## AREA OF ENV


    """ UAV-related """
    UAV_HEIGHT = 30              ## UAV_HEIGHT H = 30 m
    UAV_WEIGHT = 9.65            ## UAV_WEIGHT W = 9.65 KG
    UAV_SPEED = 20               ## UAV speed V = 20m/s
    UAV_ENERGY = 500*KILO        ## UAV Energy Ue = 500kJ
    UAV_HPOWER = 200             ## UAV Hovering Power : Energy/sec = POWer =  220W = 200 J/s = 200J in 1 sec
    UAV_AVAIL_CC = 8*GIGA        ## computing ability of edge server is set as 8 GHz,
    UAV_COEFF = 10**-28          ## coefficients related to the IMDs and the edge server are set as γ_{i}^k = γ^c = 10^-28

    """ IOT-related """
    IOT_CCr =(1*GIGA, 2*GIGA)        ##computing ability of IMDs is set between 1 and 2 GHz  
    IOT_Lr = (0.5*MEGA, 1*MEGA)      # bits ##For task I_{k},the size of input data L_{k} is randomly between 0.5 and 1 Mbits
    IOT_Cr = (500, 1500)             # cc/bit ##CPU cycles for computing one bit of the task is between 500 and 1500
    IOT_Or = (0.5, 1.5)              # ratio of bits out put ##number of calculation outcome per input bit is set as O_{k} = 0.5.
    IOT_Tr = (0.1, 6)                ##maximum permissible latency for completing the task is randomly set between 0.1 and 6 sec


    
    """ Communication related """
    COM_frames = 50         ##Transmission part can be partitioned into I = 50 frames
    COM_d0 = 1              ##Reference distance d0 = 1 m
    COM_h0 = -50            ##The channel power gain h0 is set to be −50 dB, 
    COM_B = 40*MEGA         ##the communication bandwidth B = 40 MHz
    COM_UDRr = (1, 2.5)    ## range of uplink to downlink


    """ Offloading related """
    n_OFF = 2               ## binary offloading scheme - 0: IOT,  1: UAV
    n_IOTr = (1, 7)         ## range of no of IOT devices - no of offloading decicsions 
    n_UAVr =  (1, 5)        ## range of no of UAV devices - no of offloding locations available

class PARAMS:
    """ Variable Parameters that may change accross but not within an experiment """
    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self, k, v)

    def verify(self, know):
        truth = \
            ii_range(       self.n_IOT,        know.n_IOTr         ) and \
            ii_range(       self.n_UAV,        know.n_UAVr         ) and \
            ii_range(       self.T,            know.IOT_Tr         ) and \
            ii_range(       self.UDR,          know.COM_UDRr       ) 
        return truth


    def default():
        return PARAMS(
            n_IOT = 7,
            n_UAV = 1,
            T = 6,
            UDR = 1,
            )
 
class PROBLEM:
    # define a problem as a single vector
    pass

class SOLUTION:
    # define a solution as a single vector
    pass

def COST(know, params, problem, solution):
    # assume that 'know' and 'params' are fixed
    # find cost of a given ('problem', 'solution') pair
    cost = 0
    return cost #<---- returns a scalar that has to be 'minimized'