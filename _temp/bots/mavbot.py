
from math import  inf, pi
import numpy as np
import gym.spaces
#import datetime
#now = datetime.datetime.now
from .common import REMAP, MEM, observation_key, action_key, done_key, step_key
from controller import Supervisor
# to pause simulation use - self.simulationSetMode(0)

    
class MAVSUPER(Supervisor):
    default_args = {
        '--time_step_multiplier':       (int, 100), # usually a large step
        '--epochs' :                    (int, 1), # no of simulation reset
        '--episodes':                   (int, 1), #<---- episdoes per epoch - each bot completes this much episodes each epoch
        '--horizon':                    (int, 0), # zero means inf (just to keep it int)
        '--members':                    (str, "") # should not be blank, do not add the superbot itself
        } 
    def __init__(self, parser):
        # remeber to call   parser = argparse.ArgumentParser() in controler
        for arg_name,(arg_type, arg_default) in self.default_args.items():
            parser.add_argument(arg_name, type=arg_type, default=arg_default)
        for (k,v) in parser.parse_args()._get_kwargs():
            setattr(self, k, v) #<---- sets on self
        # any other initialization here
            
    def run(self, verbose=False):
        members = self.members.split(',')
        while '' in members:
            members.remove('')
        if not members:
            raise Exception('No members found!')
        

        # get default_arguments of the bot
        bot_ctrl = 'mavrl'
        bot_args = { 
            '--episodes': str( self.episodes ),
            '--horizon': str( self.horizon ),
            }
        #{ k:str(t(d)) for k,(t,d) in MAVBOT.default_args.items() } #<---- everybot should implement default_args dict
        total_args = len(bot_args)

        # initialize the supervisior itself
        super().__init__() #<--- this is where supervisor initializes 
        quanta = int(self.getBasicTimeStep())
        timestep = self.time_step_multiplier * quanta

        # verbose
        robot_name = self.getName()
        #msg = lambda text : print(f'[{robot_name} @ ({now()})]::[{text}]')
        msg = ((lambda text : print(f'[{robot_name}]: {text}') ) if verbose else (lambda text : None ))
        msg(f'({bot_ctrl=})::{members=}')
        # get members and its controller
        bots = [ self.getFromDef(m) for m in members ]
        botsctrl = [ b.getField('controller') for b in bots ]
        botsctrlargs = [ b.getField('controllerArgs') for b in bots ]
        # Note: initially everything is blank - no controller and args have been created
        # ready for simulaing episodes
        self.step(quanta) # take one quant in the simulation #<---- available on super()
        for epoch in range(self.epochs):

            # reset simulation
            self.simulationReset()  # by default, it doesn not reset controller
            msg(f'Start of epoch: {epoch+1}')

            # set controllers on robots
            for a,c in zip(botsctrlargs, botsctrl):
                for k,v in bot_args.items():
                    a.insertMFString(-1, f'{k}={v}')
                c.setSFString(bot_ctrl)
            
            # <--------------------------------------------------------
            # all agents are called here
            # <--------------------------------------------------------

            # feed back loop with longer timestep ( > quanta)
            while (self.step(timestep)!=-1):
                # read each robot's last controller argument
                # the last controller argument is added by the bot itself when it is initialized
                alive = [ (c.getMFString(-1)) for c in botsctrlargs ]
                if not ('' in alive):
                    
                    break # --> all conteollers have exited, reset simulation
    
            # <--------------------------------------------------------
            # all agents return here
            # <--------------------------------------------------------

            # all contollers are terminated at this point, reset now
            # NOTE: here we could use
            #       for b in self.bots:
            #          b.restartController()
            # but we may want to load a different controller on each epoch, 
            # #so we completely remove the previous controller
            for a,c in zip(botsctrlargs, botsctrl):
                for _ in range(total_args+1):
                    a.removeMF(-1)
                c.setSFString('')
            msg(f'End of epoch: {epoch+1}')

        # spend one last timestep in the world
        self.step(timestep) 
        msg('Done! Simulation Paused')
        return self.simulationSetMode(0)



class MAVBOT(Supervisor):
    default_args = {
    #'--time_step_multiplier':       (int, 1), #<--- this is default
    #'--delta_control':              (int, 1), #<--- this is default
    '--episodes':                   (int, 1),
    '--horizon':                    (float, 0.0),
    }
    def __init__(self, parser):
        # remeber to call   parser = argparse.ArgumentParser() in controler
        for arg_name,(arg_type, arg_default) in self.default_args.items():
            parser.add_argument(arg_name, type=arg_type, default=arg_default)
        # parser.parse_args() returns a Namespace object 
        # to get arguments, use - [Namesapce.__dict__]   or  [for (k,v) in Namesapce._get_kwargs():] 
        for (k,v) in parser.parse_args()._get_kwargs():
            setattr(self, k, v) #<---- sets on self
        self.initialize()

    def initialize(self):

        if self.horizon<1:
            self.horizon = inf

        # initialize------------------------------------------------------------------------------------------
        self.state_dtype = np.float64
        self.action_dtype = np.float64

        # known constants
        self.battery_scan_time =     1000      
        self.battery_min_val =       100.0    
        self.initial_wait_time =     (self.battery_scan_time/1000) + 0.1 
        self.final_wait_time =       self.initial_wait_time * 3
        self.delta_control =         True #<--- this is default
        
        # delta continous : increments/decrements values by any amount within range (requires clipping)
        # controls delta                     roll,    pitch,    yaw,    altitude
        self.control_delta_low =  np.array((   -pi/8,      -pi/8,     -pi/8,    -0.1  ), dtype=self.action_dtype)
        self.control_delta_high = np.array((    pi/8,      pi/8,     pi/8,      0.1   ), dtype=self.action_dtype)
        #self.control_delta_zero = np.array((    0.0,     0.0,     0.0,     0.0    ), dtype=self.action_dtype) # starting default value
        self.mapped_delta_range = (self.control_delta_low, self.control_delta_high)

        # direct controls : sets the values directly within range
        # controls_direct                      roll,  pitch,   yaw,  altitude
        self.control_direct_low =  np.array((   -pi,   -pi,   -pi,    0.1    ), dtype=self.action_dtype)
        self.control_direct_high = np.array((   pi,   pi,   pi,     10.0  ), dtype=self.action_dtype)
        #self.control_direct_zero = np.array((    0.0,    0.0,    0.0,    1.0    ), dtype=self.action_dtype) # starting default value
        self.mapped_direct_range = (self.control_direct_low, self.control_direct_high)
        
        self.initial_altitude = 1.0
        self.default_actuators = np.array((    0.0,    0.0,    0.0,    self.initial_altitude   ), dtype=self.action_dtype)

        
        # control type
        if self.delta_control:
            self.control_low = self.control_delta_low
            self.control_high = self.control_delta_low
            #self.control_zero = self.control_delta_zero
            self.control_range = self.mapped_delta_range
            self.default_step = self.delta_step
        else:
            self.control_low = self.control_direct_low
            self.control_high = self.control_direct_high
            #self.control_zero = self.control_direct_zero
            self.control_range = self.mapped_direct_range
            self.default_step = self.direct_step

        # state space
        self.state_shape = (37,)
        self.state_space = gym.spaces.Box(shape=self.state_shape, low=-np.inf, high=np.inf, dtype=self.state_dtype)
        self.S = np.zeros(self.state_space.shape, dtype=self.state_space.dtype) # state vector
        
        # views Sensor
        
        # Inertial Measurement unit
        self.inertia =                          self.S[0:3]
        self.roll, self.pitch, self.yaw =       self.S[0:1], self.S[1:2], self.S[2:3]
        # gps location
        self.location =                         self.S[3:6]
        self.gX, self.gY, self.gZ =             self.S[3:4], self.S[4:5], self.S[5:6]
        # gyroscope
        self.orientation =                      self.S[6:9]
        self.rollA, self.pitchA, self.yawA =    self.S[6:7], self.S[7:8], self.S[8:9]
        # battery
        self.battery_level =     self.S[9:10]
        # now, 3+3+3+1 = 10

        # distance (4+4+2 = 10)
        self.LAND = self.S[10:11]
        self.SKY = self.S[11:12]
        self.RLS = self.S[12:13]
        self.RLF = self.S[13:14]
        self.FLS = self.S[14:15]
        self.FLF = self.S[15:16]
        self.RRS = self.S[16:17]
        self.RRF = self.S[17:18]
        self.FRS = self.S[18:19]
        self.FRF = self.S[19:20]
        # TOTAL SENSORS 
        self.SENSORS = self.S[0:20]


        # actuators
        self.ACTUATORS = self.S[20:24]
        self.roll_disturbance= self.S[20:21]
        self.pitch_disturbance=self.S[21:22]
        self.yaw_disturbance=self.S[22:23]
        self.target_altitude=self.S[23:24]
        

        # motor velocities
        self.motor_velocities = self.S[24:28]
        self.FLv, self.FRv, self. RLv, self.RRv = self.S[24:25], self.S[25:26], self.S[26:27], self.S[27:28]

        # current velocity
        self.base_velocity = self.S[28:34]
        self.linear_velocity = self.S[28:31]
        self.angular_velocity = self.S[31:34]

        # target location
        self.target_point = self.S[34:37]
        self.tx, self.ty, self.tz = self.S[34:35], self.S[35:36], self.S[36:37]

        
        
        # observation
        self.observation_shape = self.state_shape
        self.observation_space = gym.spaces.Box(shape=self.observation_shape, low=-np.inf, high=np.inf, dtype=self.state_dtype)
        self.observation = self.S[:]
    
        # action
        self.action_shape = (len(self.ACTUATORS),) # controls          roll,  pitch,   yaw,  altitude
        self.action_space = gym.spaces.Box(shape=self.action_shape, low=-1, high=1,  dtype=self.action_dtype)
        self.action_range = (self.action_space.low, self.action_space.high)
        self.action_mapper = REMAP(Input_Range=self.action_range, Mapped_Range= self.control_range)
        
        self.MAX_MOTOR_VELOCITY = np.zeros_like(self.motor_velocities) + 576
        self.MIN_MOTOR_VELOCITY = np.zeros_like(self.motor_velocities) - 576

        # velocities are available in supervisor mode 
        self.VELOCITY_LIMIT = np.array( [ 0.0 for _ in range(6) ] ) + 100
    
    def build_devices(self):
        self.imu = self.getDevice('imu')
        self.gps = self.getDevice('gps')
        self.gyro = self.getDevice('gyro')

        self.led_land = self.getDevice('led_DSEN_LAND')
        self.dLAND = self.getDevice('DSEN_LAND')
        self.dSKY = self.getDevice('DSEN_SKY')
        
        self.dRLS = self.getDevice('DSEN_RLS')
        self.dRLF = self.getDevice('DSEN_RLF')
        self.dFLS = self.getDevice('DSEN_FLS')
        self.dFLF = self.getDevice('DSEN_FLF')

        self.dRRS = self.getDevice('DSEN_RRS')
        self.dRRF = self.getDevice('DSEN_RRF')
        self.dFRS = self.getDevice('DSEN_FRS')
        self.dFRF = self.getDevice('DSEN_FRF')

        # get motors 
        self.front_left_motor = self.getDevice('flp')
        self.front_right_motor = self.getDevice('frp')
        self.rear_left_motor = self.getDevice('rlp')
        self.rear_right_motor = self.getDevice('rrp')
        self.motors = (self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor)

        # enable devices ----------------------------------
        self.imu.enable(self.timestep)
        self.gps.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.dLAND.enable(self.timestep)
        self.dSKY.enable(self.timestep)
                
        self.dRLS.enable(self.timestep)
        self.dRLF.enable(self.timestep)
        self.dFLS.enable(self.timestep)
        self.dFLF.enable(self.timestep)

        self.dRRS.enable(self.timestep)
        self.dRRF.enable(self.timestep)
        self.dFRS.enable(self.timestep)
        self.dFRF.enable(self.timestep)
        
        # enable battery ----------------------------------
        # <--- note: battery is not a 'device' its just a 'field' with 3 values - Current Level, Max level, Charging Rate
        self.batterySensorEnable(self.battery_scan_time) #<---- argument is the battery sampling period in ms
        # initialize motor position and velocity (max 576)
        for m in self.motors:
            m.setPosition(inf)
            m.setVelocity(0.0)

    def scan(self):
        # reads the state from sensor data
        # either use a build in state or return a new state from readings
        self.inertia[:] =       self.imu.getRollPitchYaw()
        self.location[:] =      self.gps.getValues()
        self.orientation[:] =   self.gyro.getValues()
        self.battery_level[:] = self.batterySensorGetValue()
        self.LAND[:], self.SKY[:], \
        self.RLS[:], self.RLF[:], \
        self.FLS[:], self.FLF[:], \
        self.RRS[:], self.RRF[:], \
        self.FRS[:], self.FRF[:] = \
        self.dLAND.getValue(), self.dSKY.getValue(), \
        self.dRLS.getValue(), self.dRLF.getValue(), \
        self.dFLS.getValue(), self.dFLF.getValue(), \
        self.dRRS.getValue(), self.dRRF.getValue(), \
        self.dFRS.getValue(), self.dFRF.getValue()
        self.base_velocity[:] = self.getSelf().getVelocity()
        self.base_velocity[:] = np.abs(self.base_velocity)
        return # self.observation

    def is_terminal(self):
        # checks if the sensors read an invalid state 
        if (self.base_velocity > self.VELOCITY_LIMIT).any():
            #self.getSelf().resetPhysics()
            return True, f"Rouge Velocity :: {self.base_velocity}"
        if self.battery_level<self.battery_min_val:
            return True, f"Battery Empty :: {self.battery_level}"
        if self.gZ<=0.0657594:
            if (self.roll < -0.7 or self.roll > 0.7 or self.pitch < -0.7 or self.pitch > 0.7):
                return True, f"Crashed :: {self.inertia}" # :: GYRO({self.orientation}), IMU:({self.inertia})
        return False, ""

    def _step(self):
        roll_input =    50.0  *   np.clip(self.roll, -1.0, 1.0) +   self.rollA +    self.roll_disturbance
        pitch_input =    30.0  *  np.clip(self.pitch, -1.0, 1.0) +  self.pitchA +   self.pitch_disturbance
        vertical_thrust = 68.5 +    (3.0 * (np.clip(self.target_altitude - self.gZ + 0.6, -1.0, 1.0)**3)) # first tem is base thrust
        self.motor_velocities[0] = (vertical_thrust  - roll_input + pitch_input - self.yaw_disturbance) # front left
        self.motor_velocities[1] = -(vertical_thrust  + roll_input + pitch_input + self.yaw_disturbance) # front right
        self.motor_velocities[2] = -(vertical_thrust  - roll_input - pitch_input + self.yaw_disturbance) # rear left
        self.motor_velocities[3] = (vertical_thrust  + roll_input - pitch_input - self.yaw_disturbance) # rear right
        self.motor_velocities = np.clip(self.motor_velocities, self.MIN_MOTOR_VELOCITY, self.MAX_MOTOR_VELOCITY)
        for m, v in zip(self.motors, self.motor_velocities):
            m.setVelocity(v)
        return (self.step(self.timestep) == -1)

    def delta_step(self, action):
        # write actuator inputs
        self.ACTUATORS[:] += self.action_mapper.in2map(action) # roll pitch, yaw, alt
        self.ACTUATORS[:] = np.clip(self.ACTUATORS[:], self.control_direct_low, self.control_direct_high)
        return self._step()

    def direct_step(self, action):
        # write actuator inputs
        self.ACTUATORS[:] = self.action_mapper.in2map(action) # roll pitch, yaw, alt
        return self._step()


    def take_action(self):
        # read self.observation
        return self.action_space.sample()

    def run(self, verbose=False):
        # -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **
        # robot setup
        # -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **

        # initialize Robot() instance
        super().__init__()

        # superbot checker - this indicates the superbot that i am still busy (control not exit)
        self.getSelf().getField('controllerArgs').insertMFString(-1, '') #<-- insert a blank string, this will be checked by superbot

        # determine timestep
        self.timestep = int( self.getBasicTimeStep() ) #* self.time_step_multiplier
        robot_name = self.getName()
        #msg = lambda text : print(f'[{self.robot_name} @ ({now()})]::[{text}]')
        msg = ((lambda text : print(f'({robot_name}): {text}')  ) if verbose else (lambda text : None ))
        self.step(self.timestep)

        # build devices
        self.build_devices()
        self.step(self.timestep) 
        #self.getSelf().getField('controllerArgs').setMFString(-1, '')  # set this field to blank to indicate superbot
        
        # start feedback loop
        msg('~ Start Bot')
        # note down initial position - will reset to this position  # self.getSelf().resetPhysics()
        irotation = self.getSelf().getField('rotation').getSFRotation()
        itranslation = self.getSelf().getField('translation').getSFVec3f()

        replay = MEM(capacity=1_000, observation_space=self.observation_space, action_space=self.action_space, seed=None)
        for episode in range(self.episodes):
            msg('------> episode:[{}]'.format(episode+1))
            # -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **
            # boot up sequence 
            # -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **
            msg('------> Booting up...')
            # 1 - (initial waiting) - specially for slow sensors - like battery
            waiting = self.getTime()
            while self.step(self.timestep) != -1:
                if (self.getTime()-waiting > self.initial_wait_time):
                    break #print('...initial wait over')

            msg('------> Launching')
            self.led_land.set(1)
            self.scan() #<--- we have stepped 
            self.ACTUATORS[:] = self.default_actuators

            # 3 - (ready the robot)
            is_ready = (self.gZ[0]>=self.initial_altitude) #<---- ready condition
            while not is_ready:
                self._step()
                self.scan()
                is_ready = (self.gZ[0]>=self.initial_altitude)
            
            # 4 - (record initial information, declare this as initial state)
            #self.initial_gps_location = -1*np.array(self.gps.getValues())       
            self.led_land.set(0)
            
            msg('------> Ready')
            # -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **
            # feedback loop
            # -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **
            timeout = self._step() # <---- first step (already scanned - when ready=True) 
            ts, done = -1, False
            terminate = (timeout or done)
            # feedback loop: step simulation until receiving an exit event
            #print('Starting feedback loop')
            while not (terminate):
                ts+=1        
                self.scan() #<--- this updates self.observation         
                if ts>=self.horizon:
                    done, reason= True, f"End of Time :: {ts} of {self.horizon}"
                else:
                    done, reason = self.is_terminal() #<---- this checks self.observation (actually self.state)
                if not done: # process behavior
                    action = self.take_action() #self.pie.predict(self.observation)
                    timeout = self.default_step(action) # write actuators inputs
                else: # robot finds itself in terminal state, cannot take action
                    action = self.action_space.sample()*0
                    timeout = False
                    msg('[!] ' + reason)
                terminate = (timeout or done)
                replay.snap(not(terminate), **{observation_key:self.observation, action_key:action, done_key:done, step_key:ts})
                # cS, A, done(cS), ts
                # nS, 0, done(nS)
                # 

            msg('------> Terminating...')

            self.getSelf().resetPhysics()
            for m in self.motors:
                m.setVelocity(0.0)
            waiting = self.getTime()
            while self.step(self.timestep) != -1:
                if ((self.getTime()-waiting ) > self.final_wait_time):
                    break #print('Killed Motors')
            self.getSelf().getField('rotation').setSFRotation(irotation)
            self.getSelf().getField('translation').setSFVec3f(itranslation)
            msg('------> Terminated.')

        msg('~ Stop Bot')
        msg(f'Replay Memeory :: Len: {replay.length()},  Count:{replay.count()},  Cap:{replay.capacity}')
        self.getSelf().getField('controllerArgs').setMFString(-1, '_')

        return  #cS, A, R






"""ARCHIVE:
    #self.simulationSetMode(0)
    #self.simulationReset()
    #for bot in self.bots:
    #    bot.restartController()

    
    #for c,a in zip(self.botsctrl, self.botsctrlargs):
    #    a.setMFString( 0, '--time_step_multiplier=1')
    #    a.setMFString( 1, '--delta_control=1')
    #    c.setSFString('mavrl')
        #for b,m in zip(self.bots,self.members):
        #    b.saveState( m + '_STATE' )
        #for c in self.botsctrl:
        #    c.setSFString('') #<--- this will cause timeout on robots
"""


